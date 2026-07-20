#include "grid_pair_2d.hpp"
#include "corner_patch_2d.hpp"
#include "p2_curve_2d.hpp"
#include "p2_projection_2d.hpp"
#include <array>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <deque>
#include <limits>
#include <algorithm>
#include <memory>
#include <stdexcept>
#include <unordered_map>
#include <utility>
#include <vector>

namespace kfbim {

using Point2  = std::array<double, 2>;
using CPoly2  = std::vector<Point2>;
using ProfileClock2D = std::chrono::steady_clock;

// ============================================================================
// Internal helpers
// ============================================================================

// x-coordinate of the first DOF along each axis for a given layout.
static std::array<double, 2> dof_first(const CartesianGrid2D& g) {
    auto o = g.origin(), h = g.spacing();
    switch (g.layout()) {
    case DofLayout2D::Node:       return {o[0],             o[1]            };
    case DofLayout2D::CellCenter: return {o[0] + 0.5*h[0],  o[1] + 0.5*h[1] };
    case DofLayout2D::FaceX:      return {o[0],             o[1] + 0.5*h[1] };
    case DofLayout2D::FaceY:      return {o[0] + 0.5*h[0],  o[1]            };
    }
    return {o[0], o[1]};
}

// Nearest bulk-node flat index to physical position (x, y).
static int nearest_node(const CartesianGrid2D& g, double x, double y) {
    auto f = dof_first(g);
    auto h = g.spacing();
    auto d = g.dof_dims();
    int nx = d[0], ny = d[1];
    int i = static_cast<int>(std::lround((x - f[0]) / h[0]));
    int j = static_cast<int>(std::lround((y - f[1]) / h[1]));
    i = std::max(0, std::min(nx - 1, i));
    j = std::max(0, std::min(ny - 1, j));
    return j * nx + i;
}

static Point2 midpoint(Point2 a, Point2 b) {
    return {0.5 * (a[0] + b[0]), 0.5 * (a[1] + b[1])};
}

static bool profile_grid_pair_2d()
{
    return std::getenv("KFBIM_PROFILE_GRID_PAIR_2D") != nullptr
        || std::getenv("KFBIM_PROFILE_INTERFACE_2D") != nullptr;
}

static bool profile_grid_pair_2d_detail()
{
    return std::getenv("KFBIM_PROFILE_GRID_PAIR_2D_DETAIL") != nullptr;
}

static double seconds_since(ProfileClock2D::time_point start)
{
    return std::chrono::duration<double>(ProfileClock2D::now() - start).count();
}

static double squared_distance(Point2 a, Point2 b)
{
    const double dx = a[0] - b[0];
    const double dy = a[1] - b[1];
    return dx * dx + dy * dy;
}

enum class PolygonSide2D {
    Outside,
    Boundary,
    Inside
};

static double orient2d(Point2 a, Point2 b, Point2 c)
{
    return (b[0] - a[0]) * (c[1] - a[1])
         - (b[1] - a[1]) * (c[0] - a[0]);
}

static bool point_on_segment(Point2 a, Point2 b, Point2 p)
{
    constexpr double kTol = 1.0e-12;
    if (std::abs(orient2d(a, b, p)) > kTol)
        return false;
    return p[0] >= std::min(a[0], b[0]) - kTol
        && p[0] <= std::max(a[0], b[0]) + kTol
        && p[1] >= std::min(a[1], b[1]) - kTol
        && p[1] <= std::max(a[1], b[1]) + kTol;
}

static PolygonSide2D polygon_bounded_side(const CPoly2& poly, Point2 p)
{
    if (poly.size() < 3)
        return PolygonSide2D::Outside;

    bool inside = false;
    for (std::size_t i = 0, j = poly.size() - 1; i < poly.size(); j = i++) {
        const Point2 a = poly[j];
        const Point2 b = poly[i];
        if (point_on_segment(a, b, p))
            return PolygonSide2D::Boundary;
        const bool crosses =
            (a[1] > p[1]) != (b[1] > p[1]);
        if (!crosses)
            continue;
        const double x_intersect =
            a[0] + (p[1] - a[1]) * (b[0] - a[0]) / (b[1] - a[1]);
        if (x_intersect > p[0])
            inside = !inside;
    }
    return inside ? PolygonSide2D::Inside : PolygonSide2D::Outside;
}

static bool is_better_nearest_candidate(double dist2,
                                        int    idx,
                                        double best_dist2,
                                        int    best_idx)
{
    constexpr double kTieTol = 1.0e-14;
    return dist2 < best_dist2 - kTieTol
        || (std::abs(dist2 - best_dist2) <= kTieTol
            && (best_idx < 0 || idx < best_idx));
}

static const double* quadratic_nodes_for_layout(PanelNodeLayout2D layout)
{
    static constexpr double gl_nodes[3] = {
        -0.77459666924148337704,
         0.0,
         0.77459666924148337704
    };
    static constexpr double lobatto_nodes[3] = {-1.0, 0.0, 1.0};

    switch (layout) {
    case PanelNodeLayout2D::LegacyGaussLegendre:
        return gl_nodes;
    case PanelNodeLayout2D::QuadraticLagrange:
        return lobatto_nodes;
    case PanelNodeLayout2D::Raw:
        return nullptr;
    }
    return nullptr;
}

static Point2 normalize_or(Point2 v, Point2 fallback)
{
    const double len = std::sqrt(v[0] * v[0] + v[1] * v[1]);
    if (len > 1.0e-14)
        return {v[0] / len, v[1] / len};
    return fallback;
}

static void push_cgal_point(CPoly2& poly, Point2 p) {
    poly.push_back(p);
}

static Point2 quadratic_panel_point(const Interface2D& iface,
                                    int                panel,
                                    const double       nodes[3],
                                    double             s)
{
    const double s0 = nodes[0];
    const double s1 = nodes[1];
    const double s2 = nodes[2];

    const double l0 = ((s - s1) * (s - s2)) / ((s0 - s1) * (s0 - s2));
    const double l1 = ((s - s0) * (s - s2)) / ((s1 - s0) * (s1 - s2));
    const double l2 = ((s - s0) * (s - s1)) / ((s2 - s0) * (s2 - s1));

    const int q0 = iface.point_index(panel, 0);
    const int q1 = iface.point_index(panel, 1);
    const int q2 = iface.point_index(panel, 2);

    return {
        l0 * iface.points()(q0, 0) + l1 * iface.points()(q1, 0) + l2 * iface.points()(q2, 0),
        l0 * iface.points()(q0, 1) + l1 * iface.points()(q1, 1) + l2 * iface.points()(q2, 1)
    };
}

static Point2 quadratic_panel_derivative(const Interface2D& iface,
                                         int                panel,
                                         const double       nodes[3],
                                         double             s)
{
    const double s0 = nodes[0];
    const double s1 = nodes[1];
    const double s2 = nodes[2];

    const double dl0 = (2.0 * s - s1 - s2) / ((s0 - s1) * (s0 - s2));
    const double dl1 = (2.0 * s - s0 - s2) / ((s1 - s0) * (s1 - s2));
    const double dl2 = (2.0 * s - s0 - s1) / ((s2 - s0) * (s2 - s1));

    const int q0 = iface.point_index(panel, 0);
    const int q1 = iface.point_index(panel, 1);
    const int q2 = iface.point_index(panel, 2);

    return {
        dl0 * iface.points()(q0, 0) + dl1 * iface.points()(q1, 0) + dl2 * iface.points()(q2, 0),
        dl0 * iface.points()(q0, 1) + dl1 * iface.points()(q1, 1) + dl2 * iface.points()(q2, 1)
    };
}

static Point2 quadratic_panel_normal(const Interface2D& iface,
                                     int                panel,
                                     const double       nodes[3],
                                     double             s)
{
    const Point2 deriv = quadratic_panel_derivative(iface, panel, nodes, s);
    Point2 normal = normalize_or({deriv[1], -deriv[0]}, {0.0, 0.0});

    const double s0 = nodes[0];
    const double s1 = nodes[1];
    const double s2 = nodes[2];
    const double l0 = ((s - s1) * (s - s2)) / ((s0 - s1) * (s0 - s2));
    const double l1 = ((s - s0) * (s - s2)) / ((s1 - s0) * (s1 - s2));
    const double l2 = ((s - s0) * (s - s1)) / ((s2 - s0) * (s2 - s1));

    const int q0 = iface.point_index(panel, 0);
    const int q1 = iface.point_index(panel, 1);
    const int q2 = iface.point_index(panel, 2);
    const Point2 ref = normalize_or({
        l0 * iface.normals()(q0, 0) + l1 * iface.normals()(q1, 0) + l2 * iface.normals()(q2, 0),
        l0 * iface.normals()(q0, 1) + l1 * iface.normals()(q1, 1) + l2 * iface.normals()(q2, 1)
    }, {iface.normals()(q1, 0), iface.normals()(q1, 1)});

    if (normal[0] * ref[0] + normal[1] * ref[1] < 0.0)
        normal = {-normal[0], -normal[1]};
    return normal;
}

struct DistanceSample2D {
    Point2 point;
    Point2 normal;
    int    component;
};

static std::vector<int> point_components_2d(const Interface2D& iface)
{
    std::vector<int> components(iface.num_points(), 0);
    for (int p = 0; p < iface.num_panels(); ++p) {
        for (int q = 0; q < iface.points_per_panel(); ++q)
            components[iface.point_index(p, q)] = iface.panel_components()(p);
    }
    return components;
}

static bool use_normal_domain_labels(const Interface2D& iface)
{
    return iface.points_per_panel() == 3
        && quadratic_nodes_for_layout(iface.panel_node_layout()) != nullptr;
}

static bool use_halfgrid_lshape_domain_labels()
{
    return std::getenv("KFBIM_GRID_PAIR_HALFGRID_LSHAPE_LABELS") != nullptr;
}

static bool skip_corner_patch_polygon_side_check()
{
    return std::getenv("KFBIM_GRID_PAIR_SKIP_PATCH_SIDE_POLYGON_CHECK") != nullptr
        || use_halfgrid_lshape_domain_labels();
}

static bool build_halfgrid_lshape_domain_labels(
    const CartesianGrid2D& grid,
    const Interface2D& iface,
    std::vector<int>& labels)
{
    std::vector<double> xs;
    std::vector<double> ys;
    for (int q = 0; q < iface.num_points(); ++q) {
        if (!iface.is_corner_point(q))
            continue;
        xs.push_back(iface.points()(q, 0));
        ys.push_back(iface.points()(q, 1));
    }
    auto unique_sorted = [](std::vector<double> values) {
        std::sort(values.begin(), values.end());
        std::vector<double> out;
        for (double value : values) {
            if (out.empty() || std::abs(value - out.back()) > 1.0e-12)
                out.push_back(value);
        }
        return out;
    };
    xs = unique_sorted(std::move(xs));
    ys = unique_sorted(std::move(ys));
    if (xs.size() != 3 || ys.size() != 3)
        return false;

    const auto dims = grid.dof_dims();
    const int nx = dims[0];
    const int ny = dims[1];
    if (static_cast<int>(labels.size()) != nx * ny)
        labels.assign(static_cast<std::size_t>(nx * ny), 0);
    constexpr double tol = 1.0e-12;
    for (int n = 0; n < nx * ny; ++n) {
        const auto c = grid.coord(n);
        const double x = c[0];
        const double y = c[1];
        const bool lower_arm = x >= xs[0] - tol && x <= xs[2] + tol
                            && y >= ys[0] - tol && y <= ys[1] + tol;
        const bool upper_arm = x >= xs[0] - tol && x <= xs[1] + tol
                            && y >= ys[1] - tol && y <= ys[2] + tol;
        labels[static_cast<std::size_t>(n)] =
            (lower_arm || upper_arm) ? 1 : 0;
    }
    return true;
}

static std::vector<DistanceSample2D> distance_samples_2d(const Interface2D& iface)
{
    const bool add_panel_samples = use_normal_domain_labels(iface);
    std::vector<DistanceSample2D> samples;
    samples.reserve(static_cast<std::size_t>(iface.num_points())
                    + (add_panel_samples ? geometry2d::kP2CenterS.size()
                                           * static_cast<std::size_t>(iface.num_panels())
                                         : 0));

    const std::vector<int> point_components = point_components_2d(iface);
    for (int q = 0; q < iface.num_points(); ++q) {
        samples.push_back({{iface.points()(q, 0), iface.points()(q, 1)},
                           {iface.normals()(q, 0), iface.normals()(q, 1)},
                           point_components[q]});
    }

    if (!add_panel_samples)
        return samples;

    const double* nodes = quadratic_nodes_for_layout(iface.panel_node_layout());
    for (int p = 0; p < iface.num_panels(); ++p) {
        for (double s : geometry2d::kP2CenterS) {
            if (iface.panel_node_layout() == PanelNodeLayout2D::QuadraticLagrange) {
                const Eigen::Vector2d pt = geometry2d::panel_point(iface, p, s);
                const Eigen::Vector2d normal = geometry2d::panel_normal(iface, p, s);
                samples.push_back({{pt[0], pt[1]},
                                   {normal[0], normal[1]},
                                   iface.panel_components()(p)});
            } else {
                samples.push_back({quadratic_panel_point(iface, p, nodes, s),
                                   quadratic_panel_normal(iface, p, nodes, s),
                                   iface.panel_components()(p)});
            }
        }
    }
    return samples;
}

static std::vector<DistanceSample2D> p2_expansion_center_samples_2d(
    const Interface2D& iface)
{
    if (!geometry2d::is_quadratic_lagrange_panel_layout(iface))
        return {};

    std::vector<DistanceSample2D> samples;
    samples.reserve(geometry2d::kP2CenterS.size()
                    * static_cast<std::size_t>(iface.num_panels()));
    for (int p = 0; p < iface.num_panels(); ++p) {
        for (double s : geometry2d::kP2CenterS) {
            const Eigen::Vector2d pt = geometry2d::panel_point(iface, p, s);
            const Eigen::Vector2d normal = geometry2d::panel_normal(iface, p, s);
            samples.push_back({{pt[0], pt[1]},
                               {normal[0], normal[1]},
                               iface.panel_components()(p)});
        }
    }
    return samples;
}

static bool has_closed_p2_components(const Interface2D& iface)
{
    if (!geometry2d::is_quadratic_lagrange_panel_layout(iface))
        return false;

    const int Nc = iface.num_components();
    std::vector<std::vector<int>> panels_by_component(Nc);
    for (int p = 0; p < iface.num_panels(); ++p)
        panels_by_component[iface.panel_components()(p)].push_back(p);

    for (const auto& panels : panels_by_component) {
        if (panels.empty())
            continue;
        for (int i = 0; i < static_cast<int>(panels.size()); ++i) {
            const int panel = panels[i];
            const int next_panel =
                panels[(i + 1) % static_cast<int>(panels.size())];
            if (iface.point_index(panel, 2) == iface.point_index(next_panel, 0))
                continue;
            bool shares_point = false;
            for (int a = 0; a < iface.points_per_panel(); ++a) {
                for (int b = 0; b < iface.points_per_panel(); ++b) {
                    shares_point = shares_point
                        || iface.point_index(panel, a)
                               == iface.point_index(next_panel, b);
                }
            }
            if (shares_point)
                continue;
            const bool connected_by_corner =
                std::any_of(iface.corners().begin(),
                            iface.corners().end(),
                            [panel, next_panel](const CornerData2D& corner) {
                                return corner.prev_panel == panel
                                    && corner.next_panel == next_panel;
                            });
            if (!connected_by_corner)
                return false;
        }
    }
    return true;
}

static bool is_outer_boundary_node(const CartesianGrid2D& grid, int idx)
{
    const auto dims = grid.dof_dims();
    const int nx = dims[0];
    const int ny = dims[1];
    const int i = idx % nx;
    const int j = idx / nx;
    return i == 0 || i == nx - 1 || j == 0 || j == ny - 1;
}

struct NearestResult2D {
    int    index = -1;
    double dist2 = std::numeric_limits<double>::infinity();
};

static NearestResult2D brute_force_nearest_point(
    const std::vector<Point2>& points,
    Point2                     query)
{
    NearestResult2D result;
    for (int idx = 0; idx < static_cast<int>(points.size()); ++idx) {
        const double dist2 = squared_distance(query, points[idx]);
        if (is_better_nearest_candidate(dist2, idx, result.dist2, result.index)) {
            result.index = idx;
            result.dist2 = dist2;
        }
    }
    return result;
}

struct HashCell2D {
    int i = 0;
    int j = 0;

    bool operator==(const HashCell2D& other) const
    {
        return i == other.i && j == other.j;
    }
};

struct HashCell2DHasher {
    std::size_t operator()(HashCell2D cell) const
    {
        const std::size_t a = static_cast<std::size_t>(
            static_cast<unsigned int>(cell.i));
        const std::size_t b = static_cast<std::size_t>(
            static_cast<unsigned int>(cell.j));
        return a * 73856093u ^ b * 19349663u;
    }
};

class PointSpatialHash2D {
public:
    PointSpatialHash2D(std::vector<Point2> points, double cell_size)
        : points_(std::move(points)),
          cell_size_(std::max(cell_size, 1.0e-300))
    {
        if (points_.empty())
            throw std::invalid_argument("PointSpatialHash2D requires at least one point");
        for (int idx = 0; idx < static_cast<int>(points_.size()); ++idx)
            cells_[cell_for(points_[idx])].push_back(idx);
    }

    NearestResult2D nearest(Point2 query) const
    {
        const HashCell2D base = cell_for(query);
        NearestResult2D best;
        constexpr int kFallbackRing = 64;

        for (int ring = 0; ring <= kFallbackRing; ++ring) {
            visit_ring(base, ring, [&](int idx) {
                const double dist2 = squared_distance(query, points_[idx]);
                if (is_better_nearest_candidate(dist2, idx, best.dist2, best.index)) {
                    best.index = idx;
                    best.dist2 = dist2;
                }
            });

            if (best.index >= 0
                && best.dist2 < min_unsearched_dist2(query, base, ring) - 1.0e-14) {
                return best;
            }
        }

        return brute_force_nearest_point(points_, query);
    }

private:
    HashCell2D cell_for(Point2 p) const
    {
        return {static_cast<int>(std::floor(p[0] / cell_size_)),
                static_cast<int>(std::floor(p[1] / cell_size_))};
    }

    template <typename Func>
    void visit_ring(HashCell2D base, int ring, Func&& fn) const
    {
        for (int dj = -ring; dj <= ring; ++dj) {
            for (int di = -ring; di <= ring; ++di) {
                if (std::max(std::abs(di), std::abs(dj)) != ring)
                    continue;
                const auto it = cells_.find({base.i + di, base.j + dj});
                if (it == cells_.end())
                    continue;
                for (int idx : it->second)
                    fn(idx);
            }
        }
    }

    double min_unsearched_dist2(Point2 query, HashCell2D base, int ring) const
    {
        const double min_x = static_cast<double>(base.i - ring) * cell_size_;
        const double max_x = static_cast<double>(base.i + ring + 1) * cell_size_;
        const double min_y = static_cast<double>(base.j - ring) * cell_size_;
        const double max_y = static_cast<double>(base.j + ring + 1) * cell_size_;
        const double dx = std::min(query[0] - min_x, max_x - query[0]);
        const double dy = std::min(query[1] - min_y, max_y - query[1]);
        const double d = std::max(0.0, std::min(dx, dy));
        return d * d;
    }

    std::vector<Point2> points_;
    double cell_size_;
    std::unordered_map<HashCell2D, std::vector<int>, HashCell2DHasher> cells_;
};

class NearestPointCloud2D {
public:
    explicit NearestPointCloud2D(const std::vector<Point2>& points)
        : raw_points_(points)
    {
        if (points.empty())
            throw std::invalid_argument("NearestPointCloud2D requires at least one point");
    }

    int nearest(Point2 query) const
    {
        return brute_force_nearest_point(raw_points_, query).index;
    }

    int nearest_stable(Point2 query) const
    {
        return nearest(query);
    }

private:
    std::vector<Point2> raw_points_;
};

static std::vector<int> build_p2_bfs_domain_labels(
    const CartesianGrid2D&              grid,
    const std::vector<DistanceSample2D>& center_samples,
    const std::vector<int>&              nearest_center_for_node,
    const std::vector<double>&           center_distance_for_node,
    const PointSpatialHash2D&            center_hash)
{
    const int N = grid.num_dofs();
    if (center_samples.empty()
        || static_cast<int>(nearest_center_for_node.size()) != N
        || static_cast<int>(center_distance_for_node.size()) != N) {
        throw std::invalid_argument(
            "GridPair2D P2 BFS labels require a nearest-center entry per grid node");
    }

    int max_component = 0;
    for (const auto& sample : center_samples)
        max_component = std::max(max_component, sample.component);

    const double hmax = std::max(grid.spacing()[0], grid.spacing()[1]);
    const double near_band = 3.0 * std::sqrt(2.0) * hmax;
    constexpr double kBoundaryTol = 1.0e-14;

    std::vector<unsigned char> near_node(N, 0);
    std::vector<int> seed_label(N, -1);
    for (int n = 0; n < N; ++n) {
        const int center_idx = nearest_center_for_node[n];
        if (center_idx < 0 || center_distance_for_node[n] > near_band)
            continue;

        const auto c = grid.coord(n);
        const Point2 pt{c[0], c[1]};
        const DistanceSample2D& center = center_samples[center_idx];
        const double dx = pt[0] - center.point[0];
        const double dy = pt[1] - center.point[1];

        near_node[n] = 1;
        const double signed_normal_distance =
            dx * center.normal[0] + dy * center.normal[1];
        seed_label[n] = (signed_normal_distance <= kBoundaryTol)
            ? center.component + 1
            : 0;
    }

    const auto edge_blocked = [&](int a, int b) {
        return near_node[a] && near_node[b]
            && seed_label[a] >= 0
            && seed_label[b] >= 0
            && seed_label[a] != seed_label[b];
    };

    std::vector<int> labels(N, -1);
    std::deque<int> queue;
    for (int n = 0; n < N; ++n) {
        if (is_outer_boundary_node(grid, n)) {
            labels[n] = 0;
            queue.push_back(n);
        }
    }

    while (!queue.empty()) {
        const int n = queue.front();
        queue.pop_front();
        for (int nb : grid.neighbors(n)) {
            if (nb < 0 || labels[nb] >= 0 || edge_blocked(n, nb))
                continue;
            labels[nb] = 0;
            queue.push_back(nb);
        }
    }

    for (int start = 0; start < N; ++start) {
        if (labels[start] >= 0)
            continue;

        std::vector<int> component_nodes;
        std::vector<int> label_counts(max_component + 2, 0);
        queue.push_back(start);
        labels[start] = -2;
        while (!queue.empty()) {
            const int n = queue.front();
            queue.pop_front();
            component_nodes.push_back(n);
            if (near_node[n] && seed_label[n] > 0
                && seed_label[n] < static_cast<int>(label_counts.size())) {
                ++label_counts[seed_label[n]];
            }

            for (int nb : grid.neighbors(n)) {
                if (nb < 0 || labels[nb] != -1 || edge_blocked(n, nb))
                    continue;
                labels[nb] = -2;
                queue.push_back(nb);
            }
        }

        int component_label = 0;
        int component_count = -1;
        for (int label = 1; label < static_cast<int>(label_counts.size()); ++label) {
            if (label_counts[label] > component_count) {
                component_count = label_counts[label];
                component_label = label;
            }
        }
        if (component_label <= 0 || component_count <= 0) {
            int center_idx = nearest_center_for_node[start];
            if (center_idx < 0) {
                const auto c = grid.coord(start);
                center_idx = center_hash.nearest({c[0], c[1]}).index;
            }
            component_label = center_samples[center_idx].component + 1;
        }

        for (int n : component_nodes)
            labels[n] = component_label;
    }

    return labels;
}

static std::vector<CPoly2> build_raw_label_polygons(const Interface2D& iface) {
    const int k = iface.points_per_panel();
    const int Np = iface.num_panels();
    const int Nc = iface.num_components();

    std::vector<CPoly2> polys(Nc);
    for (int p = 0; p < Np; ++p) {
        int c = iface.panel_components()(p);
        for (int qi = 0; qi < k; ++qi) {
            int idx = iface.point_index(p, qi);
            polys[c].push_back(
                {iface.points()(idx, 0), iface.points()(idx, 1)});
        }
    }
    return polys;
}

static std::vector<CPoly2> build_label_polygons(const Interface2D& iface) {
    const int k = iface.points_per_panel();
    if (k != 3)
        return build_raw_label_polygons(iface);

    constexpr double gl_nodes[3] = {
        -0.77459666924148337704,
         0.0,
         0.77459666924148337704
    };
    constexpr double lobatto_nodes[3] = {-1.0, 0.0, 1.0};

    const double* nodes = nullptr;
    switch (iface.panel_node_layout()) {
    case PanelNodeLayout2D::LegacyGaussLegendre:
        nodes = gl_nodes;
        break;
    case PanelNodeLayout2D::QuadraticLagrange:
        nodes = lobatto_nodes;
        break;
    case PanelNodeLayout2D::Raw:
        return build_raw_label_polygons(iface);
    }

    const int Np = iface.num_panels();
    const int Nc = iface.num_components();
    constexpr std::array<double, 4> internal_s = {-0.6, -0.2, 0.2, 0.6};

    std::vector<std::vector<int>> panels_by_component(Nc);
    for (int p = 0; p < Np; ++p)
        panels_by_component[iface.panel_components()(p)].push_back(p);

    std::vector<CPoly2> polys(Nc);
    for (int comp = 0; comp < Nc; ++comp) {
        const auto& panels = panels_by_component[comp];
        if (panels.empty())
            continue;

        std::vector<Point2> shared_starts(panels.size());
        for (int i = 0; i < static_cast<int>(panels.size()); ++i) {
            const int prev_panel = panels[(i + static_cast<int>(panels.size()) - 1)
                                          % static_cast<int>(panels.size())];
            const int curr_panel = panels[i];
            shared_starts[i] = midpoint(quadratic_panel_point(iface, prev_panel, nodes, 1.0),
                                        quadratic_panel_point(iface, curr_panel, nodes, -1.0));
        }

        for (int i = 0; i < static_cast<int>(panels.size()); ++i) {
            const int panel = panels[i];
            push_cgal_point(polys[comp], shared_starts[i]);
            for (double s : internal_s)
                push_cgal_point(polys[comp], quadratic_panel_point(iface, panel, nodes, s));
        }
    }

    return polys;
}

// ============================================================================
// Pimpl
// ============================================================================
static void rasterize_nearest_centers_to_grid(
    const CartesianGrid2D&              grid,
    const std::vector<DistanceSample2D>& centers,
    double                              radius,
    std::vector<int>&                   nearest_center_for_node,
    std::vector<double>&                distance_for_node);

struct GridPair2D::Impl {
    std::vector<int>    closest_bulk_node;  // [iface_pt] → grid flat index
    std::vector<int>    closest_iface_pt;   // [bulk_node] → iface pt index, lazy
    std::vector<double> min_iface_dist;     // P2: nearest center; legacy: nearest geometry sample
    std::vector<int>    domain_label_vec;   // [bulk_node] → 0=exterior, c+1=interior of component c
    std::vector<int>    patch_label_vec;    // [bulk_node] → -1=no patch, otherwise corner patch id
    std::vector<double> patch_radius_dist_vec; // [bulk_node] → |x-center|-R for patch nodes, inf otherwise
    std::unique_ptr<NearestPointCloud2D> iface_cloud;
    std::vector<DistanceSample2D> p2_center_samples;
    std::unique_ptr<PointSpatialHash2D> p2_center_hash;
    std::vector<int> nearest_p2_center_for_node;
    double p2_center_cache_radius = -1.0;

    void ensure_p2_center_cache_radius(const CartesianGrid2D& grid,
                                       double radius)
    {
        if (p2_center_samples.empty())
            return;
        if (radius <= p2_center_cache_radius + 1.0e-14)
            return;

        rasterize_nearest_centers_to_grid(grid,
                                          p2_center_samples,
                                          radius,
                                          nearest_p2_center_for_node,
                                          min_iface_dist);
        p2_center_cache_radius = radius;
    }
};

static std::vector<Point2> sample_points_for_hash(
    const std::vector<DistanceSample2D>& samples)
{
    std::vector<Point2> points;
    points.reserve(samples.size());
    for (const auto& sample : samples)
        points.push_back(sample.point);
    return points;
}

static void rasterize_nearest_centers_to_grid(
    const CartesianGrid2D&              grid,
    const std::vector<DistanceSample2D>& centers,
    double                              radius,
    std::vector<int>&                   nearest_center_for_node,
    std::vector<double>&                distance_for_node)
{
    const int N = grid.num_dofs();
    nearest_center_for_node.assign(N, -1);
    distance_for_node.assign(N, std::numeric_limits<double>::infinity());
    if (centers.empty() || radius < 0.0)
        return;

    std::vector<double> best_dist2(N, std::numeric_limits<double>::infinity());
    const auto first = dof_first(grid);
    const auto h = grid.spacing();
    const auto dims = grid.dof_dims();
    const double radius2 = radius * radius;
    constexpr double kRadiusTol = 1.0e-14;

    for (int cidx = 0; cidx < static_cast<int>(centers.size()); ++cidx) {
        const Point2 center = centers[cidx].point;
        int i_min = static_cast<int>(std::floor((center[0] - radius - first[0]) / h[0])) - 1;
        int i_max = static_cast<int>(std::ceil((center[0] + radius - first[0]) / h[0])) + 1;
        int j_min = static_cast<int>(std::floor((center[1] - radius - first[1]) / h[1])) - 1;
        int j_max = static_cast<int>(std::ceil((center[1] + radius - first[1]) / h[1])) + 1;
        i_min = std::max(0, i_min);
        j_min = std::max(0, j_min);
        i_max = std::min(dims[0] - 1, i_max);
        j_max = std::min(dims[1] - 1, j_max);

        for (int j = j_min; j <= j_max; ++j) {
            for (int i = i_min; i <= i_max; ++i) {
                const Point2 pt{first[0] + static_cast<double>(i) * h[0],
                                first[1] + static_cast<double>(j) * h[1]};
                const double dist2 = squared_distance(pt, center);
                if (dist2 > radius2 + kRadiusTol)
                    continue;
                const int node = grid.index(i, j);
                if (is_better_nearest_candidate(
                        dist2, cidx, best_dist2[node], nearest_center_for_node[node])) {
                    best_dist2[node] = dist2;
                    nearest_center_for_node[node] = cidx;
                }
            }
        }
    }

    for (int n = 0; n < N; ++n) {
        if (nearest_center_for_node[n] >= 0)
            distance_for_node[n] = std::sqrt(best_dist2[n]);
    }
}

static void validate_corner_patches_inside_grid(const CartesianGrid2D& grid,
                                                const Interface2D& iface)
{
    const auto origin = grid.origin();
    const auto h = grid.spacing();
    const auto cells = grid.num_cells();
    const double xmin = origin[0];
    const double ymin = origin[1];
    const double xmax = origin[0] + static_cast<double>(cells[0]) * h[0];
    const double ymax = origin[1] + static_cast<double>(cells[1]) * h[1];
    constexpr double tol = 1.0e-14;

    for (int p = 0; p < static_cast<int>(iface.corner_patches().size()); ++p) {
        const CornerPatch2D& patch = iface.corner_patches()[p];
        if (patch.center[0] - patch.radius <= xmin + tol
            || patch.center[0] + patch.radius >= xmax - tol
            || patch.center[1] - patch.radius <= ymin + tol
            || patch.center[1] + patch.radius >= ymax - tol) {
            throw std::invalid_argument(
                "corner patch " + std::to_string(p)
                + " reaches the Cartesian outer boundary");
        }
    }
}

static void build_corner_patch_labels(const CartesianGrid2D& grid,
                                      const Interface2D& iface,
                                      const std::vector<int>& domain_labels,
                                      std::vector<int>& labels,
                                      std::vector<double>& radius_dist)
{
    const int N = grid.num_dofs();
    if (static_cast<int>(domain_labels.size()) != N)
        throw std::invalid_argument(
            "corner patch labels require domain labels for every grid node");
    labels.assign(N, -1);
    radius_dist.assign(N, std::numeric_limits<double>::infinity());
    if (iface.corner_patches().empty())
        return;

    validate_corner_patches_inside_grid(grid, iface);
    if (!skip_corner_patch_polygon_side_check()) {
        const std::vector<CPoly2> label_polys = build_label_polygons(iface);
        for (int p = 0; p < static_cast<int>(iface.corner_patches().size()); ++p) {
            const CornerPatch2D& patch = iface.corner_patches()[p];
            if (patch.shape == CornerPatchShape2D::Circle)
                continue;
            const Eigen::Vector2d ray0 = patch.ray_minus.normalized();
            const double c = std::cos(0.5 * patch.opening_angle);
            const double s = std::sin(0.5 * patch.opening_angle);
            const Eigen::Vector2d bisector(
                c * ray0[0] - s * ray0[1],
                s * ray0[0] + c * ray0[1]);
            const Eigen::Vector2d sample =
                patch.center + 0.5 * patch.radius * bisector;
            const Point2 sample_point{sample[0], sample[1]};
            bool sample_is_interior = false;
            for (const CPoly2& poly : label_polys) {
                if (poly.size() < 3)
                    continue;
                const PolygonSide2D side =
                    polygon_bounded_side(poly, sample_point);
                if (side == PolygonSide2D::Inside
                    || side == PolygonSide2D::Boundary) {
                    sample_is_interior = true;
                    break;
                }
            }
            const bool patch_is_interior =
                patch.side == CornerPatchSide2D::Interior;
            if (sample_is_interior != patch_is_interior) {
                throw std::invalid_argument(
                    "corner patch side disagrees with GridPair2D domain labels");
            }
        }
    }

    for (int n = 0; n < N; ++n) {
        const auto c = grid.coord(n);
        const Eigen::Vector2d x(c[0], c[1]);
        for (int p = 0; p < static_cast<int>(iface.corner_patches().size()); ++p) {
            const CornerPatch2D& patch = iface.corner_patches()[p];
            if (!corner_patch_contains_point_2d(patch, x))
                continue;
            if (labels[n] >= 0) {
                throw std::invalid_argument(
                    "grid node belongs to overlapping corner patches");
            }
            labels[n] = p;
            radius_dist[n] =
                corner_patch_radial_signed_distance_2d(patch, x);
        }
    }
}

// ============================================================================
// Constructor
// ============================================================================
GridPair2D::GridPair2D(const CartesianGrid2D& grid, const Interface2D& iface)
    : grid_(grid), interface_(iface), impl_(std::make_unique<Impl>())
{
    const bool profile = profile_grid_pair_2d();
    const bool detail_profile = profile_grid_pair_2d_detail();
    const ProfileClock2D::time_point t_total_start = ProfileClock2D::now();
    const int Nq = iface.num_points();
    const int N  = grid.num_dofs();
    const int Nc = iface.num_components();

    // ------------------------------------------------------------------
    // 1. closest_bulk_node[q]: nearest grid DOF to each interface point.
    //    Computed analytically from grid layout — O(Nq).
    // ------------------------------------------------------------------
    impl_->closest_bulk_node.resize(Nq);
    for (int q = 0; q < Nq; ++q)
        impl_->closest_bulk_node[q] =
            nearest_node(grid, iface.points()(q, 0), iface.points()(q, 1));
    const double t_closest_bulk = seconds_since(t_total_start);

    // ------------------------------------------------------------------
    // 2. min_iface_dist[n] and optional P2 center lookup cache.
    //
    //    closest_interface_point(n) remains a DOF-index compatibility query,
    //    but its kd-tree and per-node entries are built lazily on first use.
    //    Active P2 transfer paths use only nearest expansion centers.
    // ------------------------------------------------------------------
    const ProfileClock2D::time_point t_sample_start = ProfileClock2D::now();
    impl_->p2_center_samples = p2_expansion_center_samples_2d(iface);
    impl_->closest_iface_pt.resize(N, -1);
    impl_->min_iface_dist.resize(N, std::numeric_limits<double>::infinity());
    const bool active_p2 = !impl_->p2_center_samples.empty();
    std::vector<DistanceSample2D> distance_samples;
    std::vector<int> closest_sample_idx;
    double t_sample_setup = 0.0;
    double t_sample_nearest = 0.0;
    double t_center_cache = 0.0;

    if (active_p2) {
        impl_->p2_center_hash = std::make_unique<PointSpatialHash2D>(
            sample_points_for_hash(impl_->p2_center_samples),
            std::max(grid.spacing()[0], grid.spacing()[1]));
        impl_->nearest_p2_center_for_node.resize(N, -1);
        t_sample_setup = seconds_since(t_sample_start);

        const double default_center_band =
            3.0 * std::sqrt(2.0) * std::max(grid.spacing()[0], grid.spacing()[1]);
        const ProfileClock2D::time_point t_center_cache_start =
            ProfileClock2D::now();
        impl_->ensure_p2_center_cache_radius(grid, default_center_band);
        t_center_cache = seconds_since(t_center_cache_start);
    } else {
        distance_samples = distance_samples_2d(iface);
        const NearestPointCloud2D sample_cloud(sample_points_for_hash(distance_samples));
        t_sample_setup = seconds_since(t_sample_start);

        closest_sample_idx.assign(N, 0);
        const ProfileClock2D::time_point t_sample_nearest_start =
            ProfileClock2D::now();
        for (int n = 0; n < N; ++n) {
            const auto c = grid.coord(n);
            const Point2 pt{c[0], c[1]};

            const int sidx = sample_cloud.nearest(pt);
            impl_->min_iface_dist[n] =
                std::sqrt(squared_distance(pt, distance_samples[sidx].point));
            closest_sample_idx[n] = sidx;
        }
        t_sample_nearest = seconds_since(t_sample_nearest_start);
    }
    // ------------------------------------------------------------------
    // 3. domain_label_vec[n]: 0=exterior, comp+1=interior of component comp.
    //
    // Closed active P2 panel layouts use center-sign seeds plus flood fill.
    // Open or non-P2 curved panels keep the older nearest-sample normal labels,
    // and raw layouts keep the polygon fallback.
    // ------------------------------------------------------------------

    impl_->domain_label_vec.resize(N, 0);
    const bool p2_bfs_labels =
        has_closed_p2_components(iface) && !impl_->p2_center_samples.empty();
    const bool normal_labels = use_normal_domain_labels(iface);
    const bool analytic_lshape_labels =
        use_halfgrid_lshape_domain_labels()
        && build_halfgrid_lshape_domain_labels(grid,
                                               iface,
                                               impl_->domain_label_vec);
    double t_polygon = 0.0;
    std::size_t polygon_vertices = 0;
    const ProfileClock2D::time_point t_label_start = ProfileClock2D::now();
    int label_queries = 0;
    int label_inside = 0;
    int label_boundary = 0;
    int label_outside = 0;
    double t_label_coord = 0.0;
    double t_label_query = 0.0;
    double t_label_assign = 0.0;

    if (analytic_lshape_labels) {
        label_queries = N;
    } else if (p2_bfs_labels) {
        impl_->domain_label_vec = build_p2_bfs_domain_labels(
            grid,
            impl_->p2_center_samples,
            impl_->nearest_p2_center_for_node,
            impl_->min_iface_dist,
            *impl_->p2_center_hash);
        label_queries = N;
    } else if (normal_labels) {
        constexpr double kBoundaryTol = 1.0e-14;
        for (int n = 0; n < N; ++n) {
            const ProfileClock2D::time_point t_coord_start = ProfileClock2D::now();
            const auto c = grid.coord(n);
            const Point2 pt{c[0], c[1]};
            if (detail_profile)
                t_label_coord += seconds_since(t_coord_start);

            ++label_queries;
            const ProfileClock2D::time_point t_query_start = ProfileClock2D::now();
            const DistanceSample2D* sample = nullptr;
            if (active_p2) {
                int center_idx = impl_->nearest_p2_center_for_node[n];
                if (center_idx < 0) {
                    center_idx = impl_->p2_center_hash->nearest(pt).index;
                    impl_->nearest_p2_center_for_node[n] = center_idx;
                    impl_->min_iface_dist[n] =
                        std::sqrt(squared_distance(
                            pt, impl_->p2_center_samples[center_idx].point));
                }
                sample = &impl_->p2_center_samples[center_idx];
            } else {
                sample = &distance_samples[closest_sample_idx[n]];
            }
            const double signed_normal_distance =
                (pt[0] - sample->point[0]) * sample->normal[0]
                + (pt[1] - sample->point[1]) * sample->normal[1];
            if (detail_profile)
                t_label_query += seconds_since(t_query_start);

            const ProfileClock2D::time_point t_assign_start = ProfileClock2D::now();
            if (detail_profile) {
                if (std::abs(signed_normal_distance) <= kBoundaryTol)
                    ++label_boundary;
                else if (signed_normal_distance < 0.0)
                    ++label_inside;
                else
                    ++label_outside;
            }
            if (signed_normal_distance <= kBoundaryTol)
                impl_->domain_label_vec[n] = sample->component + 1;
            if (detail_profile)
                t_label_assign += seconds_since(t_assign_start);
        }
    } else {
        const ProfileClock2D::time_point t_polygon_start = ProfileClock2D::now();
        std::vector<CPoly2> polys = build_label_polygons(iface);
        t_polygon = seconds_since(t_polygon_start);
        for (const auto& poly : polys)
            polygon_vertices += poly.size();

        for (int n = 0; n < N; ++n) {
            const ProfileClock2D::time_point t_coord_start = ProfileClock2D::now();
            auto    c = grid.coord(n);
            Point2 p{c[0], c[1]};
            if (detail_profile)
                t_label_coord += seconds_since(t_coord_start);
            for (int comp = 0; comp < Nc; ++comp) {
                if (polys[comp].size() < 3)
                    continue;
                ++label_queries;
                const ProfileClock2D::time_point t_query_start = ProfileClock2D::now();
                const PolygonSide2D side =
                    polygon_bounded_side(polys[comp], p);
                if (detail_profile)
                    t_label_query += seconds_since(t_query_start);
                const ProfileClock2D::time_point t_assign_start = ProfileClock2D::now();
                if (detail_profile) {
                    if (side == PolygonSide2D::Inside)
                        ++label_inside;
                    else if (side == PolygonSide2D::Boundary)
                        ++label_boundary;
                    else
                        ++label_outside;
                }
                if (side == PolygonSide2D::Inside
                    || side == PolygonSide2D::Boundary) {
                    impl_->domain_label_vec[n] = comp + 1;
                    if (detail_profile)
                        t_label_assign += seconds_since(t_assign_start);
                    break;
                }
                if (detail_profile)
                    t_label_assign += seconds_since(t_assign_start);
            }
        }
    }
    const double t_label = seconds_since(t_label_start);
    build_corner_patch_labels(grid,
                              iface,
                              impl_->domain_label_vec,
                              impl_->patch_label_vec,
                              impl_->patch_radius_dist_vec);

    if (profile) {
        const double total = seconds_since(t_total_start);
        const std::size_t sample_count =
            active_p2 ? impl_->p2_center_samples.size() : distance_samples.size();
        std::printf("      GridPair2D Ngrid=%d Niface=%d samples=%zu components=%d label_mode=%s polygon_vertices=%zu label_queries=%d\n",
                    N, Nq, sample_count, Nc,
                    analytic_lshape_labels
                        ? "analytic-halfgrid-lshape"
                        : (p2_bfs_labels ? "p2-center-bfs"
                                         : (normal_labels ? "nearest-normal" : "polygon")),
                    polygon_vertices, label_queries);
        std::printf("        closest_bulk %.6fs sample_setup %.6fs sample_nearest %.6fs center_cache %.6fs build_polygons %.6fs domain_labels %.6fs total %.6fs\n",
                    t_closest_bulk, t_sample_setup, t_sample_nearest,
                    t_center_cache, t_polygon, t_label, total);
    }
    if (detail_profile) {
        std::printf("        domain_label_detail coord_point %.6fs classify_query %.6fs assign %.6fs inside=%d boundary=%d outside=%d\n",
                    t_label_coord, t_label_query, t_label_assign,
                    label_inside, label_boundary, label_outside);
    }
}

GridPair2D::~GridPair2D() = default;

// ============================================================================
// Query methods
// ============================================================================
int GridPair2D::closest_bulk_node(int q) const {
    return impl_->closest_bulk_node[q];
}

int GridPair2D::closest_interface_point(int n) const {
    if (impl_->closest_iface_pt[n] < 0) {
        if (!impl_->iface_cloud) {
            std::vector<Point2> iface_points;
            iface_points.reserve(interface_.num_points());
            for (int q = 0; q < interface_.num_points(); ++q) {
                iface_points.push_back({interface_.points()(q, 0),
                                        interface_.points()(q, 1)});
            }
            impl_->iface_cloud =
                std::make_unique<NearestPointCloud2D>(iface_points);
        }
        const auto c = grid_.coord(n);
        impl_->closest_iface_pt[n] =
            impl_->iface_cloud->nearest_stable({c[0], c[1]});
    }
    return impl_->closest_iface_pt[n];
}

int GridPair2D::nearest_p2_expansion_center(int n) const {
    if (impl_->nearest_p2_center_for_node.empty()) {
        throw std::runtime_error(
            "GridPair2D::nearest_p2_expansion_center requires P2 QuadraticLagrange panels");
    }
    if (impl_->nearest_p2_center_for_node[n] < 0) {
        const auto c = grid_.coord(n);
        const NearestResult2D nearest =
            impl_->p2_center_hash->nearest({c[0], c[1]});
        impl_->nearest_p2_center_for_node[n] = nearest.index;
        impl_->min_iface_dist[n] = std::sqrt(nearest.dist2);
    }
    return impl_->nearest_p2_center_for_node[n];
}

int GridPair2D::nearest_p2_expansion_center_between(int a, int b) const {
    if (impl_->p2_center_samples.empty()) {
        throw std::runtime_error(
            "GridPair2D::nearest_p2_expansion_center_between requires P2 QuadraticLagrange panels");
    }
    const auto ca = grid_.coord(a);
    const auto cb = grid_.coord(b);
    const double ax = ca[0];
    const double ay = ca[1];
    const double bx = cb[0];
    const double by = cb[1];

    int best = -1;
    double best_score = std::numeric_limits<double>::infinity();
    double best_mid_dist2 = std::numeric_limits<double>::infinity();
    const double mx = 0.5 * (ax + bx);
    const double my = 0.5 * (ay + by);
    for (int i = 0; i < static_cast<int>(impl_->p2_center_samples.size());
         ++i) {
        const Point2& p = impl_->p2_center_samples[static_cast<std::size_t>(i)].point;
        const double dax = p[0] - ax;
        const double day = p[1] - ay;
        const double dbx = p[0] - bx;
        const double dby = p[1] - by;
        const double score =
            std::sqrt(dax * dax + day * day)
            + std::sqrt(dbx * dbx + dby * dby);
        const double dmx = p[0] - mx;
        const double dmy = p[1] - my;
        const double mid_dist2 = dmx * dmx + dmy * dmy;
        if (score < best_score - 1.0e-14
            || (std::abs(score - best_score) <= 1.0e-14
                && mid_dist2 < best_mid_dist2)) {
            best = i;
            best_score = score;
            best_mid_dist2 = mid_dist2;
        }
    }
    return best;
}

int GridPair2D::domain_label(int n) const {
    return impl_->domain_label_vec[n];
}

int GridPair2D::corner_patch_label(int n) const {
    return impl_->patch_label_vec[n];
}

bool GridPair2D::is_in_corner_patch(int n) const {
    return corner_patch_label(n) >= 0;
}

double GridPair2D::corner_patch_radial_signed_distance(int n) const {
    return impl_->patch_radius_dist_vec[n];
}

bool GridPair2D::is_near_interface(int n, double radius) const {
    impl_->ensure_p2_center_cache_radius(grid_, radius);
    return impl_->min_iface_dist[n] < radius;
}

std::vector<int> GridPair2D::near_interface_nodes(double radius) const {
    impl_->ensure_p2_center_cache_radius(grid_, radius);
    int N = grid_.num_dofs();
    std::vector<int> result;
    result.reserve(N / 8);
    for (int n = 0; n < N; ++n)
        if (impl_->min_iface_dist[n] < radius)
            result.push_back(n);
    return result;
}

std::vector<int> GridPair2D::near_interface_points(int n, double radius) const {
    auto   c  = grid_.coord(n);
    double cx = c[0], cy = c[1];
    int Nq = interface_.num_points();
    std::vector<int> result;
    for (int q = 0; q < Nq; ++q) {
        double dx = cx - interface_.points()(q, 0);
        double dy = cy - interface_.points()(q, 1);
        if (std::sqrt(dx*dx + dy*dy) < radius)
            result.push_back(q);
    }
    return result;
}

NarrowBandProjection2D GridPair2D::project_near_interface_nodes(double radius) const
{
    return project_p2_near_interface_nodes_2d(*this, radius);
}

NarrowBandProjection2D GridPair2D::project_grid_nodes_to_interface(
    const std::vector<int>& bulk_node_indices) const
{
    return project_p2_grid_nodes_to_interface_2d(*this, bulk_node_indices);
}

} // namespace kfbim
