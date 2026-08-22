#include "grid_pair_3d.hpp"
#include "nurbs_cartesian_domain_3d.hpp"
#include "p2_surface_3d.hpp"
#include <CGAL/Exact_predicates_inexact_constructions_kernel.h>
#include <CGAL/Surface_mesh.h>
#include <CGAL/Side_of_triangle_mesh.h>
#include <CGAL/AABB_tree.h>
#include <CGAL/AABB_traits.h>
#include <CGAL/AABB_face_graph_triangle_primitive.h>
#include <CGAL/Search_traits_3.h>
#include <CGAL/Search_traits_adapter.h>
#include <CGAL/Orthogonal_k_neighbor_search.h>
#include <CGAL/property_map.h>
#include <CGAL/version.h>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <limits>
#include <algorithm>
#include <array>
#include <memory>
#include <stdexcept>
#include <tuple>
#include <unordered_map>
#include <utility>
#include <variant>
#include <boost/variant/get.hpp>

namespace kfbim {

using K3      = CGAL::Exact_predicates_inexact_constructions_kernel;
using CPoint3 = K3::Point_3;
using CMesh3  = CGAL::Surface_mesh<CPoint3>;
using CSideOf = CGAL::Side_of_triangle_mesh<CMesh3, K3>;
using CPointWithIndex3 = std::pair<CPoint3, int>;
using CSearchBaseTraits3 = CGAL::Search_traits_3<K3>;
using CSearchTraits3 =
    CGAL::Search_traits_adapter<CPointWithIndex3,
                                CGAL::First_of_pair_property_map<CPointWithIndex3>,
                                CSearchBaseTraits3>;
using CNeighborSearch3 = CGAL::Orthogonal_k_neighbor_search<CSearchTraits3>;
using CSearchTree3 = CNeighborSearch3::Tree;
using CSegment3 = K3::Segment_3;
using CFace3 = boost::graph_traits<CMesh3>::face_descriptor;
using CAabbPrimitive3 = CGAL::AABB_face_graph_triangle_primitive<CMesh3>;
using CAabbTraits3 = CGAL::AABB_traits<K3, CAabbPrimitive3>;
using CAabbTree3 = CGAL::AABB_tree<CAabbTraits3>;

// ============================================================================
// Internal helpers
// ============================================================================

// First-DOF coordinate per axis for a given 3D layout.
static std::array<double, 3> dof_first(const CartesianGrid3D& g) {
    auto o = g.origin(), h = g.spacing();
    switch (g.layout()) {
    case DofLayout3D::Node:
        return {o[0],            o[1],            o[2]           };
    case DofLayout3D::CellCenter:
        return {o[0]+0.5*h[0],   o[1]+0.5*h[1],   o[2]+0.5*h[2]  };
    case DofLayout3D::FaceX:
        return {o[0],            o[1]+0.5*h[1],   o[2]+0.5*h[2]  };
    case DofLayout3D::FaceY:
        return {o[0]+0.5*h[0],   o[1],            o[2]+0.5*h[2]  };
    case DofLayout3D::FaceZ:
        return {o[0]+0.5*h[0],   o[1]+0.5*h[1],   o[2]           };
    }
    return {o[0], o[1], o[2]};
}

// Nearest bulk-node flat index to physical position (x, y, z).
static int nearest_node(const CartesianGrid3D& g, double x, double y, double z) {
    auto f = dof_first(g);
    auto h = g.spacing();
    auto d = g.dof_dims();
    int nx = d[0], ny = d[1], nz = d[2];
    int i = static_cast<int>(std::lround((x - f[0]) / h[0]));
    int j = static_cast<int>(std::lround((y - f[1]) / h[1]));
    int k = static_cast<int>(std::lround((z - f[2]) / h[2]));
    i = std::max(0, std::min(nx - 1, i));
    j = std::max(0, std::min(ny - 1, j));
    k = std::max(0, std::min(nz - 1, k));
    return k * (nx * ny) + j * nx + i;
}

static bool profile_grid_pair_3d()
{
    return std::getenv("KFBIM_PROFILE_GRID_PAIR_3D") != nullptr
        || std::getenv("KFBIM_PROFILE_INTERFACE_3D") != nullptr;
}

using ProfileClock3D = std::chrono::steady_clock;

static double seconds_since(ProfileClock3D::time_point start)
{
    return std::chrono::duration<double>(ProfileClock3D::now() - start).count();
}

static double max_grid_spacing(const CartesianGrid3D& grid)
{
    const auto h = grid.spacing();
    return std::max(h[0], std::max(h[1], h[2]));
}

struct DistanceSample3D {
    Eigen::Vector3d point;
    Eigen::Vector3d normal;
    int             component;
    int             panel = -1;
    Eigen::Vector3d barycentric = Eigen::Vector3d::Zero();
    int             nurbs_patch = -1;
};

struct NearestResult3D {
    int    index = -1;
    double dist2 = std::numeric_limits<double>::infinity();
};

P2NativeEventOwnerCenterSelection3D
select_p2_native_event_owner_center_3d(
    const geometry3d::GridEdgeEvent3D& event,
    const std::vector<int>& patch_g1_components,
    const std::vector<P2NativeSheetCenterCandidate3D>& center_candidates,
    double distance_tie_tolerance)
{
    if (!std::isfinite(distance_tie_tolerance)
        || distance_tie_tolerance < 0.0) {
        throw std::invalid_argument(
            "native event owner-center tie tolerance must be finite and nonnegative");
    }
    if (!event.certified_transverse() || !event.point.allFinite()
        || event.owners.empty()) {
        throw std::invalid_argument(
            "native event owner-center selection requires a certified owned event");
    }
    if (patch_g1_components.empty() || center_candidates.empty()) {
        throw std::runtime_error(
            "native event owner-center selection has no G1 sheet metadata");
    }

    P2NativeEventOwnerCenterSelection3D best;
    double best_distance = std::numeric_limits<double>::infinity();
    double best_transversality = -1.0;
    int best_owner_patch = std::numeric_limits<int>::max();
    double best_owner_u = std::numeric_limits<double>::infinity();
    double best_owner_v = std::numeric_limits<double>::infinity();
    int best_center_index = std::numeric_limits<int>::max();

    for (int owner_index = 0;
         owner_index < static_cast<int>(event.owners.size()); ++owner_index) {
        const geometry3d::NurbsSurfaceRootOwner3D& owner =
            event.owners[static_cast<std::size_t>(owner_index)];
        if (owner.patch_index < 0
            || owner.patch_index
                   >= static_cast<int>(patch_g1_components.size())
            || !owner.point.allFinite()
            || !std::isfinite(owner.u) || !std::isfinite(owner.v)
            || !std::isfinite(owner.transversality)) {
            throw std::invalid_argument(
                "native event owner-center selection received an invalid owner");
        }
        const int owner_sheet = patch_g1_components[
            static_cast<std::size_t>(owner.patch_index)];
        for (int candidate_index = 0;
             candidate_index < static_cast<int>(center_candidates.size());
             ++candidate_index) {
            const P2NativeSheetCenterCandidate3D& center =
                center_candidates[static_cast<std::size_t>(candidate_index)];
            if (center.center_index < 0 || center.panel_index < 0
                || center.nurbs_patch_index < 0
                || center.nurbs_patch_index
                       >= static_cast<int>(patch_g1_components.size())
                || !center.point.allFinite()) {
                throw std::invalid_argument(
                    "native event owner-center selection received an invalid center");
            }
            if (patch_g1_components[static_cast<std::size_t>(
                    center.nurbs_patch_index)] != owner_sheet) {
                continue;
            }
            const double distance = (center.point - event.point).norm();
            const bool closer = distance
                < best_distance - distance_tie_tolerance;
            const bool tied = std::abs(distance - best_distance)
                <= distance_tie_tolerance;
            const bool more_transverse = tied
                && owner.transversality > best_transversality;
            const bool same_transversality = tied
                && owner.transversality == best_transversality;
            const bool stable_owner = same_transversality
                && std::tie(owner.patch_index, owner.u, owner.v,
                            center.center_index)
                     < std::tie(best_owner_patch, best_owner_u, best_owner_v,
                                best_center_index);
            if (!closer && !more_transverse && !stable_owner)
                continue;
            best = {owner_index, candidate_index};
            best_distance = distance;
            best_transversality = owner.transversality;
            best_owner_patch = owner.patch_index;
            best_owner_u = owner.u;
            best_owner_v = owner.v;
            best_center_index = center.center_index;
        }
    }
    if (best.owner_index < 0) {
        throw std::runtime_error(
            "native grid-edge event has no P2 expansion center on any owner G1 sheet");
    }
    return best;
}

static std::vector<int> point_components_3d(const Interface3D& iface)
{
    std::vector<int> components(iface.num_points(), 0);
    for (int p = 0; p < iface.num_panels(); ++p) {
        for (int q = 0; q < iface.points_per_panel(); ++q)
            components[iface.point_index(p, q)] = iface.panel_components()[p];
    }
    return components;
}

static std::vector<DistanceSample3D> distance_samples_3d(const Interface3D& iface)
{
    std::vector<DistanceSample3D> samples;
    samples.reserve(static_cast<std::size_t>(iface.num_points())
                    + 16 * static_cast<std::size_t>(iface.num_panels()));
    const std::vector<int> point_components = point_components_3d(iface);
    for (int q = 0; q < iface.num_points(); ++q) {
        samples.push_back({iface.points().row(q).transpose(),
                           iface.normals().row(q).transpose(),
                           point_components[q]});
    }

    if (iface.panel_node_layout() == PanelNodeLayout3D::QuadraticLagrange) {
        for (int p = 0; p < iface.num_panels(); ++p) {
            for (const auto& ref_tri : geometry3d::subdivided_reference_triangles()) {
                const std::array<Eigen::Vector3d, 3> tri = {
                    geometry3d::panel_point(iface, p, ref_tri.bary[0]),
                    geometry3d::panel_point(iface, p, ref_tri.bary[1]),
                    geometry3d::panel_point(iface, p, ref_tri.bary[2])};
                const Eigen::Vector3d center = (tri[0] + tri[1] + tri[2]) / 3.0;
                Eigen::Vector3d normal = (tri[1] - tri[0]).cross(tri[2] - tri[0]);
                const double len = normal.norm();
                if (len > 1.0e-14)
                    normal /= len;
                else
                    normal = iface.normals().row(iface.point_index(p, 0)).transpose();
                const Eigen::Vector3d ref =
                    iface.normals().row(iface.point_index(p, 0)).transpose();
                if (normal.dot(ref) < 0.0)
                    normal = -normal;
                samples.push_back({center, normal, iface.panel_components()[p]});
            }
        }
    }

    return samples;
}

static std::vector<DistanceSample3D> p2_expansion_center_samples_3d(
    const Interface3D& iface,
    const std::vector<int>& panel_nurbs_patches)
{
    if (iface.points_per_panel() != 6
        || iface.panel_node_layout() != PanelNodeLayout3D::QuadraticLagrange) {
        return {};
    }

    const std::vector<Eigen::Vector3d> center_bary =
        geometry3d::expansion_center_barycentrics();
    std::vector<DistanceSample3D> samples;
    samples.reserve(center_bary.size()
                    * static_cast<std::size_t>(iface.num_panels()));
    for (int p = 0; p < iface.num_panels(); ++p) {
        const int nurbs_patch = panel_nurbs_patches.empty()
            ? -1 : panel_nurbs_patches[static_cast<std::size_t>(p)];
        for (const Eigen::Vector3d& bary : center_bary) {
            samples.push_back({geometry3d::panel_point(iface, p, bary),
                               geometry3d::panel_oriented_normal(iface, p, bary),
                               iface.panel_components()[p],
                               p,
                               bary,
                               nurbs_patch});
        }
    }
    return samples;
}

static Eigen::Vector3d to_eigen(const CPoint3& point)
{
    return {CGAL::to_double(point.x()),
            CGAL::to_double(point.y()),
            CGAL::to_double(point.z())};
}

static Eigen::Vector3d triangle_barycentric(const Interface3D& iface,
                                            int panel,
                                            const Eigen::Vector3d& point)
{
    const Eigen::Vector3d a =
        iface.vertices().row(iface.panels()(panel, 0)).transpose();
    const Eigen::Vector3d b =
        iface.vertices().row(iface.panels()(panel, 1)).transpose();
    const Eigen::Vector3d c =
        iface.vertices().row(iface.panels()(panel, 2)).transpose();
    const Eigen::Vector3d e0 = b - a;
    const Eigen::Vector3d e1 = c - a;
    const Eigen::Vector3d displacement = point - a;
    const double d00 = e0.dot(e0);
    const double d01 = e0.dot(e1);
    const double d11 = e1.dot(e1);
    const double d20 = displacement.dot(e0);
    const double d21 = displacement.dot(e1);
    const double denominator = d00 * d11 - d01 * d01;
    if (!std::isfinite(denominator) || std::abs(denominator) <= 1.0e-28)
        throw std::runtime_error("crossing geometry contains a degenerate triangle");
    const double lambda1 = (d11 * d20 - d01 * d21) / denominator;
    const double lambda2 = (d00 * d21 - d01 * d20) / denominator;
    return {1.0 - lambda1 - lambda2, lambda1, lambda2};
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

static double squared_distance(Eigen::Vector3d a, Eigen::Vector3d b)
{
    return (a - b).squaredNorm();
}

static NearestResult3D brute_force_nearest_point(
    const std::vector<Eigen::Vector3d>& points,
    Eigen::Vector3d                     query)
{
    NearestResult3D result;
    for (int idx = 0; idx < static_cast<int>(points.size()); ++idx) {
        const double dist2 = squared_distance(query, points[idx]);
        if (is_better_nearest_candidate(dist2, idx, result.dist2, result.index)) {
            result.index = idx;
            result.dist2 = dist2;
        }
    }
    return result;
}

struct HashCell3D {
    int i = 0;
    int j = 0;
    int k = 0;

    bool operator==(const HashCell3D& other) const
    {
        return i == other.i && j == other.j && k == other.k;
    }
};

struct HashCell3DHasher {
    std::size_t operator()(HashCell3D cell) const
    {
        const std::size_t a = static_cast<std::size_t>(
            static_cast<unsigned int>(cell.i));
        const std::size_t b = static_cast<std::size_t>(
            static_cast<unsigned int>(cell.j));
        const std::size_t c = static_cast<std::size_t>(
            static_cast<unsigned int>(cell.k));
        return a * 73856093u ^ b * 19349663u ^ c * 83492791u;
    }
};

class PointSpatialHash3D {
public:
    PointSpatialHash3D(std::vector<Eigen::Vector3d> points, double cell_size)
        : points_(std::move(points))
        , cell_size_(std::max(cell_size, 1.0e-300))
    {
        if (points_.empty())
            throw std::invalid_argument("PointSpatialHash3D requires at least one point");
        for (int idx = 0; idx < static_cast<int>(points_.size()); ++idx)
            cells_[cell_for(points_[idx])].push_back(idx);
    }

    NearestResult3D nearest(Eigen::Vector3d query) const
    {
        const HashCell3D base = cell_for(query);
        NearestResult3D best;
        constexpr int kFallbackRing = 64;

        for (int ring = 0; ring <= kFallbackRing; ++ring) {
            visit_shell(base, ring, [&](int idx) {
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

    std::vector<int> within_radius(Eigen::Vector3d query, double radius) const
    {
        if (radius < 0.0)
            throw std::invalid_argument(
                "PointSpatialHash3D radius must be nonnegative");

        const HashCell3D base = cell_for(query);
        const int ring = static_cast<int>(std::ceil(radius / cell_size_)) + 1;
        const double radius2 = radius * radius;
        std::vector<int> result;
        for (int dk = -ring; dk <= ring; ++dk) {
            for (int dj = -ring; dj <= ring; ++dj) {
                for (int di = -ring; di <= ring; ++di) {
                    const auto it = cells_.find(
                        {base.i + di, base.j + dj, base.k + dk});
                    if (it == cells_.end())
                        continue;
                    for (int idx : it->second) {
                        if (squared_distance(query, points_[idx])
                            <= radius2 + 1.0e-14) {
                            result.push_back(idx);
                        }
                    }
                }
            }
        }
        std::sort(result.begin(), result.end());
        result.erase(std::unique(result.begin(), result.end()), result.end());
        return result;
    }

private:
    HashCell3D cell_for(Eigen::Vector3d p) const
    {
        return {static_cast<int>(std::floor(p[0] / cell_size_)),
                static_cast<int>(std::floor(p[1] / cell_size_)),
                static_cast<int>(std::floor(p[2] / cell_size_))};
    }

    template <typename Func>
    void visit_shell(HashCell3D base, int ring, Func&& fn) const
    {
        for (int dk = -ring; dk <= ring; ++dk) {
            for (int dj = -ring; dj <= ring; ++dj) {
                for (int di = -ring; di <= ring; ++di) {
                    if (std::max(std::abs(di),
                                 std::max(std::abs(dj), std::abs(dk))) != ring) {
                        continue;
                    }
                    const auto it =
                        cells_.find({base.i + di, base.j + dj, base.k + dk});
                    if (it == cells_.end())
                        continue;
                    for (int idx : it->second)
                        fn(idx);
                }
            }
        }
    }

    double min_unsearched_dist2(Eigen::Vector3d query,
                                HashCell3D     base,
                                int            ring) const
    {
        const double min_x = static_cast<double>(base.i - ring) * cell_size_;
        const double max_x = static_cast<double>(base.i + ring + 1) * cell_size_;
        const double min_y = static_cast<double>(base.j - ring) * cell_size_;
        const double max_y = static_cast<double>(base.j + ring + 1) * cell_size_;
        const double min_z = static_cast<double>(base.k - ring) * cell_size_;
        const double max_z = static_cast<double>(base.k + ring + 1) * cell_size_;
        const double dx = std::min(query[0] - min_x, max_x - query[0]);
        const double dy = std::min(query[1] - min_y, max_y - query[1]);
        const double dz = std::min(query[2] - min_z, max_z - query[2]);
        const double d = std::max(0.0, std::min(dx, std::min(dy, dz)));
        return d * d;
    }

    std::vector<Eigen::Vector3d> points_;
    double cell_size_;
    std::unordered_map<HashCell3D, std::vector<int>, HashCell3DHasher> cells_;
};

class NearestPointCloud3D {
public:
    explicit NearestPointCloud3D(const std::vector<Eigen::Vector3d>& points)
    {
        if (points.empty())
            throw std::invalid_argument("NearestPointCloud3D requires at least one point");

        points_.reserve(points.size());
        raw_points_ = points;
        for (int i = 0; i < static_cast<int>(points.size()); ++i) {
            const Eigen::Vector3d& p = points[i];
            points_.emplace_back(CPoint3(p[0], p[1], p[2]), i);
        }
        tree_ = std::make_unique<CSearchTree3>(points_.begin(), points_.end());
    }

    int nearest(Eigen::Vector3d query) const
    {
        CNeighborSearch3 search(*tree_, CPoint3(query[0], query[1], query[2]), 1);
        return search.begin()->first.second;
    }

    int nearest_stable(Eigen::Vector3d query) const
    {
        const int k = std::min(8, static_cast<int>(points_.size()));
        CNeighborSearch3 search(*tree_, CPoint3(query[0], query[1], query[2]), k);

        int best_idx = search.begin()->first.second;
        double best_dist2 = squared_distance(query, raw_points_[best_idx]);
        for (auto it = search.begin(); it != search.end(); ++it) {
            const int idx = it->first.second;
            const double dist2 = squared_distance(query, raw_points_[idx]);
            if (is_better_nearest_candidate(dist2, idx, best_dist2, best_idx)) {
                best_dist2 = dist2;
                best_idx = idx;
            }
        }
        return best_idx;
    }

    std::vector<int> nearest_k(Eigen::Vector3d query, int k) const
    {
        if (k <= 0)
            throw std::invalid_argument("NearestPointCloud3D nearest_k requires k > 0");
        k = std::min(k, static_cast<int>(points_.size()));

        CNeighborSearch3 search(*tree_, CPoint3(query[0], query[1], query[2]), k);
        std::vector<int> result;
        result.reserve(k);
        for (auto it = search.begin(); it != search.end(); ++it)
            result.push_back(it->first.second);
        return result;
    }

private:
    std::vector<Eigen::Vector3d> raw_points_;
    std::vector<CPointWithIndex3> points_;
    std::unique_ptr<CSearchTree3> tree_;
};

static std::vector<Eigen::Vector3d> sample_points_for_hash(
    const std::vector<DistanceSample3D>& samples)
{
    std::vector<Eigen::Vector3d> points;
    points.reserve(samples.size());
    for (const auto& sample : samples)
        points.push_back(sample.point);
    return points;
}

static void rasterize_nearest_centers_to_grid(
    const CartesianGrid3D&              grid,
    const std::vector<DistanceSample3D>& centers,
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

    const int nx = dims[0];
    const int ny = dims[1];
    const int nxy = nx * ny;

    for (int cidx = 0; cidx < static_cast<int>(centers.size()); ++cidx) {
        const Eigen::Vector3d center = centers[cidx].point;
        int i_min = static_cast<int>(std::floor((center[0] - radius - first[0]) / h[0])) - 1;
        int i_max = static_cast<int>(std::ceil((center[0] + radius - first[0]) / h[0])) + 1;
        int j_min = static_cast<int>(std::floor((center[1] - radius - first[1]) / h[1])) - 1;
        int j_max = static_cast<int>(std::ceil((center[1] + radius - first[1]) / h[1])) + 1;
        int k_min = static_cast<int>(std::floor((center[2] - radius - first[2]) / h[2])) - 1;
        int k_max = static_cast<int>(std::ceil((center[2] + radius - first[2]) / h[2])) + 1;
        i_min = std::max(0, i_min);
        j_min = std::max(0, j_min);
        k_min = std::max(0, k_min);
        i_max = std::min(dims[0] - 1, i_max);
        j_max = std::min(dims[1] - 1, j_max);
        k_max = std::min(dims[2] - 1, k_max);

        for (int k = k_min; k <= k_max; ++k) {
            const double z = first[2] + static_cast<double>(k) * h[2];
            const double dz = z - center[2];
            const double dz2 = dz * dz;
            for (int j = j_min; j <= j_max; ++j) {
                const double y = first[1] + static_cast<double>(j) * h[1];
                const double dy = y - center[1];
                const double dyz2 = dy * dy + dz2;
                const int row_offset = k * nxy + j * nx;
                for (int i = i_min; i <= i_max; ++i) {
                    const double x = first[0] + static_cast<double>(i) * h[0];
                    const double dx = x - center[0];
                    const double dist2 = dx * dx + dyz2;
                    if (dist2 > radius2 + kRadiusTol)
                        continue;
                    const int node = row_offset + i;
                    if (is_better_nearest_candidate(
                            dist2, cidx, best_dist2[node],
                            nearest_center_for_node[node])) {
                        best_dist2[node] = dist2;
                        nearest_center_for_node[node] = cidx;
                    }
                }
            }
        }
    }

    for (int n = 0; n < N; ++n) {
        if (nearest_center_for_node[n] >= 0)
            distance_for_node[n] = std::sqrt(best_dist2[n]);
    }
}

static std::vector<int> build_p2_nearest_center_domain_labels(
    const CartesianGrid3D&              grid,
    const std::vector<DistanceSample3D>& center_samples,
    const std::vector<int>&              nearest_center_for_node)
{
    const int N = grid.num_dofs();
    if (center_samples.empty()
        || static_cast<int>(nearest_center_for_node.size()) != N) {
        throw std::invalid_argument(
            "GridPair3D P2 labels require a nearest-center entry per grid node");
    }

    constexpr double kBoundaryTol = 1.0e-14;
    std::vector<int> labels(N, 0);
    for (int n = 0; n < N; ++n) {
        const int center_idx = nearest_center_for_node[n];
        if (center_idx < 0
            || center_idx >= static_cast<int>(center_samples.size())) {
            throw std::runtime_error(
                "GridPair3D P2 label has an invalid nearest center");
        }

        const auto c = grid.coord(n);
        const Eigen::Vector3d pt(c[0], c[1], c[2]);
        const DistanceSample3D& center = center_samples[center_idx];
        const double signed_normal_distance =
            (pt - center.point).dot(center.normal);
        labels[n] = (signed_normal_distance <= kBoundaryTol)
            ? center.component + 1
            : 0;
    }

    return labels;
}

static void verify_native_label_crossing_invariant(
    const CartesianGrid3D& grid,
    const std::vector<int>& labels,
    const geometry3d::NurbsCartesianDomain3D& domain)
{
    const auto dims = grid.dof_dims();
    for (int k = 0; k < dims[2]; ++k) {
        for (int j = 0; j < dims[1]; ++j) {
            for (int i = 0; i < dims[0]; ++i) {
                const int node = grid.index(i, j, k);
                for (int axis = 0; axis < 3; ++axis) {
                    std::array<int, 3> neighbor_index{{i, j, k}};
                    if (++neighbor_index[static_cast<std::size_t>(axis)]
                        >= dims[static_cast<std::size_t>(axis)]) {
                        continue;
                    }
                    const int neighbor = grid.index(
                        neighbor_index[0],
                        neighbor_index[1],
                        neighbor_index[2]);
                    if (labels[static_cast<std::size_t>(node)]
                        != labels[static_cast<std::size_t>(neighbor)]) {
                        (void)domain.correction_crossing_between(
                            node, neighbor);
                    }
                }
            }
        }
    }
}

// ============================================================================
// Pimpl
// ============================================================================
struct P2G1SheetCenterCloud3D {
    std::vector<int> global_center_indices;
    std::unique_ptr<NearestPointCloud3D> cloud;
};

struct GridPair3D::Impl {
    std::vector<int>    closest_bulk_node;
    std::vector<int>    closest_iface_pt;
    std::vector<double> min_iface_dist;
    std::vector<int>    domain_label_vec;
    std::unique_ptr<NearestPointCloud3D> iface_cloud;
    std::vector<DistanceSample3D> p2_center_samples;
    std::unique_ptr<PointSpatialHash3D> p2_center_hash;
    std::unique_ptr<NearestPointCloud3D> p2_center_cloud;
    std::unordered_map<int, P2G1SheetCenterCloud3D>
        p2_centers_by_g1_sheet;
    bool has_native_panel_patch_metadata = false;
    std::vector<int> nearest_p2_center_for_node;
    std::vector<int> owner_p2_center_for_point;
    double p2_center_cache_radius = -1.0;
    std::unique_ptr<CMesh3> crossing_mesh;
    std::unique_ptr<CAabbTree3> crossing_tree;
    std::vector<int> geometry_panel_for_face;

    void ensure_p2_center_cache_radius(const CartesianGrid3D& grid,
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

// ============================================================================
// Constructor
// ============================================================================
GridPair3D::GridPair3D(const CartesianGrid3D& grid, const Interface3D& iface)
    : GridPair3D(grid, iface, iface, nullptr)
{
}

GridPair3D::GridPair3D(const CartesianGrid3D& grid,
                       const Interface3D& iface,
                       const Interface3D& crossing_geometry)
    : GridPair3D(grid, iface, crossing_geometry, nullptr)
{
}

GridPair3D::GridPair3D(
    const CartesianGrid3D& grid,
    const Interface3D& iface,
    const Interface3D& crossing_geometry,
    std::shared_ptr<const geometry3d::NurbsCartesianDomain3D> nurbs_domain)
    : GridPair3D(
          grid, iface, crossing_geometry, std::move(nurbs_domain), {})
{
}

GridPair3D::GridPair3D(
    const CartesianGrid3D& grid,
    const Interface3D& iface,
    const Interface3D& crossing_geometry,
    std::shared_ptr<const geometry3d::NurbsCartesianDomain3D> nurbs_domain,
    std::vector<int> correction_panel_nurbs_patches)
    : grid_(grid)
    , interface_(iface)
    , crossing_geometry_(crossing_geometry)
    , nurbs_domain_(std::move(nurbs_domain))
    , impl_(std::make_unique<Impl>())
{
    if (nurbs_domain_ && !nurbs_domain_->is_compatible_grid(grid_)) {
        throw std::invalid_argument(
            "GridPair3D native NURBS domain grid does not match the supplied Cartesian grid");
    }
    if (!correction_panel_nurbs_patches.empty()) {
        if (!nurbs_domain_) {
            throw std::invalid_argument(
                "GridPair3D panel-to-NURBS metadata requires a native domain");
        }
        if (correction_panel_nurbs_patches.size()
            != static_cast<std::size_t>(iface.num_panels())) {
            throw std::invalid_argument(
                "GridPair3D panel-to-NURBS metadata does not match the correction panels");
        }
        const std::vector<int>& patch_g1_components =
            nurbs_domain_->patch_g1_components();
        std::unordered_map<int, int> g1_to_interface_group;
        std::unordered_map<int, int> interface_group_to_g1;
        for (int panel = 0; panel < iface.num_panels(); ++panel) {
            const int patch = correction_panel_nurbs_patches[
                static_cast<std::size_t>(panel)];
            if (patch < 0
                || patch >= static_cast<int>(patch_g1_components.size())) {
                throw std::invalid_argument(
                    "GridPair3D panel-to-NURBS metadata contains an invalid patch");
            }
            const int g1 = patch_g1_components[static_cast<std::size_t>(patch)];
            const int interface_group = iface.panel_smooth_groups()[panel];
            const auto first = g1_to_interface_group.emplace(
                g1, interface_group);
            const auto second = interface_group_to_g1.emplace(
                interface_group, g1);
            if ((!first.second && first.first->second != interface_group)
                || (!second.second && second.first->second != g1)) {
                throw std::invalid_argument(
                    "GridPair3D panel smooth groups disagree with native G1 topology");
            }
        }
        impl_->has_native_panel_patch_metadata = true;
    }
    const bool profile = profile_grid_pair_3d();
    const ProfileClock3D::time_point t_total_start = ProfileClock3D::now();
    const int Nq = iface.num_points();
    const int N  = grid.num_dofs();
    const int Np = iface.num_panels();
    const int Nc = iface.num_components();
    const bool active_p2 =
        iface.points_per_panel() == 6
        && iface.panel_node_layout() == PanelNodeLayout3D::QuadraticLagrange;
    const bool crossing_p2 =
        crossing_geometry.points_per_panel() == 6
        && crossing_geometry.panel_node_layout()
               == PanelNodeLayout3D::QuadraticLagrange;
    if (active_p2 && !crossing_p2) {
        throw std::invalid_argument(
            "GridPair3D P2 correction interface requires P2 crossing geometry");
    }
    if (nurbs_domain_
        && nurbs_domain_->labels().size() != static_cast<std::size_t>(N)) {
        throw std::invalid_argument(
            "GridPair3D native NURBS domain label count does not match the grid");
    }

    // ------------------------------------------------------------------
    // 1. closest_bulk_node[q]: nearest grid DOF to each interface point.
    // ------------------------------------------------------------------
    impl_->closest_bulk_node.resize(Nq);
    for (int q = 0; q < Nq; ++q)
        impl_->closest_bulk_node[q] = nearest_node(
            grid,
            iface.points()(q, 0), iface.points()(q, 1), iface.points()(q, 2));
    const double t_closest_bulk = seconds_since(t_total_start);

    // ------------------------------------------------------------------
    // 2. min_iface_dist[n] and optional P2 center lookup cache.
    //
    //    Active P2 paths use only expansion centers for geometry lookups.
    //    closest_interface_point(n) remains a lazy compatibility query.
    // ------------------------------------------------------------------
    const ProfileClock3D::time_point t_sample_start = ProfileClock3D::now();
    impl_->closest_iface_pt.resize(N, -1);
    impl_->min_iface_dist.resize(N, std::numeric_limits<double>::infinity());
    double t_sample_setup = 0.0;
    double t_sample_nearest = 0.0;
    double t_center_cache = 0.0;
    std::vector<DistanceSample3D> distance_samples;
    std::vector<int> closest_sample_idx;

    if (active_p2) {
        if (!nurbs_domain_) {
            impl_->crossing_mesh = std::make_unique<CMesh3>();
            std::vector<CMesh3::Vertex_index> mesh_vertices;
            mesh_vertices.reserve(crossing_geometry.num_vertices());
            for (int vertex = 0; vertex < crossing_geometry.num_vertices(); ++vertex) {
                mesh_vertices.push_back(impl_->crossing_mesh->add_vertex(CPoint3(
                    crossing_geometry.vertices()(vertex, 0),
                    crossing_geometry.vertices()(vertex, 1),
                    crossing_geometry.vertices()(vertex, 2))));
            }
            for (int panel = 0; panel < crossing_geometry.num_panels(); ++panel) {
                const CFace3 face = impl_->crossing_mesh->add_face(
                    mesh_vertices[static_cast<std::size_t>(
                        crossing_geometry.panels()(panel, 0))],
                    mesh_vertices[static_cast<std::size_t>(
                        crossing_geometry.panels()(panel, 1))],
                    mesh_vertices[static_cast<std::size_t>(
                        crossing_geometry.panels()(panel, 2))]);
                if (face == CMesh3::null_face()) {
                    throw std::runtime_error(
                        "failed to build the P2 crossing-geometry triangle mesh");
                }
                if (face.idx() >= impl_->geometry_panel_for_face.size()) {
                    impl_->geometry_panel_for_face.resize(
                        static_cast<std::size_t>(face.idx()) + 1, -1);
                }
                impl_->geometry_panel_for_face[face.idx()] = panel;
            }
            impl_->crossing_tree = std::make_unique<CAabbTree3>(
                faces(*impl_->crossing_mesh).first,
                faces(*impl_->crossing_mesh).second,
                *impl_->crossing_mesh);
            impl_->crossing_tree->accelerate_distance_queries();
        }

        impl_->p2_center_samples = p2_expansion_center_samples_3d(
            iface, correction_panel_nurbs_patches);
        const std::vector<Eigen::Vector3d> p2_center_points =
            sample_points_for_hash(impl_->p2_center_samples);
        impl_->p2_center_hash = std::make_unique<PointSpatialHash3D>(
            p2_center_points, max_grid_spacing(grid));
        impl_->p2_center_cloud = std::make_unique<NearestPointCloud3D>(
            p2_center_points);
        if (impl_->has_native_panel_patch_metadata) {
            const std::vector<int>& patch_g1_components =
                nurbs_domain_->patch_g1_components();
            for (int center = 0;
                 center < static_cast<int>(impl_->p2_center_samples.size());
                 ++center) {
                const int patch = impl_->p2_center_samples[
                    static_cast<std::size_t>(center)].nurbs_patch;
                const int g1 = patch_g1_components[
                    static_cast<std::size_t>(patch)];
                impl_->p2_centers_by_g1_sheet[g1].global_center_indices.
                    push_back(center);
            }
            for (auto& entry : impl_->p2_centers_by_g1_sheet) {
                std::vector<Eigen::Vector3d> points;
                points.reserve(entry.second.global_center_indices.size());
                for (int center : entry.second.global_center_indices) {
                    points.push_back(impl_->p2_center_samples[
                        static_cast<std::size_t>(center)].point);
                }
                entry.second.cloud =
                    std::make_unique<NearestPointCloud3D>(points);
            }
        }
        impl_->nearest_p2_center_for_node.resize(N, -1);
        impl_->owner_p2_center_for_point.resize(Nq, -1);
        std::vector<double> owner_center_dist2(
            static_cast<std::size_t>(Nq),
            std::numeric_limits<double>::infinity());
        const int centers_per_panel = static_cast<int>(
            geometry3d::expansion_center_barycentrics().size());
        for (int panel = 0; panel < Np; ++panel) {
            for (int local = 0; local < iface.points_per_panel(); ++local) {
                const int q = iface.point_index(panel, local);
                const auto anchor_coord =
                    grid.coord(impl_->closest_bulk_node[static_cast<std::size_t>(q)]);
                const Eigen::Vector3d anchor(
                    anchor_coord[0], anchor_coord[1], anchor_coord[2]);
                for (int center_local = 0;
                     center_local < centers_per_panel;
                     ++center_local) {
                    const int center_index =
                        centers_per_panel * panel + center_local;
                    const double dist2 =
                        (impl_->p2_center_samples[
                             static_cast<std::size_t>(center_index)].point
                         - anchor).squaredNorm();
                    const int previous_center = impl_->owner_p2_center_for_point[
                        static_cast<std::size_t>(q)];
                    if (is_better_nearest_candidate(
                            dist2,
                            center_index,
                            owner_center_dist2[static_cast<std::size_t>(q)],
                            previous_center)) {
                        owner_center_dist2[static_cast<std::size_t>(q)] = dist2;
                        impl_->owner_p2_center_for_point[
                            static_cast<std::size_t>(q)] = center_index;
                    }
                }
            }
        }
        t_sample_setup = seconds_since(t_sample_start);

        const double default_center_band =
            3.0 * std::sqrt(3.0) * max_grid_spacing(grid);
        const ProfileClock3D::time_point t_center_cache_start =
            ProfileClock3D::now();
        impl_->ensure_p2_center_cache_radius(grid, default_center_band);
        for (int n = 0; n < N; ++n) {
            if (impl_->nearest_p2_center_for_node[n] >= 0)
                continue;
            const auto c = grid.coord(n);
            const Eigen::Vector3d point(c[0], c[1], c[2]);
            const int nearest = impl_->p2_center_cloud->nearest_stable(point);
            impl_->nearest_p2_center_for_node[n] = nearest;
            impl_->min_iface_dist[n] = std::sqrt(squared_distance(
                point,
                impl_->p2_center_samples[
                    static_cast<std::size_t>(nearest)].point));
        }
        t_center_cache = seconds_since(t_center_cache_start);
    } else {
        distance_samples = distance_samples_3d(iface);
        const NearestPointCloud3D sample_cloud(sample_points_for_hash(distance_samples));
        t_sample_setup = seconds_since(t_sample_start);

        closest_sample_idx.assign(N, 0);
        impl_->closest_iface_pt.assign(N, 0);

        std::vector<Eigen::Vector3d> iface_points;
        iface_points.reserve(Nq);
        for (int q = 0; q < Nq; ++q)
            iface_points.push_back(iface.points().row(q).transpose());
        const NearestPointCloud3D iface_cloud(iface_points);

        const ProfileClock3D::time_point t_sample_nearest_start =
            ProfileClock3D::now();
        for (int n = 0; n < N; ++n) {
            const auto c = grid.coord(n);
            const Eigen::Vector3d pt(c[0], c[1], c[2]);

            const int q = iface_cloud.nearest(pt);
            impl_->closest_iface_pt[n] = q;

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
    // Active P2 surfaces use the normal of each grid node's nearest center.
    // Raw surfaces keep the closed-mesh CGAL labeler below.
    // ------------------------------------------------------------------

    impl_->domain_label_vec.resize(N, 0);
    const ProfileClock3D::time_point t_label_start = ProfileClock3D::now();
    if (nurbs_domain_) {
        impl_->domain_label_vec = nurbs_domain_->labels();
        verify_native_label_crossing_invariant(
            grid, impl_->domain_label_vec, *nurbs_domain_);

        const double t_label = seconds_since(t_label_start);
        if (profile) {
            const double total = seconds_since(t_total_start);
            std::printf("      GridPair3D Ngrid=%d Niface=%d samples=%zu components=%d label_mode=native-nurbs\n",
                        N, Nq, impl_->p2_center_samples.size(), Nc);
            std::printf("        closest_bulk %.6fs sample_setup %.6fs sample_nearest %.6fs center_cache %.6fs domain_labels %.6fs total %.6fs\n",
                        t_closest_bulk, t_sample_setup, t_sample_nearest,
                        t_center_cache, t_label, total);
        }
        return;
    }
    if (active_p2) {
        impl_->domain_label_vec = build_p2_nearest_center_domain_labels(
            grid,
            impl_->p2_center_samples,
            impl_->nearest_p2_center_for_node);

        const double t_label = seconds_since(t_label_start);
        if (profile) {
            const double total = seconds_since(t_total_start);
            std::printf("      GridPair3D Ngrid=%d Niface=%d samples=%zu components=%d label_mode=p2-nearest-center-normal\n",
                        N, Nq, impl_->p2_center_samples.size(), Nc);
            std::printf("        closest_bulk %.6fs sample_setup %.6fs sample_nearest %.6fs center_cache %.6fs domain_labels %.6fs total %.6fs\n",
                        t_closest_bulk, t_sample_setup, t_sample_nearest,
                        t_center_cache, t_label, total);
        }
        return;
    }

    for (int comp = 0; comp < Nc; ++comp) {
        CMesh3 mesh;
        std::unordered_map<int, CMesh3::Vertex_index> vmap;
        auto get_or_add = [&](int gv) {
            auto [it, inserted] = vmap.emplace(gv, CMesh3::Vertex_index{});
            if (inserted)
                it->second = mesh.add_vertex(CPoint3(
                    iface.vertices()(gv, 0),
                    iface.vertices()(gv, 1),
                    iface.vertices()(gv, 2)));
            return it->second;
        };
        for (int p = 0; p < Np; ++p) {
            if (iface.panel_components()(p) != comp) continue;
            mesh.add_face(get_or_add(iface.panels()(p, 0)),
                          get_or_add(iface.panels()(p, 1)),
                          get_or_add(iface.panels()(p, 2)));
        }

        CSideOf side_of(mesh);
        for (int n = 0; n < N; ++n) {
            if (impl_->domain_label_vec[n] != 0) continue;
            auto   c    = grid.coord(n);
            auto   side = side_of(CPoint3(c[0], c[1], c[2]));
            if (side == CGAL::ON_BOUNDED_SIDE || side == CGAL::ON_BOUNDARY)
                impl_->domain_label_vec[n] = comp + 1;
        }
    }
    const double t_label = seconds_since(t_label_start);
    if (profile) {
        const double total = seconds_since(t_total_start);
        std::printf("      GridPair3D Ngrid=%d Niface=%d samples=%zu components=%d label_mode=mesh\n",
                    N, Nq, distance_samples.size(), Nc);
        std::printf("        closest_bulk %.6fs sample_setup %.6fs sample_nearest %.6fs center_cache %.6fs domain_labels %.6fs total %.6fs\n",
                    t_closest_bulk, t_sample_setup, t_sample_nearest,
                    t_center_cache, t_label, total);
    }
}

GridPair3D::~GridPair3D() = default;

// ============================================================================
// Query methods
// ============================================================================
int GridPair3D::closest_bulk_node(int q) const {
    return impl_->closest_bulk_node[q];
}

int GridPair3D::closest_interface_point(int n) const {
    if (impl_->closest_iface_pt[n] < 0) {
        if (!impl_->iface_cloud) {
            std::vector<Eigen::Vector3d> iface_points;
            iface_points.reserve(interface_.num_points());
            for (int q = 0; q < interface_.num_points(); ++q)
                iface_points.push_back(interface_.points().row(q).transpose());
            impl_->iface_cloud =
                std::make_unique<NearestPointCloud3D>(iface_points);
        }
        const auto c = grid_.coord(n);
        impl_->closest_iface_pt[n] =
            impl_->iface_cloud->nearest_stable({c[0], c[1], c[2]});
    }
    return impl_->closest_iface_pt[n];
}

int GridPair3D::nearest_p2_expansion_center(int n) const {
    if (impl_->nearest_p2_center_for_node.empty()) {
        throw std::runtime_error(
            "GridPair3D::nearest_p2_expansion_center requires P2 QuadraticLagrange panels");
    }
    if (impl_->nearest_p2_center_for_node[n] < 0) {
        const auto c = grid_.coord(n);
        const NearestResult3D nearest =
            impl_->p2_center_hash->nearest({c[0], c[1], c[2]});
        impl_->nearest_p2_center_for_node[n] = nearest.index;
        impl_->min_iface_dist[n] = std::sqrt(nearest.dist2);
    }
    return impl_->nearest_p2_center_for_node[n];
}

P2CrossingOwner3D GridPair3D::p2_crossing_owner_between(int a, int b) const {
    if (impl_->p2_center_samples.empty()) {
        throw std::runtime_error(
            "GridPair3D::p2_crossing_owner_between requires P2 QuadraticLagrange panels");
    }
    if (a < 0 || a >= grid_.num_dofs()
        || b < 0 || b >= grid_.num_dofs()) {
        throw std::out_of_range(
            "GridPair3D::p2_crossing_owner_between bulk-node index is outside the grid");
    }

    const auto ca = grid_.coord(a);
    const auto cb = grid_.coord(b);
    const Eigen::Vector3d pa(ca[0], ca[1], ca[2]);
    const Eigen::Vector3d pb(cb[0], cb[1], cb[2]);
    if ((pb - pa).norm() <= 1.0e-14) {
        throw std::invalid_argument(
            "GridPair3D::p2_crossing_owner_between requires distinct bulk nodes");
    }

    const int center_a = nearest_p2_expansion_center(a);
    const int center_b = nearest_p2_expansion_center(b);
    auto distance_sum = [&](int center_index) {
        const Eigen::Vector3d& center = impl_->p2_center_samples[
            static_cast<std::size_t>(center_index)].point;
        return (center - pa).norm() + (center - pb).norm();
    };
    const double score_a = distance_sum(center_a);
    const double score_b = distance_sum(center_b);
    const int center = score_a < score_b - 1.0e-14
        ? center_a
        : (score_b < score_a - 1.0e-14
               ? center_b
               : std::min(center_a, center_b));
    const DistanceSample3D& sample =
        impl_->p2_center_samples[static_cast<std::size_t>(center)];

    P2CrossingOwner3D owner;
    owner.center_index = center;
    owner.panel_index = sample.panel;
    owner.edge_parameter = 0.5;
    owner.barycentric = sample.barycentric;
    owner.status = P2CrossingOwnerStatus3D::EndpointNearestCenter;

    if (nurbs_domain_) {
        if ((impl_->domain_label_vec[static_cast<std::size_t>(a)] > 0)
            != (impl_->domain_label_vec[static_cast<std::size_t>(b)] > 0)) {
            const geometry3d::NurbsSurfaceCrossing3D& crossing =
                nurbs_domain_->correction_crossing_between(a, b);
            owner.nurbs_patch_index = crossing.patch_index;
            owner.nurbs_parameter = {crossing.u, crossing.v};
            owner.crossing_point = crossing.point;
            owner.crossing_normal = crossing.normal;
            owner.surface_component = crossing.component;
            owner.crossing_residual = crossing.residual;
            const Eigen::Vector3d edge = pb - pa;
            owner.edge_parameter = std::max(
                0.0,
                std::min(1.0,
                         (owner.crossing_point - pa).dot(edge)
                             / edge.squaredNorm()));
            owner.status = P2CrossingOwnerStatus3D::ExactIntersection;
        }
        return owner;
    }

    if (impl_->crossing_tree && !impl_->crossing_tree->empty()) {
        const CSegment3 segment(CPoint3(pa.x(), pa.y(), pa.z()),
                                CPoint3(pb.x(), pb.y(), pb.z()));
        Eigen::Vector3d hit = 0.5 * (pa + pb);
        CFace3 face = CMesh3::null_face();
        const auto intersection =
            impl_->crossing_tree->any_intersection(segment);
        if (intersection) {
#if CGAL_VERSION_MAJOR >= 6
            if (const CPoint3* point =
                    std::get_if<CPoint3>(&(intersection->first))) {
                hit = to_eigen(*point);
            } else if (const CSegment3* overlap =
                           std::get_if<CSegment3>(&(intersection->first))) {
                hit = 0.5 * (to_eigen(overlap->source())
                           + to_eigen(overlap->target()));
            }
#else
            if (const CPoint3* point =
                    boost::get<CPoint3>(&(intersection->first))) {
                hit = to_eigen(*point);
            } else if (const CSegment3* overlap =
                           boost::get<CSegment3>(&(intersection->first))) {
                hit = 0.5 * (to_eigen(overlap->source())
                           + to_eigen(overlap->target()));
            }
#endif
            face = intersection->second;
            owner.status = P2CrossingOwnerStatus3D::ExactIntersection;
        } else {
            const auto closest = impl_->crossing_tree->closest_point_and_primitive(
                CPoint3(hit.x(), hit.y(), hit.z()));
            hit = to_eigen(closest.first);
            face = closest.second;
            owner.status = P2CrossingOwnerStatus3D::GapFallback;
        }

        if (face != CMesh3::null_face()
            && face.idx() < impl_->geometry_panel_for_face.size()) {
            owner.geometry_panel_index =
                impl_->geometry_panel_for_face[face.idx()];
        }
        if (owner.geometry_panel_index >= 0) {
            owner.geometry_barycentric = triangle_barycentric(
                crossing_geometry_, owner.geometry_panel_index, hit);
            const Eigen::Vector3d edge = pb - pa;
            owner.edge_parameter = std::max(
                0.0,
                std::min(1.0, (hit - pa).dot(edge) / edge.squaredNorm()));
        } else {
            owner.status = P2CrossingOwnerStatus3D::EndpointNearestCenter;
        }
    }
    return owner;

}

P2CrossingOwner3D GridPair3D::p2_crossing_owner_for_grid_edge_event(
    const geometry3d::GridEdgeEventId3D& event_id) const
{
    if (!nurbs_domain_) {
        throw std::runtime_error(
            "event-specific crossing ownership requires a native NURBS domain");
    }
    if (impl_->p2_center_samples.empty()) {
        throw std::runtime_error(
            "event-specific crossing ownership requires P2 QuadraticLagrange panels");
    }
    if (event_id.first_node < 0
        || event_id.first_node >= grid_.num_dofs()
        || event_id.second_node < 0
        || event_id.second_node >= grid_.num_dofs()
        || event_id.first_node >= event_id.second_node
        || event_id.ordinal < 0) {
        throw std::out_of_range(
            "invalid canonical native grid-edge event id");
    }

    const std::vector<geometry3d::GridEdgeEventView3D> views =
        nurbs_domain_->grid_edge_events_between(
            event_id.first_node, event_id.second_node);
    const geometry3d::GridEdgeEvent3D* event = nullptr;
    for (const geometry3d::GridEdgeEventView3D& view : views) {
        if (view.id() == event_id) {
            event = &view.canonical_event();
            break;
        }
    }
    if (event == nullptr) {
        throw std::runtime_error(
            "native grid-edge event id is absent from the domain catalog");
    }
    if (!event->certified_transverse()) {
        throw std::runtime_error(
            "event-specific crossing ownership rejects an uncertified event");
    }
    if (event->owners.empty()) {
        throw std::logic_error(
            "certified native grid-edge event has no NURBS owner");
    }

    if (!impl_->has_native_panel_patch_metadata) {
        throw std::runtime_error(
            "event-specific native ownership requires panel-to-NURBS patch metadata; refusing an uncertified mixed-sheet owner");
    }
    const std::vector<int>& patch_g1_components =
        nurbs_domain_->patch_g1_components();
    std::vector<P2NativeSheetCenterCandidate3D> candidates;
    candidates.reserve(event->owners.size());
    for (const geometry3d::NurbsSurfaceRootOwner3D& owner : event->owners) {
        if (owner.patch_index < 0
            || owner.patch_index
                   >= static_cast<int>(patch_g1_components.size())) {
            throw std::logic_error(
                "native grid-edge event contains an invalid NURBS owner patch");
        }
        const int g1 = patch_g1_components[
            static_cast<std::size_t>(owner.patch_index)];
        const auto sheet = impl_->p2_centers_by_g1_sheet.find(g1);
        if (sheet == impl_->p2_centers_by_g1_sheet.end()
            || !sheet->second.cloud
            || sheet->second.global_center_indices.empty()) {
            continue;
        }
        const int local_center = sheet->second.cloud->nearest_stable(
            event->point);
        if (local_center < 0
            || local_center >= static_cast<int>(
                   sheet->second.global_center_indices.size())) {
            throw std::logic_error(
                "native G1 sheet center index is outside its lookup table");
        }
        const int center = sheet->second.global_center_indices[
            static_cast<std::size_t>(local_center)];
        const DistanceSample3D& sample = impl_->p2_center_samples[
            static_cast<std::size_t>(center)];
        candidates.push_back({
            center, sample.panel, sample.nurbs_patch, sample.point});
    }
    const P2NativeEventOwnerCenterSelection3D selection =
        select_p2_native_event_owner_center_3d(
            *event, patch_g1_components, candidates,
            std::max(nurbs_domain_->geometry_tolerance(), 1.0e-14));
    const P2NativeSheetCenterCandidate3D& selected_center =
        candidates[static_cast<std::size_t>(
            selection.center_candidate_index)];
    const int center_index = selected_center.center_index;
    const DistanceSample3D& sample = impl_->p2_center_samples[
        static_cast<std::size_t>(center_index)];
    const geometry3d::NurbsSurfaceRootOwner3D& nurbs_owner =
        event->owners[static_cast<std::size_t>(selection.owner_index)];
    if (patch_g1_components[static_cast<std::size_t>(
            nurbs_owner.patch_index)]
        != patch_g1_components[static_cast<std::size_t>(
            sample.nurbs_patch)]) {
        throw std::logic_error(
            "native event owner and P2 expansion center escaped their selected G1 sheet");
    }
    P2CrossingOwner3D owner;
    owner.center_index = center_index;
    owner.panel_index = sample.panel;
    owner.edge_parameter = event->canonical_parameter;
    owner.barycentric = sample.barycentric;
    owner.nurbs_patch_index = nurbs_owner.patch_index;
    owner.nurbs_parameter = {nurbs_owner.u, nurbs_owner.v};
    owner.crossing_point = event->point;
    owner.crossing_normal = nurbs_owner.normal;
    owner.surface_component = event->component;
    owner.crossing_residual = nurbs_owner.residual;
    owner.status = P2CrossingOwnerStatus3D::ExactIntersection;
    return owner;
}

int GridPair3D::nearest_p2_expansion_center_between(int a, int b) const {
    return p2_crossing_owner_between(a, b).center_index;
}

int GridPair3D::nearest_p2_expansion_center_for_interface_point(
    int q) const
{
    if (q < 0
        || q >= static_cast<int>(impl_->owner_p2_center_for_point.size())
        || impl_->owner_p2_center_for_point[static_cast<std::size_t>(q)] < 0) {
        throw std::runtime_error(
            "GridPair3D interface point has no owning P2 expansion center");
    }
    return impl_->owner_p2_center_for_point[static_cast<std::size_t>(q)];
}

int GridPair3D::domain_label(int n) const {
    return impl_->domain_label_vec[n];
}

bool GridPair3D::has_nurbs_domain() const noexcept
{
    return static_cast<bool>(nurbs_domain_);
}

const geometry3d::NurbsCartesianDomain3D*
GridPair3D::nurbs_cartesian_domain() const noexcept
{
    return nurbs_domain_.get();
}

const geometry3d::NurbsCartesianDomainDiagnostics3D&
GridPair3D::nurbs_domain_diagnostics() const
{
    if (!nurbs_domain_)
        throw std::runtime_error("GridPair3D has no native NURBS domain");
    return nurbs_domain_->diagnostics();
}

bool GridPair3D::is_near_interface(int n, double radius) const {
    impl_->ensure_p2_center_cache_radius(grid_, radius);
    return impl_->min_iface_dist[n] < radius;
}

std::vector<int> GridPair3D::near_interface_nodes(double radius) const {
    impl_->ensure_p2_center_cache_radius(grid_, radius);
    int N = grid_.num_dofs();
    std::vector<int> result;
    result.reserve(N / 8);
    for (int n = 0; n < N; ++n)
        if (impl_->min_iface_dist[n] < radius)
            result.push_back(n);
    return result;
}

std::vector<int> GridPair3D::near_interface_points(int n, double radius) const {
    auto   c  = grid_.coord(n);
    double cx = c[0], cy = c[1], cz = c[2];
    int Nq = interface_.num_points();
    std::vector<int> result;
    for (int q = 0; q < Nq; ++q) {
        double dx = cx - interface_.points()(q, 0);
        double dy = cy - interface_.points()(q, 1);
        double dz = cz - interface_.points()(q, 2);
        if (std::sqrt(dx*dx + dy*dy + dz*dz) < radius)
            result.push_back(q);
    }
    return result;
}

NarrowBandProjection3D GridPair3D::project_near_interface_nodes(double radius) const
{
    return project_p2_near_interface_nodes_3d(*this, radius);
}

NarrowBandProjection3D GridPair3D::project_grid_nodes_to_interface(
    const std::vector<int>& bulk_node_indices) const
{
    return project_p2_grid_nodes_to_interface_3d(*this, bulk_node_indices);
}

} // namespace kfbim
