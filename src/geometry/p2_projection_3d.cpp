#include "p2_projection_3d.hpp"

#include "grid_pair_3d.hpp"
#include "p2_surface_3d.hpp"

#include <CGAL/Exact_predicates_inexact_constructions_kernel.h>
#include <CGAL/Orthogonal_k_neighbor_search.h>
#include <CGAL/Search_traits_3.h>
#include <CGAL/Search_traits_adapter.h>
#include <CGAL/property_map.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <limits>
#include <memory>
#include <stdexcept>
#include <string>
#include <utility>

namespace kfbim {

namespace {

using K3 = CGAL::Exact_predicates_inexact_constructions_kernel;
using CPoint3 = K3::Point_3;
using CPointWithIndex3 = std::pair<CPoint3, int>;
using CSearchBaseTraits3 = CGAL::Search_traits_3<K3>;
using CSearchTraits3 =
    CGAL::Search_traits_adapter<CPointWithIndex3,
                                CGAL::First_of_pair_property_map<CPointWithIndex3>,
                                CSearchBaseTraits3>;
using CNeighborSearch3 = CGAL::Orthogonal_k_neighbor_search<CSearchTraits3>;
using CSearchTree3 = CNeighborSearch3::Tree;

class NearestPointCloud3D {
public:
    explicit NearestPointCloud3D(const std::vector<Eigen::Vector3d>& points)
    {
        if (points.empty())
            throw std::invalid_argument("NearestPointCloud3D requires at least one point");

        points_.reserve(points.size());
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
    std::vector<CPointWithIndex3> points_;
    std::unique_ptr<CSearchTree3> tree_;
};

struct ProjectionSeed3D {
    Eigen::Vector3d point;
    Eigen::Vector3d barycentric;
    int             panel;
    int             component;
};

struct ProjectionEval3D {
    Eigen::Vector3d point;
    Eigen::Vector3d normal;
    Eigen::Vector2d tangential_equations;
    double          distance;
    double          signed_distance;
    double          tangential_residual;
    double          merit;
};

static Eigen::Vector3d grid_point(const CartesianGrid3D& grid, int idx)
{
    const auto c = grid.coord(idx);
    return {c[0], c[1], c[2]};
}

static bool finite_vector(Eigen::Vector2d v)
{
    return std::isfinite(v[0]) && std::isfinite(v[1]);
}

static bool reference_triangle_contains(Eigen::Vector3d bary, double tol)
{
    return bary[0] >= -tol
        && bary[1] >= -tol
        && bary[2] >= -tol
        && std::abs(bary.sum() - 1.0) <= tol;
}

static Eigen::Vector2d clamp_uv_to_reference_triangle(Eigen::Vector2d uv)
{
    if (!finite_vector(uv))
        return Eigen::Vector2d::Zero();

    const Eigen::Vector3d bary = geometry3d::barycentric_from_uv(uv);
    if (reference_triangle_contains(bary, 0.0))
        return uv;

    auto best = Eigen::Vector2d(0.0, std::max(0.0, std::min(1.0, uv[1])));
    double best_dist2 = (best - uv).squaredNorm();

    auto try_candidate = [&](Eigen::Vector2d candidate) {
        const double dist2 = (candidate - uv).squaredNorm();
        if (dist2 < best_dist2) {
            best = candidate;
            best_dist2 = dist2;
        }
    };

    try_candidate({std::max(0.0, std::min(1.0, uv[0])), 0.0});

    double edge_u = 0.5 * (uv[0] - uv[1] + 1.0);
    edge_u = std::max(0.0, std::min(1.0, edge_u));
    try_candidate({edge_u, 1.0 - edge_u});

    return best;
}

static Eigen::Vector3d closest_flat_triangle_barycentric(const Interface3D& iface,
                                                         int                panel,
                                                         Eigen::Vector3d    target)
{
    const Eigen::Vector3d a =
        geometry3d::panel_point(iface, panel, Eigen::Vector3d(1.0, 0.0, 0.0));
    const Eigen::Vector3d b =
        geometry3d::panel_point(iface, panel, Eigen::Vector3d(0.0, 1.0, 0.0));
    const Eigen::Vector3d c =
        geometry3d::panel_point(iface, panel, Eigen::Vector3d(0.0, 0.0, 1.0));

    const Eigen::Vector3d ab = b - a;
    const Eigen::Vector3d ac = c - a;
    const Eigen::Vector3d ap = target - a;
    const double d1 = ab.dot(ap);
    const double d2 = ac.dot(ap);
    if (d1 <= 0.0 && d2 <= 0.0)
        return {1.0, 0.0, 0.0};

    const Eigen::Vector3d bp = target - b;
    const double d3 = ab.dot(bp);
    const double d4 = ac.dot(bp);
    if (d3 >= 0.0 && d4 <= d3)
        return {0.0, 1.0, 0.0};

    const double vc = d1 * d4 - d3 * d2;
    if (vc <= 0.0 && d1 >= 0.0 && d3 <= 0.0) {
        const double v = d1 / (d1 - d3);
        return {1.0 - v, v, 0.0};
    }

    const Eigen::Vector3d cp = target - c;
    const double d5 = ab.dot(cp);
    const double d6 = ac.dot(cp);
    if (d6 >= 0.0 && d5 <= d6)
        return {0.0, 0.0, 1.0};

    const double vb = d5 * d2 - d1 * d6;
    if (vb <= 0.0 && d2 >= 0.0 && d6 <= 0.0) {
        const double w = d2 / (d2 - d6);
        return {1.0 - w, 0.0, w};
    }

    const double va = d3 * d6 - d5 * d4;
    if (va <= 0.0 && (d4 - d3) >= 0.0 && (d5 - d6) >= 0.0) {
        const double w = (d4 - d3) / ((d4 - d3) + (d5 - d6));
        return {0.0, 1.0 - w, w};
    }

    const double denom = va + vb + vc;
    if (std::abs(denom) <= 1.0e-28 || !std::isfinite(denom))
        return {1.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0};

    const double v = vb / denom;
    const double w = vc / denom;
    return {1.0 - v - w, v, w};
}

static ProjectionEval3D projection_eval(const Interface3D& iface,
                                        int                panel,
                                        Eigen::Vector3d    target,
                                        Eigen::Vector2d    uv)
{
    const Eigen::Vector3d bary = geometry3d::barycentric_from_uv(uv);
    const Eigen::Vector3d point = geometry3d::panel_point(iface, panel, bary);
    const Eigen::Vector3d normal =
        geometry3d::panel_oriented_normal(iface, panel, bary);
    const Eigen::Vector3d r = point - target;
    const Eigen::Vector3d Xu = geometry3d::panel_tangent_u(iface, panel, bary);
    const Eigen::Vector3d Xv = geometry3d::panel_tangent_v(iface, panel, bary);

    ProjectionEval3D eval;
    eval.point = point;
    eval.normal = normal;
    eval.tangential_equations = {r.dot(Xu), r.dot(Xv)};
    eval.distance = r.norm();
    eval.signed_distance = (target - point).dot(normal);
    eval.merit = eval.tangential_equations.squaredNorm();

    Eigen::Matrix2d metric;
    metric << Xu.dot(Xu), Xu.dot(Xv),
              Xv.dot(Xu), Xv.dot(Xv);
    const double det = metric.determinant();
    if (std::abs(det) > 1.0e-28) {
        const Eigen::Vector2d coeff =
            metric.fullPivLu().solve(eval.tangential_equations);
        const Eigen::Vector3d tangential = coeff[0] * Xu + coeff[1] * Xv;
        eval.tangential_residual = tangential.norm();
    } else {
        eval.tangential_residual = std::sqrt(eval.merit);
    }

    return eval;
}

static SurfaceProjection3D make_surface_projection(const Interface3D& iface,
                                                   int grid_node,
                                                   const ProjectionSeed3D& seed,
                                                   Eigen::Vector3d target,
                                                   Eigen::Vector2d uv,
                                                   int iterations,
                                                   bool converged)
{
    Eigen::Vector3d bary = geometry3d::barycentric_from_uv(uv);
    if (!reference_triangle_contains(bary, 1.0e-8)) {
        uv = clamp_uv_to_reference_triangle(uv);
        bary = geometry3d::barycentric_from_uv(uv);
        converged = false;
    }

    const ProjectionEval3D eval =
        projection_eval(iface, seed.panel, target, uv);

    SurfaceProjection3D projection;
    projection.grid_node = grid_node;
    projection.panel = seed.panel;
    projection.component = seed.component;
    projection.barycentric = bary;
    projection.point = eval.point;
    projection.normal = eval.normal;
    projection.signed_distance = eval.signed_distance;
    projection.distance = eval.distance;
    projection.tangential_residual = eval.tangential_residual;
    projection.iterations = iterations;
    projection.converged = converged;
    return projection;
}

static SurfaceProjection3D project_from_seed(const Interface3D& iface,
                                             int grid_node,
                                             Eigen::Vector3d target,
                                             const ProjectionSeed3D& seed)
{
    constexpr int kMaxIterations = 30;
    constexpr double kResidualTol = 1.0e-11;
    constexpr double kStepTol = 1.0e-13;

    Eigen::Vector2d uv = geometry3d::uv_from_barycentric(seed.barycentric);
    int iterations = 0;
    bool converged = false;

    const Eigen::Vector3d Xuu = geometry3d::panel_second_uu(iface, seed.panel);
    const Eigen::Vector3d Xuv = geometry3d::panel_second_uv(iface, seed.panel);
    const Eigen::Vector3d Xvv = geometry3d::panel_second_vv(iface, seed.panel);

    for (int iter = 0; iter < kMaxIterations; ++iter) {
        const Eigen::Vector3d bary = geometry3d::barycentric_from_uv(uv);
        const ProjectionEval3D eval =
            projection_eval(iface, seed.panel, target, uv);
        if (eval.tangential_residual <= kResidualTol
            && reference_triangle_contains(bary, 1.0e-8)) {
            converged = true;
            iterations = iter;
            break;
        }

        const Eigen::Vector3d point =
            geometry3d::panel_point(iface, seed.panel, bary);
        const Eigen::Vector3d r = point - target;
        const Eigen::Vector3d Xu =
            geometry3d::panel_tangent_u(iface, seed.panel, bary);
        const Eigen::Vector3d Xv =
            geometry3d::panel_tangent_v(iface, seed.panel, bary);

        Eigen::Matrix2d J;
        J(0, 0) = Xu.dot(Xu) + r.dot(Xuu);
        J(0, 1) = Xv.dot(Xu) + r.dot(Xuv);
        J(1, 0) = Xu.dot(Xv) + r.dot(Xuv);
        J(1, 1) = Xv.dot(Xv) + r.dot(Xvv);

        if (std::abs(J.determinant()) <= 1.0e-28)
            break;

        const Eigen::Vector2d delta = J.fullPivLu().solve(eval.tangential_equations);
        if (!finite_vector(delta))
            break;

        bool accepted = false;
        Eigen::Vector2d accepted_uv = uv;
        Eigen::Vector2d accepted_step = Eigen::Vector2d::Zero();
        for (double damping = 1.0; damping >= 1.0 / 64.0; damping *= 0.5) {
            Eigen::Vector2d candidate = uv - damping * delta;
            if (candidate[0] < -0.5 || candidate[1] < -0.5
                || candidate[0] + candidate[1] > 1.5) {
                candidate = clamp_uv_to_reference_triangle(candidate);
            }
            if (!finite_vector(candidate))
                continue;

            const ProjectionEval3D candidate_eval =
                projection_eval(iface, seed.panel, target, candidate);
            if (candidate_eval.merit <= eval.merit || damping <= 1.0 / 64.0) {
                accepted = true;
                accepted_uv = candidate;
                accepted_step = damping * delta;
                break;
            }
        }

        if (!accepted)
            break;

        uv = accepted_uv;
        iterations = iter + 1;
        if (accepted_step.norm() <= kStepTol)
            break;
    }

    const Eigen::Vector3d final_bary = geometry3d::barycentric_from_uv(uv);
    const ProjectionEval3D final_eval =
        projection_eval(iface, seed.panel, target, uv);
    converged = converged
        || (final_eval.tangential_residual <= kResidualTol
            && reference_triangle_contains(final_bary, 1.0e-8));

    return make_surface_projection(iface,
                                   grid_node,
                                   seed,
                                   target,
                                   uv,
                                   iterations,
                                   converged);
}

static SurfaceProjection3D project_from_nearest_center_with_flat_fallback(
    const Interface3D& iface,
    int grid_node,
    Eigen::Vector3d target,
    const ProjectionSeed3D& nearest_seed)
{
    ProjectionSeed3D flat_seed = nearest_seed;
    flat_seed.barycentric =
        closest_flat_triangle_barycentric(iface, nearest_seed.panel, target);

    SurfaceProjection3D projection =
        project_from_seed(iface, grid_node, target, flat_seed);
    if (projection.converged)
        return projection;

    return make_surface_projection(iface,
                                   grid_node,
                                   flat_seed,
                                   target,
                                   geometry3d::uv_from_barycentric(
                                       flat_seed.barycentric),
                                   0,
                                   false);
}

static std::vector<ProjectionSeed3D> projection_seeds_3d(const Interface3D& iface)
{
    const std::vector<Eigen::Vector3d> center_bary =
        geometry3d::expansion_center_barycentrics();
    std::vector<ProjectionSeed3D> seeds;
    seeds.reserve(static_cast<std::size_t>(iface.num_panels())
                  * center_bary.size());

    for (int p = 0; p < iface.num_panels(); ++p) {
        for (const Eigen::Vector3d& bary : center_bary) {
            seeds.push_back({geometry3d::panel_point(iface, p, bary),
                             bary,
                             p,
                             iface.panel_components()[p]});
        }
    }
    return seeds;
}

static bool better_projection(const SurfaceProjection3D& candidate,
                              const SurfaceProjection3D& current)
{
    if (current.panel < 0)
        return true;
    if (candidate.converged != current.converged)
        return candidate.converged;
    if (candidate.converged)
        return candidate.distance < current.distance;
    if (candidate.tangential_residual != current.tangential_residual)
        return candidate.tangential_residual < current.tangential_residual;
    return candidate.distance < current.distance;
}

static bool profile_projection_3d()
{
    return std::getenv("KFBIM_PROFILE_INTERFACE_3D") != nullptr
        || std::getenv("KFBIM_PROFILE_TRANSFER_3D") != nullptr;
}

using ProjectionProfileClock = std::chrono::steady_clock;

static double projection_seconds_since(ProjectionProfileClock::time_point start)
{
    return std::chrono::duration<double>(
        ProjectionProfileClock::now() - start).count();
}

static void require_p2_projection_interface(const GridPair3D& grid_pair,
                                            const char* context)
{
    const Interface3D& iface = grid_pair.interface();
    if (iface.points_per_panel() != 6
        || iface.panel_node_layout() != PanelNodeLayout3D::QuadraticLagrange) {
        throw std::invalid_argument(
            std::string(context) + " requires P2 QuadraticLagrange panels");
    }
}

} // namespace

bool NarrowBandProjection3D::has_projection(int bulk_node_idx) const
{
    return bulk_node_idx >= 0
        && bulk_node_idx < static_cast<int>(projection_index_by_grid_node_.size())
        && projection_index_by_grid_node_[bulk_node_idx] >= 0;
}

const SurfaceProjection3D& NarrowBandProjection3D::projection(int bulk_node_idx) const
{
    if (!has_projection(bulk_node_idx))
        throw std::out_of_range("NarrowBandProjection3D missing grid-node projection");
    return projections_[projection_index_by_grid_node_[bulk_node_idx]];
}

NarrowBandProjection3D project_p2_near_interface_nodes_3d(
    const GridPair3D& grid_pair,
    double radius)
{
    if (radius < 0.0)
        throw std::invalid_argument("GridPair3D projection radius must be nonnegative");
    require_p2_projection_interface(grid_pair,
                                    "GridPair3D::project_near_interface_nodes");

    const CartesianGrid3D& grid = grid_pair.grid();
    const Interface3D& iface = grid_pair.interface();
    NarrowBandProjection3D band;
    band.radius_ = radius;
    band.nodes_ = grid_pair.near_interface_nodes(radius);
    band.projection_index_by_grid_node_.assign(grid.num_dofs(), -1);
    band.projections_.reserve(band.nodes_.size());

    if (band.nodes_.empty())
        return band;

    const ProjectionProfileClock::time_point setup_start =
        ProjectionProfileClock::now();
    const std::vector<ProjectionSeed3D> seeds = projection_seeds_3d(iface);
    const double t_setup = projection_seconds_since(setup_start);

    constexpr int kRetrySeeds = 64;
    const int k = std::min(kRetrySeeds, static_cast<int>(seeds.size()));
    std::unique_ptr<NearestPointCloud3D> seed_cloud;
    auto ensure_seed_cloud = [&]() -> const NearestPointCloud3D& {
        if (!seed_cloud) {
            std::vector<Eigen::Vector3d> seed_points;
            seed_points.reserve(seeds.size());
            for (const auto& seed : seeds)
                seed_points.push_back(seed.point);
            seed_cloud = std::make_unique<NearestPointCloud3D>(seed_points);
        }
        return *seed_cloud;
    };

    long long seed_attempts = 0;
    long long first_seed_converged = 0;
    long long converged_best = 0;
    long long newton_iterations = 0;
    const ProjectionProfileClock::time_point node_start =
        ProjectionProfileClock::now();
    for (int node : band.nodes_) {
        const Eigen::Vector3d target = grid_point(grid, node);
        const int first_seed_idx = grid_pair.nearest_p2_expansion_center(node);
        SurfaceProjection3D best;

        const SurfaceProjection3D first_candidate =
            project_from_seed(iface, node, target, seeds[first_seed_idx]);
        ++seed_attempts;
        newton_iterations += first_candidate.iterations;
        best = first_candidate;
        if (first_candidate.converged) {
            ++first_seed_converged;
        } else {
            const std::vector<int> seed_indices =
                ensure_seed_cloud().nearest_k(target, k);
            for (int seed_idx : seed_indices) {
                if (seed_idx == first_seed_idx)
                    continue;
                const SurfaceProjection3D candidate =
                    project_from_seed(iface, node, target, seeds[seed_idx]);
                ++seed_attempts;
                newton_iterations += candidate.iterations;
                if (better_projection(candidate, best))
                    best = candidate;
            }
        }

        if (best.converged)
            ++converged_best;

        band.projection_index_by_grid_node_[node] =
            static_cast<int>(band.projections_.size());
        band.projections_.push_back(best);
    }
    const double t_nodes = projection_seconds_since(node_start);

    if (profile_projection_3d()) {
        const double nodes =
            static_cast<double>(std::max<std::size_t>(1, band.nodes_.size()));
        std::printf("        project_near_interface_nodes radius=%.4e nodes=%zu seeds=%zu k=%d setup %.3fs node_project %.3fs attempts=%lld avg_attempts=%.2f first_converged=%lld best_converged=%lld avg_newton_iters=%.2f\n",
                    radius,
                    band.nodes_.size(),
                    seeds.size(),
                    k,
                    t_setup,
                    t_nodes,
                    seed_attempts,
                    static_cast<double>(seed_attempts) / nodes,
                    first_seed_converged,
                    converged_best,
                    static_cast<double>(newton_iterations) / nodes);
    }

    return band;
}

NarrowBandProjection3D project_p2_grid_nodes_to_interface_3d(
    const GridPair3D& grid_pair,
    const std::vector<int>& bulk_node_indices)
{
    require_p2_projection_interface(grid_pair,
                                    "GridPair3D::project_grid_nodes_to_interface");

    const CartesianGrid3D& grid = grid_pair.grid();
    const Interface3D& iface = grid_pair.interface();
    NarrowBandProjection3D band;
    band.radius_ = 0.0;
    band.projection_index_by_grid_node_.assign(grid.num_dofs(), -1);

    band.nodes_ = bulk_node_indices;
    std::sort(band.nodes_.begin(), band.nodes_.end());
    band.nodes_.erase(std::unique(band.nodes_.begin(), band.nodes_.end()),
                      band.nodes_.end());
    for (int node : band.nodes_) {
        if (node < 0 || node >= grid.num_dofs())
            throw std::out_of_range("projection support contains invalid grid node");
    }
    band.projections_.reserve(band.nodes_.size());

    if (band.nodes_.empty())
        return band;

    const ProjectionProfileClock::time_point setup_start =
        ProjectionProfileClock::now();
    const std::vector<ProjectionSeed3D> seeds = projection_seeds_3d(iface);
    const double t_setup = projection_seconds_since(setup_start);

    long long converged_count = 0;
    long long newton_iterations = 0;
    const ProjectionProfileClock::time_point node_start =
        ProjectionProfileClock::now();
    for (int node : band.nodes_) {
        const Eigen::Vector3d target = grid_point(grid, node);
        const int seed_idx = grid_pair.nearest_p2_expansion_center(node);
        SurfaceProjection3D projection =
            project_from_nearest_center_with_flat_fallback(iface,
                                                           node,
                                                           target,
                                                           seeds[seed_idx]);
        if (projection.converged)
            ++converged_count;
        newton_iterations += projection.iterations;

        band.projection_index_by_grid_node_[node] =
            static_cast<int>(band.projections_.size());
        band.projections_.push_back(projection);
    }
    const double t_nodes = projection_seconds_since(node_start);

    if (profile_projection_3d()) {
        const double nodes =
            static_cast<double>(std::max<std::size_t>(1, band.nodes_.size()));
        std::printf("        project_grid_nodes_to_interface nodes=%zu seeds=%zu setup %.3fs node_project %.3fs avg_attempts=1.00 converged=%lld avg_newton_iters=%.2f\n",
                    band.nodes_.size(),
                    seeds.size(),
                    t_setup,
                    t_nodes,
                    converged_count,
                    static_cast<double>(newton_iterations) / nodes);
    }

    return band;
}

} // namespace kfbim
