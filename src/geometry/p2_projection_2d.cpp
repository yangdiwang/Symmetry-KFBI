#include "p2_projection_2d.hpp"

#include "grid_pair_2d.hpp"
#include "p2_curve_2d.hpp"

#include <algorithm>
#include <cmath>
#include <stdexcept>
#include <string>

namespace kfbim {

namespace {

struct ProjectionSeed2D {
    Eigen::Vector2d point;
    double          local_s = 0.0;
    int             panel = -1;
    int             component = -1;
};

static Eigen::Vector2d grid_point(const CartesianGrid2D& grid, int idx)
{
    const auto c = grid.coord(idx);
    return {c[0], c[1]};
}

static ProjectionSeed2D seed_from_center_index(const Interface2D& iface,
                                               int                center_idx)
{
    const int centers_per_panel =
        static_cast<int>(geometry2d::kP2CenterS.size());
    if (center_idx < 0 || center_idx >= centers_per_panel * iface.num_panels())
        throw std::out_of_range("P2 curve projection seed index is invalid");
    const int panel = center_idx / centers_per_panel;
    const int local = center_idx % centers_per_panel;
    const double s = geometry2d::kP2CenterS[local];
    return {geometry2d::panel_point(iface, panel, s),
            s,
            panel,
            iface.panel_components()(panel)};
}

static CurveProjection2D make_curve_projection(const Interface2D& iface,
                                               int                grid_node,
                                               const ProjectionSeed2D& seed,
                                               Eigen::Vector2d    target,
                                               double             local_s,
                                               int                iterations,
                                               bool               converged)
{
    const Eigen::Vector2d point =
        geometry2d::panel_point(iface, seed.panel, local_s);
    const Eigen::Vector2d tangent =
        geometry2d::panel_tangent(iface, seed.panel, local_s);
    const Eigen::Vector2d normal =
        geometry2d::panel_normal(iface, seed.panel, local_s);
    const Eigen::Vector2d r = point - target;

    CurveProjection2D projection;
    projection.grid_node = grid_node;
    projection.panel = seed.panel;
    projection.component = seed.component;
    projection.local_s = local_s;
    projection.point = point;
    projection.normal = normal;
    projection.signed_distance = (target - point).dot(normal);
    projection.distance = r.norm();
    const double tangent_len = tangent.norm();
    projection.tangential_residual =
        tangent_len > 1.0e-14 ? std::abs(r.dot(tangent)) / tangent_len
                              : std::abs(r.dot(tangent));
    projection.iterations = iterations;
    projection.converged = converged;
    return projection;
}

static CurveProjection2D project_from_seed(const Interface2D& iface,
                                           int                grid_node,
                                           Eigen::Vector2d    target,
                                           const ProjectionSeed2D& seed)
{
    constexpr int kMaxIterations = 16;
    constexpr double kResidualTol = 1.0e-12;
    constexpr double kStepTol = 1.0e-13;

    double s = seed.local_s;
    bool clamped = false;
    bool converged = false;
    int iterations = 0;

    for (; iterations < kMaxIterations; ++iterations) {
        const Eigen::Vector2d point =
            geometry2d::panel_point(iface, seed.panel, s);
        const Eigen::Vector2d tangent =
            geometry2d::panel_tangent(iface, seed.panel, s);
        const Eigen::Vector2d second =
            geometry2d::panel_second_derivative(iface, seed.panel, s);
        const Eigen::Vector2d r = point - target;
        const double f = r.dot(tangent);
        const double tangent_len = tangent.norm();
        const double residual =
            tangent_len > 1.0e-14 ? std::abs(f) / tangent_len : std::abs(f);
        if (residual <= kResidualTol) {
            converged = !clamped;
            break;
        }

        const double jac = tangent.squaredNorm() + r.dot(second);
        if (std::abs(jac) <= 1.0e-28 || !std::isfinite(jac))
            break;

        const double raw_next = s - f / jac;
        const double next = geometry2d::clamp_to_panel(raw_next);
        if (next != raw_next)
            clamped = true;
        if (std::abs(next - s) <= kStepTol) {
            s = next;
            break;
        }
        s = next;
    }

    if (!converged && !clamped) {
        const Eigen::Vector2d point =
            geometry2d::panel_point(iface, seed.panel, s);
        const Eigen::Vector2d tangent =
            geometry2d::panel_tangent(iface, seed.panel, s);
        const Eigen::Vector2d r = point - target;
        const double tangent_len = tangent.norm();
        const double residual =
            tangent_len > 1.0e-14 ? std::abs(r.dot(tangent)) / tangent_len
                                  : std::abs(r.dot(tangent));
        converged = residual <= 10.0 * kResidualTol;
    }

    return make_curve_projection(iface,
                                 grid_node,
                                 seed,
                                 target,
                                 geometry2d::clamp_to_panel(s),
                                 iterations,
                                 converged);
}

static void require_p2_projection_interface(const GridPair2D& grid_pair,
                                            const char* context)
{
    const Interface2D& iface = grid_pair.interface();
    if (!geometry2d::is_quadratic_lagrange_panel_layout(iface)) {
        throw std::invalid_argument(
            std::string(context) + " requires P2 quadratic 3-point panels");
    }
}

} // namespace

bool NarrowBandProjection2D::has_projection(int bulk_node_idx) const
{
    return bulk_node_idx >= 0
        && bulk_node_idx < static_cast<int>(projection_index_by_grid_node_.size())
        && projection_index_by_grid_node_[bulk_node_idx] >= 0;
}

const CurveProjection2D& NarrowBandProjection2D::projection(int bulk_node_idx) const
{
    if (!has_projection(bulk_node_idx))
        throw std::out_of_range("NarrowBandProjection2D missing grid-node projection");
    return projections_[projection_index_by_grid_node_[bulk_node_idx]];
}

NarrowBandProjection2D project_p2_near_interface_nodes_2d(
    const GridPair2D& grid_pair,
    double radius)
{
    if (radius < 0.0)
        throw std::invalid_argument("GridPair2D projection radius must be nonnegative");
    require_p2_projection_interface(grid_pair,
                                    "GridPair2D::project_near_interface_nodes");

    const CartesianGrid2D& grid = grid_pair.grid();
    const Interface2D& iface = grid_pair.interface();
    NarrowBandProjection2D band;
    band.radius_ = radius;
    band.nodes_ = grid_pair.near_interface_nodes(radius);
    band.projection_index_by_grid_node_.assign(grid.num_dofs(), -1);
    band.projections_.reserve(band.nodes_.size());

    if (band.nodes_.empty())
        return band;

    for (int node : band.nodes_) {
        const Eigen::Vector2d target = grid_point(grid, node);
        const int seed_idx = grid_pair.nearest_p2_expansion_center(node);
        const ProjectionSeed2D seed = seed_from_center_index(iface, seed_idx);
        CurveProjection2D projection =
            project_from_seed(iface, node, target, seed);
        band.projection_index_by_grid_node_[node] =
            static_cast<int>(band.projections_.size());
        band.projections_.push_back(projection);
    }

    return band;
}

NarrowBandProjection2D project_p2_grid_nodes_to_interface_2d(
    const GridPair2D& grid_pair,
    const std::vector<int>& bulk_node_indices)
{
    require_p2_projection_interface(grid_pair,
                                    "GridPair2D::project_grid_nodes_to_interface");

    const CartesianGrid2D& grid = grid_pair.grid();
    const Interface2D& iface = grid_pair.interface();
    NarrowBandProjection2D band;
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

    for (int node : band.nodes_) {
        const Eigen::Vector2d target = grid_point(grid, node);
        const int seed_idx = grid_pair.nearest_p2_expansion_center(node);
        const ProjectionSeed2D seed = seed_from_center_index(iface, seed_idx);
        CurveProjection2D projection =
            project_from_seed(iface, node, target, seed);
        band.projection_index_by_grid_node_[node] =
            static_cast<int>(band.projections_.size());
        band.projections_.push_back(projection);
    }

    return band;
}

} // namespace kfbim
