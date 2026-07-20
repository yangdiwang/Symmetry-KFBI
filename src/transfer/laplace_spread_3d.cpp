#include "laplace_spread_3d.hpp"

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <limits>
#include <memory>
#include <stdexcept>
#include <unordered_map>
#include <utility>

#include "../geometry/p2_projection_3d.hpp"
#include "../grid/structured_grid_ops.hpp"
#include "../local_cauchy/edge_patch_laplacian_3d.hpp"
#include "../local_cauchy/laplace_patch_solver_3d.hpp"
#include "laplace_projection_correction_3d.hpp"

namespace kfbim {

namespace {

Eigen::Vector3d node_coord(const CartesianGrid3D& grid, int idx)
{
    return structured_grid::point(grid, idx);
}

LocalPoly3D center_poly(const PatchCenterCauchyResult3D& cauchy, int idx)
{
    LocalPoly3D poly;
    poly.center = cauchy.centers.row(idx).transpose();
    poly.coeffs = cauchy.coeffs.row(idx).transpose();
    return poly;
}

bool profile_projection_transfer_3d()
{
    return std::getenv("KFBIM_PROFILE_INTERFACE_3D") != nullptr
        || std::getenv("KFBIM_PROFILE_TRANSFER_3D") != nullptr;
}

using ProfileClock = std::chrono::steady_clock;

double seconds_since(ProfileClock::time_point start)
{
    return std::chrono::duration<double>(ProfileClock::now() - start).count();
}

std::vector<P2CrossingOwner3D> build_crossing_owners_for_spread_3d(
    const GridPair3D& grid_pair,
    const LaplaceCorrectionSupport3D& support)
{
    std::vector<P2CrossingOwner3D> owners(support.crossing_ops.size());
    std::unordered_map<std::uint64_t, P2CrossingOwner3D>
        owner_by_undirected_edge;
    owner_by_undirected_edge.reserve(support.crossing_ops.size());
    for (std::size_t op_idx = 0; op_idx < support.crossing_ops.size();
         ++op_idx) {
        const LaplaceCrossingCorrectionOp& op =
            support.crossing_ops[op_idx];
        const std::uint32_t lo = static_cast<std::uint32_t>(
            std::min(op.rhs_node, op.correction_node));
        const std::uint32_t hi = static_cast<std::uint32_t>(
            std::max(op.rhs_node, op.correction_node));
        const std::uint64_t key =
            (static_cast<std::uint64_t>(lo) << 32)
            | static_cast<std::uint64_t>(hi);
        const auto found = owner_by_undirected_edge.find(key);
        if (found != owner_by_undirected_edge.end()) {
            owners[op_idx] = found->second;
            continue;
        }
        const P2CrossingOwner3D owner =
            grid_pair.p2_crossing_owner_between(
            op.rhs_node, op.correction_node);
        owner_by_undirected_edge.emplace(key, owner);
        owners[op_idx] = owner;
    }
    return owners;
}

Eigen::Vector3d crossing_point(const CartesianGrid3D& grid,
                               const LaplaceCrossingCorrectionOp& op,
                               const P2CrossingOwner3D& owner)
{
    const Eigen::Vector3d a = node_coord(grid, op.rhs_node);
    const Eigen::Vector3d b = node_coord(grid, op.correction_node);
    const double t = std::max(0.0, std::min(1.0, owner.edge_parameter));
    return a + t * (b - a);
}

} // namespace

LaplaceQuadraticPatchCenterSpread3D::LaplaceQuadraticPatchCenterSpread3D(
    const GridPair3D& grid_pair,
    double            kappa,
    LaplaceCorrectionMethod3D correction_method,
    int projection_restrict_stencil_radius)
    : grid_pair_(grid_pair)
    , kappa_(kappa)
    , correction_method_(correction_method)
    , projection_restrict_stencil_radius_(projection_restrict_stencil_radius)
    , support_(build_laplace_correction_support_3d(
          grid_pair,
          "LaplaceQuadraticPatchCenterSpread3D"))
    , crossing_owners_(
          correction_method_ == LaplaceCorrectionMethod3D::NearestExpansionCenter
              ? build_crossing_owners_for_spread_3d(grid_pair_, support_)
              : std::vector<P2CrossingOwner3D>{})
{
    if (projection_restrict_stencil_radius_ < 1)
        throw std::invalid_argument(
            "LaplaceQuadraticPatchCenterSpread3D projection restrict stencil radius must be positive");
    if (correction_method_ == LaplaceCorrectionMethod3D::ProjectionPoint) {
        projection_cache_ =
            project_p2_grid_nodes_to_interface_3d(grid_pair_, support_.projection_nodes);
    }
}

LaplaceSpreadResult3D LaplaceQuadraticPatchCenterSpread3D::apply(
    const std::vector<LaplaceJumpData3D>& jumps,
    Eigen::VectorXd&                      rhs_correction) const
{
    return apply(jumps, rhs_correction, EdgePatchSpreadCorrection3D{});
}

LaplaceSpreadResult3D LaplaceQuadraticPatchCenterSpread3D::apply(
    const std::vector<LaplaceJumpData3D>& jumps,
    Eigen::VectorXd&                      rhs_correction,
    const EdgePatchSpreadCorrection3D&   edge_correction) const
{
    const auto& grid = grid_pair_.grid();
    const auto& iface = grid_pair_.interface();
    const int n_grid = grid.num_dofs();
    const int n_iface = iface.num_points();

    if (iface.points_per_panel() != 6
        || iface.panel_node_layout() != PanelNodeLayout3D::QuadraticLagrange) {
        throw std::invalid_argument(
            "LaplaceQuadraticPatchCenterSpread3D requires P2 QuadraticLagrange panels");
    }
    if (static_cast<int>(jumps.size()) != n_iface)
        throw std::invalid_argument(
            "LaplaceQuadraticPatchCenterSpread3D jumps size must equal interface point count");
    if (rhs_correction.size() != n_grid)
        throw std::invalid_argument(
            "LaplaceQuadraticPatchCenterSpread3D rhs_correction size must equal grid DOF count");
    if (!edge_correction.singular_fields.empty()
        && correction_method_ !=
               LaplaceCorrectionMethod3D::NearestExpansionCenter) {
        throw std::invalid_argument(
            "edge-singular Cauchy regularization requires nearest-expansion-center correction");
    }
    for (const EdgePatchSingularLaplacian3D* field
         : edge_correction.singular_fields) {
        if (field == nullptr)
            throw std::invalid_argument(
                "edge spread correction contains a null singular field");
        if (&field->grid() != &grid)
            throw std::invalid_argument(
                "edge singular field and spread must use the same CartesianGrid3D object");
    }

    Eigen::VectorXd u_jump(n_iface);
    Eigen::VectorXd un_jump(n_iface);
    Eigen::VectorXd rhs_jump(n_iface);
    for (int q = 0; q < n_iface; ++q) {
        if (jumps[q].rhs_derivs.size() < 1) {
            throw std::invalid_argument(
                "LaplaceQuadraticPatchCenterSpread3D requires rhs_derivs[0] at every interface point");
        }
        u_jump[q] = jumps[q].u_jump;
        un_jump[q] = jumps[q].un_jump;
        rhs_jump[q] = jumps[q].rhs_derivs[0];
    }

    LaplaceSpreadResult3D result;
    result.correction_method = correction_method_;
    result.u_jump = u_jump;
    result.un_jump = un_jump;
    result.rhs_jump = rhs_jump;
    result.alpha = kappa_;

    if (correction_method_ == LaplaceCorrectionMethod3D::NearestExpansionCenter) {
        const PatchCenterCauchyResult3D cauchy =
            laplace_p2_patch_center_cauchy_3d(iface,
                                              u_jump,
                                              un_jump,
                                              rhs_jump,
                                              kappa_);

        std::vector<LocalPoly3D> center_polys(cauchy.centers.rows());
        for (int i = 0; i < cauchy.centers.rows(); ++i)
            center_polys[i] = center_poly(cauchy, i);

        std::vector<std::vector<int>> crossing_field_indices(
            support_.crossing_ops.size());
        std::unordered_map<int, std::vector<int>> fields_by_center;
        for (std::size_t op_idx = 0; op_idx < support_.crossing_ops.size();
             ++op_idx) {
            const LaplaceCrossingCorrectionOp& op =
                support_.crossing_ops[op_idx];
            const P2CrossingOwner3D owner =
                op_idx < crossing_owners_.size()
                    ? crossing_owners_[op_idx]
                    : grid_pair_.p2_crossing_owner_between(
                          op.rhs_node, op.correction_node);
            if (owner.center_index < 0)
                continue;
            const Eigen::Vector3d hit = crossing_point(grid, op, owner);
            std::vector<int>& active = crossing_field_indices[op_idx];
            for (int field_index = 0;
                 field_index
                     < static_cast<int>(edge_correction.singular_fields.size());
                 ++field_index) {
                if (edge_correction.singular_fields[
                        static_cast<std::size_t>(field_index)]->contains_point(hit)) {
                    active.push_back(field_index);
                    std::vector<int>& center_fields =
                        fields_by_center[owner.center_index];
                    if (std::find(center_fields.begin(),
                                  center_fields.end(),
                                  field_index) == center_fields.end()) {
                        center_fields.push_back(field_index);
                    }
                }
            }
        }

        // Restrict may use an expansion center that no crossed Cartesian edge
        // selected during spread.  Register every interface-point owner inside
        // an artificial edge cylinder so its half-correction is built from the
        // same singular-removed jump data.
        for (int q = 0; q < n_iface; ++q) {
            const Eigen::Vector3d interface_location =
                iface.points().row(q).transpose();
            const int center_index =
                grid_pair_.nearest_p2_expansion_center_for_interface_point(q);
            if (center_index < 0)
                continue;
            for (int field_index = 0;
                 field_index
                     < static_cast<int>(edge_correction.singular_fields.size());
                 ++field_index) {
                if (!edge_correction.singular_fields[
                         static_cast<std::size_t>(field_index)]
                         ->contains_point(interface_location)) {
                    continue;
                }
                std::vector<int>& center_fields =
                    fields_by_center[center_index];
                if (std::find(center_fields.begin(),
                              center_fields.end(),
                              field_index) == center_fields.end()) {
                    center_fields.push_back(field_index);
                }
            }
        }

        IndexedLocalPolys3D regularized_polys;
        regularized_polys.center_indices.reserve(fields_by_center.size());
        for (const auto& entry : fields_by_center)
            regularized_polys.center_indices.push_back(entry.first);
        std::sort(regularized_polys.center_indices.begin(),
                  regularized_polys.center_indices.end());

        if (!regularized_polys.center_indices.empty()) {
            const PatchCenterJumpModifier3D remove_edge_singular =
                [&fields_by_center,
                 &edge_correction,
                 &iface,
                 this](int center_index,
                       int panel,
                       Eigen::Vector3d barycentric,
                       Eigen::Vector3d,
                       Eigen::Vector3d,
                       double& value_jump,
                       double& normal_jump,
                       double& rhs_jump_value) {
                    const auto found = fields_by_center.find(center_index);
                    if (found == fields_by_center.end())
                        return;
                    double shape[6];
                    detail3d::p2_shape(barycentric, shape);
                    for (int field_index : found->second) {
                        const EdgePatchSingularLaplacian3D& field =
                            *edge_correction.singular_fields[
                                static_cast<std::size_t>(field_index)];
                        for (int local = 0; local < 6; ++local) {
                            const int q = iface.point_index(panel, local);
                            const Eigen::Vector3d node_point =
                                iface.points().row(q).transpose();
                            const Eigen::Vector3d node_normal =
                                iface.normals().row(q).transpose();
                            const double weight = shape[local];
                            value_jump -= weight
                                * field.singular_jump_value(node_point);
                            normal_jump -= weight
                                * field.singular_jump_normal_derivative(
                                      node_point, node_normal);
                            rhs_jump_value -= weight
                                * field.singular_jump_operator_value(
                                      node_point, kappa_);
                        }
                    }
                };
            const PatchCenterCauchyResult3D regularized_cauchy =
                laplace_p2_patch_center_cauchy_3d(
                    iface,
                    u_jump,
                    un_jump,
                    rhs_jump,
                    kappa_,
                    remove_edge_singular,
                    &regularized_polys.center_indices);
            regularized_polys.polys.resize(
                regularized_polys.center_indices.size());
            for (std::size_t pos = 0;
                 pos < regularized_polys.center_indices.size();
                 ++pos) {
                regularized_polys.polys[pos] = center_poly(
                    regularized_cauchy,
                    regularized_polys.center_indices[pos]);
            }
        }

        for (std::size_t op_idx = 0; op_idx < support_.crossing_ops.size();
             ++op_idx) {
            const LaplaceCrossingCorrectionOp& op =
                support_.crossing_ops[op_idx];
            const Eigen::Vector3d pt = node_coord(grid, op.correction_node);
            const int center_idx =
                op_idx < crossing_owners_.size()
                    ? crossing_owners_[op_idx].center_index
                    : grid_pair_.nearest_p2_expansion_center_between(
                          op.rhs_node, op.correction_node);
            if (center_idx < 0
                || center_idx >= static_cast<int>(center_polys.size())) {
                throw std::runtime_error(
                    "LaplaceQuadraticPatchCenterSpread3D invalid crossing expansion center");
            }
            const LocalPoly3D* correction_poly = &center_polys[center_idx];
            if (!crossing_field_indices[op_idx].empty()) {
                const LocalPoly3D* regularized =
                    find_indexed_local_poly_3d(regularized_polys, center_idx);
                if (regularized == nullptr) {
                    throw std::runtime_error(
                        "edge crossing has no singular-removed Cauchy polynomial");
                }
                correction_poly = regularized;
            }
            const double correction =
                evaluate_taylor_poly_3d(*correction_poly, pt);
            rhs_correction[op.rhs_node] += static_cast<double>(op.side_delta)
                                         * correction
                                         * op.stencil_weight;
        }

        if (edge_correction.accumulate_volume_rhs) {
            for (const EdgePatchSingularLaplacian3D* field
                 : edge_correction.singular_fields) {
                Eigen::VectorXd edge_rhs =
                    field->evaluate_rhs_correction_all();
                // evaluate_rhs_correction_all() supplies Delta A for the
                // Laplace part.  For L = -Delta + kappa and u = w + A,
                // the regular variable satisfies
                //     L w = f + Delta A - kappa A.
                if (kappa_ != 0.0) {
                    for (int node = 0; node < n_grid; ++node)
                        edge_rhs[node] -= kappa_ * field->field_value(node);
                }
                rhs_correction += edge_rhs;
            }
        }

        result.edge_regularized_correction_polys =
            std::move(regularized_polys);
        result.edge_regularization_patches.reserve(
            edge_correction.singular_fields.size());
        result.edge_restrict_data.reserve(
            edge_correction.singular_fields.size());
        for (const EdgePatchSingularLaplacian3D* field
             : edge_correction.singular_fields) {
            result.edge_regularization_patches.push_back(field->patch());

            EdgePatchRestrictData3D restrict_data;
            restrict_data.patch = field->patch();
            restrict_data.active_domain_label = field->active_domain_label();
            restrict_data.physical_grid_values =
                Eigen::VectorXd::Zero(n_grid);
            for (int node = 0; node < n_grid; ++node)
                restrict_data.physical_grid_values[node] =
                    field->field_value(node);
            restrict_data.continuation_grid_values =
                Eigen::VectorXd::Zero(n_grid);
            for (int node : support_.correction_nodes) {
                if (grid_pair_.domain_label(node)
                    != restrict_data.active_domain_label) {
                    continue;
                }
                restrict_data.continuation_grid_values[node] =
                    field->continuation_field_value(node_coord(grid, node));
            }
            restrict_data.interface_values = Eigen::VectorXd::Zero(n_iface);
            restrict_data.interface_gradients =
                Eigen::MatrixX3d::Zero(n_iface, 3);
            for (int q = 0; q < n_iface; ++q) {
                const Eigen::Vector3d interface_location =
                    iface.points().row(q).transpose();
                restrict_data.interface_values[q] =
                    field->physical_field_value(interface_location);
                restrict_data.interface_gradients.row(q) =
                    field->physical_field_gradient_at(interface_location)
                        .transpose();
            }
            result.edge_restrict_data.push_back(std::move(restrict_data));
        }

        result.correction_polys = std::move(center_polys);
        return result;
    }

    if (correction_method_ == LaplaceCorrectionMethod3D::ProjectionPoint) {
        result.projection_cache = projection_cache_;

        const ProfileClock::time_point correction_start = ProfileClock::now();
        for (const LaplaceCrossingCorrectionOp& op : support_.crossing_ops) {
            const double correction =
                evaluate_projection_point_correction_3d(grid_pair_,
                                                        result.projection_cache,
                                                        op.correction_node,
                                                        result);
            rhs_correction[op.rhs_node] += static_cast<double>(op.side_delta)
                                         * correction
                                         * op.stencil_weight;
        }
        const double t_correction = seconds_since(correction_start);

        if (profile_projection_transfer_3d()) {
            std::printf("      projection support nodes=%zu crossing_edges=%zu restrict_sample_visits=%d correction %.3fs\n",
                        result.projection_cache.nodes().size(),
                        support_.crossing_ops.size(),
                        support_.restrict_sample_visits,
                        t_correction);
        }

        return result;
    }

    throw std::invalid_argument(
        "LaplaceQuadraticPatchCenterSpread3D unsupported correction method");
}

} // namespace kfbim
