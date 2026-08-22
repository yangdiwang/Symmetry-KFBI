#pragma once

#include <array>
#include <cstddef>
#include <map>
#include <stdexcept>
#include <vector>

#include "../geometry/corner_patch_2d.hpp"
#include "../geometry/grid_pair_2d.hpp"
#include "../geometry/grid_pair_3d.hpp"
#include "../geometry/nurbs_cartesian_domain_3d.hpp"
#include "../grid/structured_grid_ops.hpp"
#include "laplace_correction_op.hpp"
#include "laplace_grid_edge_event_correction_3d.hpp"

namespace kfbim {

enum class LaplaceJumpCorrectionSource2D {
    RawInterface,
    PatchRegularizedInterface,
    PatchBoundarySingular
};

enum class PatchInterfaceJumpMode2D {
    PhysicalJump,
    SingularRemovedJump
};

inline P2CrossingOwner3D laplace_crossing_owner_3d(
    const GridPair3D& grid_pair,
    const LaplaceCrossingCorrectionOp& op)
{
    return has_grid_edge_event_3d(op)
        ? grid_pair.p2_crossing_owner_for_grid_edge_event(
              op.grid_edge_event)
        : grid_pair.p2_crossing_owner_between(
              op.rhs_node, op.correction_node);
}

inline LaplaceJumpCorrectionSource2D laplace_crossing_source_2d(
    const LaplaceCrossingCorrectionOp& op)
{
    if (op.kind == LaplaceCrossingKind::CornerPatchBoundary)
        return LaplaceJumpCorrectionSource2D::PatchBoundarySingular;
    return op.patch >= 0
             ? LaplaceJumpCorrectionSource2D::PatchRegularizedInterface
             : LaplaceJumpCorrectionSource2D::RawInterface;
}

struct LaplaceCorrectionSupport2D {
    std::vector<LaplaceCrossingCorrectionOp> crossing_ops;
    std::vector<std::array<int, 6>> restrict_stencils;
    std::vector<int> correction_nodes;
    std::vector<int> projection_nodes;
    int restrict_sample_visits = 0;
};

struct LaplaceCorrectionSupport3D {
    std::vector<LaplaceCrossingCorrectionOp> crossing_ops;
    // Event-aware native NURBS catalog.  crossing_ops contains two directed
    // entries for every retained physical event; the legacy label-change scan
    // is used only when no native Cartesian domain is attached.
    std::vector<geometry3d::GridEdgeEvent3D>
        certified_grid_edge_events;
    std::vector<std::array<int, 10>> restrict_stencils;
    std::vector<int> correction_nodes;
    std::vector<int> projection_nodes;
    int restrict_sample_visits = 0;
    std::size_t certified_label_preserving_event_count = 0;
};

namespace laplace_correction_support_detail {

inline int side_from_label(int label)
{
    return label == 0 ? 0 : 1;
}

inline Eigen::Vector2d interface_point(const Interface2D& iface, int q)
{
    return iface.points().row(q).transpose();
}

inline Eigen::Vector3d interface_point(const Interface3D& iface, int q)
{
    return iface.points().row(q).transpose();
}

inline void append_unique_projection_nodes(const std::vector<char>& mark,
                                           std::vector<int>&        nodes)
{
    nodes.clear();
    for (int idx = 0; idx < static_cast<int>(mark.size()); ++idx) {
        if (mark[idx])
            nodes.push_back(idx);
    }
}

} // namespace laplace_correction_support_detail

inline LaplaceCorrectionSupport2D build_laplace_correction_support_2d(
    const GridPair2D& grid_pair,
    const char*       context)
{
    const auto& grid = grid_pair.grid();
    const auto& iface = grid_pair.interface();
    const int n_grid = grid.num_dofs();
    std::vector<char> needs_c(n_grid, 0);

    LaplaceCorrectionSupport2D support;
    for (int n = 0; n < n_grid; ++n) {
        if (structured_grid::is_boundary_node(grid, n))
            continue;

        const int side_n =
            laplace_correction_support_detail::side_from_label(
                grid_pair.domain_label(n));
        const auto neighbors = grid.neighbors(n);
        for (int slot = 0; slot < 4; ++slot) {
            const int nb = neighbors[slot];
            if (nb < 0 || structured_grid::is_boundary_node(grid, nb))
                continue;

            const int side_nb =
                laplace_correction_support_detail::side_from_label(
                    grid_pair.domain_label(nb));
            const int patch_n = grid_pair.corner_patch_label(n);
            const int patch_nb = grid_pair.corner_patch_label(nb);
            if (side_nb != side_n) {
                const int patch_id = patch_n >= 0 ? patch_n : patch_nb;
                needs_c[n] = 1;
                needs_c[nb] = 1;
                LaplaceCrossingCorrectionOp op;
                op.kind = LaplaceCrossingKind::InterfaceJump;
                op.rhs_node = n;
                op.correction_node = nb;
                op.neighbor_slot = slot;
                op.side_delta = side_n - side_nb;
                op.patch = patch_id;
                op.stencil_weight =
                    structured_grid::stencil_weight_for_neighbor(grid, slot);
                op.rhs_patch = patch_n;
                op.correction_patch = patch_nb;
                support.crossing_ops.push_back(op);
            }

            // Corner-patch boundary corrections are same-phase decomposition
            // corrections. Edges crossing the physical interface use only the
            // regularized interface jump correction.
            if (patch_n != patch_nb && side_n == side_nb) {
                const int patch_id = patch_n >= 0 ? patch_n : patch_nb;

                needs_c[n] = 1;
                needs_c[nb] = 1;
                const int in_patch_n = patch_n >= 0 ? 1 : 0;
                const int in_patch_nb = patch_nb >= 0 ? 1 : 0;
                LaplaceCrossingCorrectionOp op;
                op.kind = LaplaceCrossingKind::CornerPatchBoundary;
                op.rhs_node = n;
                op.correction_node = nb;
                op.neighbor_slot = slot;
                op.side_delta = in_patch_n - in_patch_nb;
                op.patch = patch_id;
                op.stencil_weight =
                    structured_grid::stencil_weight_for_neighbor(grid, slot);
                op.rhs_patch = patch_n;
                op.correction_patch = patch_nb;
                support.crossing_ops.push_back(op);
            }
        }
    }

    support.restrict_stencils.reserve(iface.num_points());
    for (int q = 0; q < iface.num_points(); ++q) {
        const int closest = grid_pair.closest_bulk_node(q);
        const auto stencil =
            structured_grid::quadratic_restrict_stencil_nodes_2d(
                context,
                grid,
                closest,
                laplace_correction_support_detail::interface_point(iface, q));
        support.restrict_stencils.push_back(stencil);
        for (int node : stencil) {
            needs_c[node] = 1;
            ++support.restrict_sample_visits;
        }
    }

    laplace_correction_support_detail::append_unique_projection_nodes(
        needs_c, support.correction_nodes);
    support.projection_nodes = support.correction_nodes;
    return support;
}

inline LaplaceCorrectionSupport3D build_laplace_correction_support_3d(
    const GridPair3D& grid_pair,
    const char*       context)
{
    const auto& grid = grid_pair.grid();
    const auto& iface = grid_pair.interface();
    const int n_grid = grid.num_dofs();
    std::vector<char> needs_c(n_grid, 0);

    LaplaceCorrectionSupport3D support;
    if (const geometry3d::NurbsCartesianDomain3D* domain =
            grid_pair.nurbs_cartesian_domain()) {
        if (domain->diagnostics().uncertified_grid_edge_count != 0) {
            throw std::runtime_error(
                "native multi-event correction requires every candidate grid edge to have a certified complete root set");
        }
        for (const geometry3d::GridEdgeEvent3D& event :
             domain->grid_edge_event_catalog()) {
            const int first = event.id.first_node;
            const int second = event.id.second_node;
            const std::vector<LaplaceCrossingCorrectionOp> event_ops =
                native_grid_edge_event_correction_ops_3d(grid, event);
            if (event_ops.empty())
                continue;

            support.certified_grid_edge_events.push_back(event);
            const int first_side =
                laplace_correction_support_detail::side_from_label(
                    grid_pair.domain_label(first));
            const int second_side =
                laplace_correction_support_detail::side_from_label(
                    grid_pair.domain_label(second));
            if (first_side == second_side)
                ++support.certified_label_preserving_event_count;

            support.crossing_ops.insert(
                support.crossing_ops.end(),
                event_ops.begin(), event_ops.end());
            needs_c[first] = 1;
            needs_c[second] = 1;
        }
    } else for (int n = 0; n < n_grid; ++n) {
        if (structured_grid::is_boundary_node(grid, n))
            continue;

        const int side_n =
            laplace_correction_support_detail::side_from_label(
                grid_pair.domain_label(n));
        const auto neighbors = grid.neighbors(n);
        for (int slot = 0; slot < 6; ++slot) {
            const int nb = neighbors[slot];
            if (nb < 0 || structured_grid::is_boundary_node(grid, nb))
                continue;

            const int side_nb =
                laplace_correction_support_detail::side_from_label(
                    grid_pair.domain_label(nb));
            if (side_nb == side_n)
                continue;

            needs_c[nb] = 1;
            LaplaceCrossingCorrectionOp op;
            op.kind = LaplaceCrossingKind::InterfaceJump;
            op.rhs_node = n;
            op.correction_node = nb;
            op.neighbor_slot = slot;
            op.side_delta = side_n - side_nb;
            op.patch = -1;
            op.stencil_weight =
                structured_grid::stencil_weight_for_neighbor(grid, slot);
            support.crossing_ops.push_back(op);
        }
    }

    support.restrict_stencils.reserve(iface.num_points());
    for (int q = 0; q < iface.num_points(); ++q) {
        const int closest = grid_pair.closest_bulk_node(q);
        const auto stencil =
            structured_grid::quadratic_restrict_stencil_nodes_3d(
                context,
                grid,
                closest,
                laplace_correction_support_detail::interface_point(iface, q));
        support.restrict_stencils.push_back(stencil);
        for (int node : stencil) {
            needs_c[node] = 1;
            ++support.restrict_sample_visits;
        }
    }

    laplace_correction_support_detail::append_unique_projection_nodes(
        needs_c, support.correction_nodes);
    support.projection_nodes = support.correction_nodes;
    return support;
}

} // namespace kfbim
