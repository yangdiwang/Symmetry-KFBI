#pragma once

#include "laplace_correction_op.hpp"
#include "../grid/structured_grid_ops.hpp"

#include <stdexcept>
#include <vector>

namespace kfbim {

// Convert one canonical physical event into the two directed finite-
// difference corrections that share its immutable id.
inline std::vector<LaplaceCrossingCorrectionOp>
native_grid_edge_event_correction_ops_3d(
    const CartesianGrid3D& grid,
    const geometry3d::GridEdgeEvent3D& event)
{
    const int n_grid = grid.num_dofs();
    const int first = event.id.first_node;
    const int second = event.id.second_node;
    if (first < 0 || first >= n_grid
        || second < 0 || second >= n_grid || first >= second) {
        throw std::logic_error(
            "native grid-edge event has invalid canonical endpoints");
    }
    if (structured_grid::is_boundary_node(grid, first)
        || structured_grid::is_boundary_node(grid, second)) {
        return {};
    }
    if (!event.certified_transverse()) {
        throw std::runtime_error(
            "native multi-event correction encountered an uncertified grid-edge event");
    }
    if ((event.canonical_sign != -1 && event.canonical_sign != 1)
        || !event.point.allFinite()) {
        throw std::logic_error(
            "certified native grid-edge event has invalid signed geometry");
    }
    const auto neighbor_slot = [&grid](int node, int neighbor) {
        const auto neighbors = grid.neighbors(node);
        for (int slot = 0; slot < 6; ++slot) {
            if (neighbors[static_cast<std::size_t>(slot)] == neighbor)
                return slot;
        }
        throw std::logic_error(
            "native grid-edge event endpoints are not Cartesian neighbors");
    };

    const int forward_slot = neighbor_slot(first, second);
    const int reverse_slot = neighbor_slot(second, first);
    LaplaceCrossingCorrectionOp forward;
    forward.kind = LaplaceCrossingKind::InterfaceJump;
    forward.rhs_node = first;
    forward.correction_node = second;
    forward.neighbor_slot = forward_slot;
    forward.side_delta = event.canonical_sign;
    forward.stencil_weight =
        structured_grid::stencil_weight_for_neighbor(grid, forward_slot);
    forward.grid_edge_event = event.id;

    LaplaceCrossingCorrectionOp reverse = forward;
    reverse.rhs_node = second;
    reverse.correction_node = first;
    reverse.neighbor_slot = reverse_slot;
    reverse.side_delta = -event.canonical_sign;
    reverse.stencil_weight =
        structured_grid::stencil_weight_for_neighbor(grid, reverse_slot);
    return {forward, reverse};
}

} // namespace kfbim
