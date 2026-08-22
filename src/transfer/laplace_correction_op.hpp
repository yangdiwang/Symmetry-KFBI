#pragma once

#include "../geometry/nurbs_cartesian_domain_3d.hpp"

namespace kfbim {

enum class LaplaceCrossingKind {
    InterfaceJump,
    CornerPatchBoundary
};

struct LaplaceCrossingCorrectionOp {
    LaplaceCrossingKind kind = LaplaceCrossingKind::InterfaceJump;
    int rhs_node = -1;
    int correction_node = -1;
    int neighbor_slot = -1;
    int side_delta = 0;
    int patch = -1;
    double stencil_weight = 0.0;
    int rhs_patch = -1;
    int correction_patch = -1;
    // Set only for native 3D NURBS Cartesian events.  The id is canonical and
    // direction independent; side_delta carries this directed op's sign.
    geometry3d::GridEdgeEventId3D grid_edge_event;
};

inline bool has_grid_edge_event_3d(
    const LaplaceCrossingCorrectionOp& op) noexcept
{
    return op.grid_edge_event.first_node >= 0
        && op.grid_edge_event.second_node >= 0
        && op.grid_edge_event.ordinal >= 0;
}

} // namespace kfbim
