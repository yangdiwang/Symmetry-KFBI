#pragma once

#include "src/geometry/grid_pair_3d.hpp"
#include "src/geometry/nurbs_surface_closest_point_3d.hpp"
#include "src/support/geometry/native_nurbs_surface_3d.hpp"

namespace kfbim::app3d {

enum class ClosestPointTraceAnchorStatus3D {
    Accepted,
    InvalidInput,
    UnresolvedProjection,
    AmbiguousProjection,
    IncompatibleSheet,
    FeatureContact,
    NonNormalProjection,
    TooFar
};

struct ClosestPointTraceAnchor3D {
    ClosestPointTraceAnchorStatus3D status = ClosestPointTraceAnchorStatus3D::InvalidInput;
    P2CrossingOwner3D owner;
    double distance = 0.0;
    double localization_radius = 0.0;

    [[nodiscard]] bool accepted() const noexcept
    {
        return status == ClosestPointTraceAnchorStatus3D::Accepted;
    }
};

[[nodiscard]] const char* closest_point_trace_anchor_status_name_3d(
    ClosestPointTraceAnchorStatus3D status) noexcept;

// Select the supplied closest point unchanged, for a local smooth-sheet jump
// extension evaluated at support_point. trace_point must belong to target_patch.
// Acceptance is a bounded numerical/locality check, not a uniqueness theorem.
// C0 contacts, other sheets and unresolved projections remain explicit refusals.
// A target or owner patch with an internal knot of multiplicity >= degree is
// conservatively refused in full; parameter-side smooth regions are not inferred.
// The owner is reused as an exact native expansion-point carrier: its residual
// measures surface-point consistency, not support_point-to-surface distance.
[[nodiscard]] ClosestPointTraceAnchor3D select_closest_point_trace_anchor_3d(
    const NativeNurbsSurface3D& surface,
    const geometry3d::NurbsSurfaceClosestPointResult3D& projection,
    const Eigen::Vector3d& support_point,
    const Eigen::Vector3d& trace_point,
    int target_patch,
    const Eigen::Vector3d& target_normal,
    double maximum_distance);

} // namespace kfbim::app3d
