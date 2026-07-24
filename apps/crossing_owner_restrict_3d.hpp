#pragma once

#include "native_nurbs_surface_3d.hpp"
#include "src/geometry/nurbs_surface_intersector_3d.hpp"

#include <Eigen/Dense>

namespace kfbim::app3d {

enum class RestrictOwnerDecisionKind3D {
    TargetSideNode,
    TargetOrG1SingleCrossing,
    ForeignNonG1SingleCrossing,
    NoCrossingFallback,
    MultipleCrossingFallback,
    DegenerateCrossingFallback,
    AmbiguousEdgeFallback
};

struct RestrictOwnerDecision3D {
    int owner_dof = -1;
    RestrictOwnerDecisionKind3D kind =
        RestrictOwnerDecisionKind3D::TargetSideNode;
    int crossing_patch = -1;
    double segment_parameter = 0.0;
    double residual = 0.0;
    double transversality = 0.0;
};

RestrictOwnerDecision3D select_restrict_correction_owner_3d(
    int target_dof,
    const Eigen::Vector3d& query,
    const Eigen::Vector3d& support,
    const NativeNurbsSurface3D& surface,
    const SurfaceDofCloud3D& cloud,
    const geometry3d::NurbsSurfaceIntersectionResult3D& intersection);

} // namespace kfbim::app3d
