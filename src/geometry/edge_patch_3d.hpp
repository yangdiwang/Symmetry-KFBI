#pragma once

#include <Eigen/Dense>

#include <array>
#include <vector>

namespace kfbim::geometry3d {

// Straight feature-edge data used to construct an artificial cylindrical
// region.  The radius is a physical length; it is deliberately independent
// of the Cartesian-grid spacing.
struct EdgePatchSource3D {
    Eigen::Vector3d start = Eigen::Vector3d::Zero();
    Eigen::Vector3d end = Eigen::Vector3d::Zero();
    std::array<int, 2> incident_patches{{-1, -1}};
};

struct EdgePatch3D {
    int edge = -1;
    Eigen::Vector3d start = Eigen::Vector3d::Zero();
    Eigen::Vector3d end = Eigen::Vector3d::Zero();
    Eigen::Vector3d tangent = Eigen::Vector3d::Zero();
    double length = 0.0;
    double radius = 0.0;
    std::array<int, 2> incident_patches{{-1, -1}};
};

struct EdgePatchOptions3D {
    double geometry_tol = 1.0e-12;
};

// This mirrors CornerPatchConfig2D.  A positive radius is interpreted in
// physical coordinates and is never multiplied by h.  Positive finite
// per-edge entries replace the common radius.
struct EdgePatchConfig3D {
    bool enabled = false;
    double radius = 0.0;
    std::vector<double> radius_overrides_by_edge;
    EdgePatchOptions3D options;
};

struct EdgePatchCoordinates3D {
    // Physical arclength from start along the edge tangent.
    double s = 0.0;
    // Distance from the infinite edge axis.
    double rho = 0.0;
    Eigen::Vector3d axis_point = Eigen::Vector3d::Zero();
};

[[nodiscard]] std::vector<EdgePatch3D> build_edge_patches_3d(
    const std::vector<EdgePatchSource3D>& edges,
    const EdgePatchConfig3D& config);

[[nodiscard]] EdgePatchCoordinates3D edge_patch_coordinates_3d(
    const EdgePatch3D& patch,
    const Eigen::Vector3d& point);

// The artificial region is the finite open-axis cylinder 0 <= s <= length,
// 0 < rho <= radius.  Excluding the singular axis matches the 2D corner
// patch convention, while the end sections remain available for a later
// vertex correction to overlap additively.
[[nodiscard]] bool edge_patch_contains_point_3d(
    const EdgePatch3D& patch,
    const Eigen::Vector3d& point,
    double tolerance = 1.0e-12);

[[nodiscard]] double edge_patch_radial_signed_distance_3d(
    const EdgePatch3D& patch,
    const Eigen::Vector3d& point);

} // namespace kfbim::geometry3d
