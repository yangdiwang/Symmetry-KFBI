#pragma once

#include "src/geometry/nurbs_surface_model_3d.hpp"

#include <Eigen/Dense>

#include <array>
#include <functional>
#include <optional>
#include <string>
#include <vector>

namespace kfbim::app3d {

enum class GeometryKind3D {
    Torus,
    HollowCylinder,
    LPrism,
    UPrism
};

using PatchEdge3D = geometry3d::NurbsPatchEdge3D;

struct SmoothPatchNeighbor3D {
    int patch = -1;
    PatchEdge3D edge = PatchEdge3D::UMin;
    bool reversed = false;
};

struct NativeNurbsSurface3D {
    std::string name;
    std::string description;
    std::vector<geometry3d::NurbsSurfacePatch3D> patches;
    std::vector<std::string> patch_names;
    std::vector<std::array<std::optional<SmoothPatchNeighbor3D>, 4>>
        smooth_neighbors;
    std::vector<std::vector<int>> topological_patch_neighbors;
    std::vector<geometry3d::NurbsPatchEdgeConnection3D>
        geometric_connections;
    std::vector<int> patch_components;
    double expected_area = 0.0;
    std::function<bool(const Eigen::Vector3d&)> exact_inside;

    [[nodiscard]] geometry3d::NurbsSurfaceModel3D geometry_model() const;
};

[[nodiscard]] NativeNurbsSurface3D make_native_nurbs_surface_3d(
    GeometryKind3D kind);

// Exact 16-patch rational-biquadratic torus; historical defaults unchanged.
[[nodiscard]] NativeNurbsSurface3D make_native_nurbs_torus_3d(
    double major_radius, double minor_radius,
    const Eigen::Vector3d& center = Eigen::Vector3d::Zero());

} // namespace kfbim::app3d
