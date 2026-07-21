#pragma once

#include "src/geometry/nurbs_patch_triangulator_3d.hpp"

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
    LPrism
};

enum class PatchEdge3D {
    UMin = 0,
    UMax = 1,
    VMin = 2,
    VMax = 3
};

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
    double expected_area = 0.0;
    std::function<bool(const Eigen::Vector3d&)> exact_inside;
};

[[nodiscard]] NativeNurbsSurface3D make_native_nurbs_surface_3d(
    GeometryKind3D kind);

} // namespace kfbim::app3d
