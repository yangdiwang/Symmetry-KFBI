#pragma once

#include "src/geometry/nurbs_patch_triangulator_3d.hpp"
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

struct SurfaceDof3D {
    Eigen::Vector3d point = Eigen::Vector3d::Zero();
    Eigen::Vector3d normal = Eigen::Vector3d::Zero();
    Eigen::Vector3d tangent1 = Eigen::Vector3d::Zero();
    Eigen::Vector3d tangent2 = Eigen::Vector3d::Zero();
    double weight = 0.0;
    double u = 0.0;
    double v = 0.0;
    int patch_id = -1;
    int i = -1;
    int j = -1;
};

struct SurfaceDofPatch3D {
    std::string name;
    int nu = 0;
    int nv = 0;
    int first_dof = 0;
    std::vector<int> smooth_patch_ids;

    [[nodiscard]] int dof_index(int i, int j) const
    {
        return first_dof + i * nv + j;
    }

    [[nodiscard]] int dof_count() const
    {
        return nu * nv;
    }
};

struct SurfaceDofCloud3D {
    std::vector<SurfaceDofPatch3D> patches;
    std::vector<SurfaceDof3D> dofs;
    double expected_area = 0.0;
};

[[nodiscard]] NativeNurbsSurface3D make_native_nurbs_surface_3d(
    GeometryKind3D kind);

// Exact 16-patch rational-biquadratic torus; historical defaults unchanged.
[[nodiscard]] NativeNurbsSurface3D make_native_nurbs_torus_3d(
    double major_radius, double minor_radius,
    const Eigen::Vector3d& center = Eigen::Vector3d::Zero());

[[nodiscard]] SurfaceDofCloud3D make_native_surface_dofs_3d(
    const NativeNurbsSurface3D& surface,
    double h);

[[nodiscard]] std::vector<int> nearest_topological_cauchy_dofs(
    const NativeNurbsSurface3D& surface,
    const SurfaceDofCloud3D& cloud,
    int center_dof,
    int count);

[[nodiscard]] std::vector<int> nearest_g1_cauchy_dofs(
    const NativeNurbsSurface3D& surface,
    const SurfaceDofCloud3D& cloud,
    int center_dof,
    int count);

[[nodiscard]] std::vector<int> nearest_same_patch_cauchy_dofs(
    const SurfaceDofCloud3D& cloud,
    int center_dof,
    int count);

// Return every patch joined to `patch` by a physical G1 connection.  The
// canonical source is geometric_connections, whose edge intervals can
// represent a long edge joined to several short edges.  The legacy
// one-neighbor-per-edge table is retained as a fallback for older synthetic
// surfaces which do not carry geometric connection records.
[[nodiscard]] std::vector<int> smooth_patch_neighbors_3d(
    const NativeNurbsSurface3D& surface,
    int patch);

[[nodiscard]] std::vector<int> balanced_topological_cauchy_dofs(
    const NativeNurbsSurface3D& surface,
    const SurfaceDofCloud3D& cloud,
    int center_dof,
    int count);

[[nodiscard]] std::array<int, 4> parameter_dof_candidates_2x2(
    const NativeNurbsSurface3D& surface,
    const SurfaceDofCloud3D& cloud,
    int patch,
    double u,
    double v);

[[nodiscard]] Eigen::Vector2d interpolate_triangle_parameter(
    const geometry3d::NurbsParamTriangle3D& triangle,
    const Eigen::Vector3d& barycentric);

[[nodiscard]] std::vector<int> smooth_patch_component(
    const NativeNurbsSurface3D& surface,
    int patch);

} // namespace kfbim::app3d
