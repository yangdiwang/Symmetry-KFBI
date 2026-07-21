#include "native_nurbs_surface_3d.hpp"

#include <cmath>
#include <iostream>
#include <set>
#include <stdexcept>
#include <string>

namespace {

using kfbim::app3d::GeometryKind3D;
using kfbim::app3d::NativeNurbsSurface3D;
using kfbim::app3d::PatchEdge3D;
using kfbim::app3d::SmoothPatchNeighbor3D;
using kfbim::app3d::SurfaceDofCloud3D;
using kfbim::app3d::interpolate_triangle_parameter;
using kfbim::app3d::make_native_nurbs_surface_3d;
using kfbim::app3d::make_native_surface_dofs_3d;
using kfbim::app3d::parameter_dof_candidates_2x2;
using kfbim::app3d::smooth_patch_component;

void require(bool condition, const std::string& message)
{
    if (!condition)
        throw std::runtime_error(message);
}

void check_patch_regular(const NativeNurbsSurface3D& surface)
{
    require(surface.patch_names.size() == surface.patches.size(),
            surface.name + " patch-name count");
    require(surface.smooth_neighbors.size() == surface.patches.size(),
            surface.name + " smooth-neighbor count");
    for (const auto& patch : surface.patches) {
        const double u = 0.5 * (patch.domain_start_u() + patch.domain_end_u());
        const double v = 0.5 * (patch.domain_start_v() + patch.domain_end_v());
        const auto d = patch.evaluate_with_derivatives(u, v);
        require(d.point.allFinite() && d.du.allFinite() && d.dv.allFinite(),
                surface.name + " finite patch evaluation");
        require(d.du.cross(d.dv).norm() > 1.0e-12,
                surface.name + " nondegenerate patch Jacobian");
    }
}

void test_native_models()
{
    constexpr double pi = 3.141592653589793238462643383279502884;
    const NativeNurbsSurface3D torus =
        make_native_nurbs_surface_3d(GeometryKind3D::Torus);
    const NativeNurbsSurface3D cylinder =
        make_native_nurbs_surface_3d(GeometryKind3D::HollowCylinder);
    const NativeNurbsSurface3D lprism =
        make_native_nurbs_surface_3d(GeometryKind3D::LPrism);

    require(torus.patches.size() == 16,
            "torus must have 4x4 native patches");
    require(cylinder.patches.size() == 16,
            "cylinder must have four quarters on each of four sheets");
    require(lprism.patches.size() == 12,
            "L-prism must retain its twelve native patches");
    check_patch_regular(torus);
    check_patch_regular(cylinder);
    check_patch_regular(lprism);

    for (const auto& patch : torus.patches) {
        const double u = 0.5 * (patch.domain_start_u() + patch.domain_end_u());
        const double v = 0.5 * (patch.domain_start_v() + patch.domain_end_v());
        const Eigen::Vector3d p = patch.evaluate(u, v);
        const double rho = std::hypot(p.x() - 0.07, p.y() + 0.04);
        const double residual = (rho - 0.55) * (rho - 0.55)
                              + (p.z() - 0.03) * (p.z() - 0.03)
                              - 0.20 * 0.20;
        require(std::abs(residual) < 1.0e-11,
                "torus NURBS patch must be exact");
    }
    require(std::abs(torus.expected_area - 4.0 * pi * pi * 0.55 * 0.20)
                < 1.0e-13,
            "torus exact area");

    for (std::size_t patch_id = 0; patch_id < cylinder.patches.size(); ++patch_id) {
        const auto& patch = cylinder.patches[patch_id];
        const Eigen::Vector3d p = patch.evaluate(
            0.5 * (patch.domain_start_u() + patch.domain_end_u()),
            0.5 * (patch.domain_start_v() + patch.domain_end_v()));
        const double radius = std::hypot(p.x() - 0.06, p.y() + 0.05);
        if (patch_id < 4)
            require(std::abs(radius - 0.55) < 1.0e-12, "outer cylinder wall");
        else if (patch_id < 8)
            require(std::abs(radius - 0.25) < 1.0e-12, "inner cylinder wall");
        else if (patch_id < 12)
            require(std::abs(p.z() - 0.67) < 1.0e-12, "top annulus");
        else
            require(std::abs(p.z() + 0.63) < 1.0e-12, "bottom annulus");
    }

    const double lprism_area = 2.0 * 3.0 * 0.60 * 0.60
                             + 8.0 * 0.60 * 1.30;
    require(std::abs(lprism.expected_area - lprism_area) < 1.0e-13,
            "L-prism exact area");
}

double dof_area(const SurfaceDofCloud3D& cloud)
{
    double result = 0.0;
    for (const auto& dof : cloud.dofs)
        result += dof.weight;
    return result;
}

double check_dof_cloud(const NativeNurbsSurface3D& surface, double h)
{
    const SurfaceDofCloud3D cloud = make_native_surface_dofs_3d(surface, h);
    require(cloud.patches.size() == surface.patches.size(),
            surface.name + " one DOF patch per NURBS patch");
    require(!cloud.dofs.empty(), surface.name + " nonempty DOF cloud");
    for (const auto& tensor_patch : cloud.patches) {
        require(tensor_patch.nu >= 2 && tensor_patch.nv >= 2,
                surface.name + " at least two cells per parameter direction");
        require(tensor_patch.dof_count() == tensor_patch.nu * tensor_patch.nv,
                surface.name + " dense tensor indexing");
    }
    for (const auto& dof : cloud.dofs) {
        require(dof.patch_id >= 0
                    && dof.patch_id < static_cast<int>(surface.patches.size()),
                surface.name + " valid native patch ownership");
        const auto& patch = surface.patches[static_cast<std::size_t>(dof.patch_id)];
        require(dof.u > patch.domain_start_u()
                    && dof.u < patch.domain_end_u(),
                surface.name + " strict interior u center");
        require(dof.v > patch.domain_start_v()
                    && dof.v < patch.domain_end_v(),
                surface.name + " strict interior v center");
        require((dof.point - patch.evaluate(dof.u, dof.v)).norm() < 1.0e-13,
                surface.name + " native point evaluation");
        require(std::abs(dof.normal.norm() - 1.0) < 1.0e-13,
                surface.name + " unit normal");
        require(std::abs(dof.tangent1.norm() - 1.0) < 1.0e-13
                    && std::abs(dof.tangent2.norm() - 1.0) < 1.0e-13
                    && std::abs(dof.tangent1.dot(dof.tangent2)) < 1.0e-13
                    && std::abs(dof.normal.dot(dof.tangent1)) < 1.0e-13
                    && std::abs(dof.normal.dot(dof.tangent2)) < 1.0e-13,
                surface.name + " orthonormal local frame");
        require(dof.weight > 0.0 && std::isfinite(dof.weight),
                surface.name + " positive finite midpoint weight");
        const double offset = 0.1 * h;
        require(surface.exact_inside(dof.point - offset * dof.normal),
                surface.name + " inward normal enters domain");
        require(!surface.exact_inside(dof.point + offset * dof.normal),
                surface.name + " outward normal exits domain");
    }
    return std::abs(dof_area(cloud) - surface.expected_area)
         / surface.expected_area;
}

void test_uniform_native_dofs()
{
    for (GeometryKind3D kind : {GeometryKind3D::Torus,
                                GeometryKind3D::HollowCylinder,
                                GeometryKind3D::LPrism}) {
        const NativeNurbsSurface3D surface = make_native_nurbs_surface_3d(kind);
        const double coarse_error = check_dof_cloud(surface, 3.0 / 16.0);
        const double fine_error = check_dof_cloud(surface, 3.0 / 32.0);
        require(fine_error < 3.0e-2,
                surface.name + " midpoint area accuracy at N=32");
        require((coarse_error < 1.0e-12 && fine_error < 1.0e-12)
                    || fine_error < coarse_error,
                surface.name + " midpoint area improves under refinement");
    }
}

bool four_unique_valid_ids(const std::array<int, 4>& ids,
                           const SurfaceDofCloud3D& cloud)
{
    std::set<int> unique;
    for (int id : ids) {
        if (id < 0 || id >= static_cast<int>(cloud.dofs.size()))
            return false;
        unique.insert(id);
    }
    return unique.size() == 4;
}

bool contains_patch(const std::array<int, 4>& ids,
                    const SurfaceDofCloud3D& cloud,
                    int patch_id)
{
    for (int id : ids) {
        if (cloud.dofs[static_cast<std::size_t>(id)].patch_id == patch_id)
            return true;
    }
    return false;
}

bool all_on_patch(const std::array<int, 4>& ids,
                  const SurfaceDofCloud3D& cloud,
                  int patch_id)
{
    for (int id : ids) {
        if (cloud.dofs[static_cast<std::size_t>(id)].patch_id != patch_id)
            return false;
    }
    return true;
}

void test_parameter_candidates()
{
    const NativeNurbsSurface3D torus =
        make_native_nurbs_surface_3d(GeometryKind3D::Torus);
    const SurfaceDofCloud3D torus_cloud =
        make_native_surface_dofs_3d(torus, 3.0 / 32.0);
    const auto interior = parameter_dof_candidates_2x2(
        torus, torus_cloud, 0, 0.5, 0.5);
    require(four_unique_valid_ids(interior, torus_cloud),
            "interior query must have four valid DOFs");
    require(all_on_patch(interior, torus_cloud, 0),
            "interior query remains on its native patch");

    const auto& torus_tensor = torus_cloud.patches[0];
    const double torus_du = 1.0 / static_cast<double>(torus_tensor.nu);
    const auto periodic = parameter_dof_candidates_2x2(
        torus, torus_cloud, 0, 0.1 * torus_du, 0.5);
    require(four_unique_valid_ids(periodic, torus_cloud),
            "periodic query must have four valid DOFs");
    require(contains_patch(periodic, torus_cloud, 12),
            "torus closed seam reaches previous u quarter");

    const NativeNurbsSurface3D lprism =
        make_native_nurbs_surface_3d(GeometryKind3D::LPrism);
    const SurfaceDofCloud3D lcloud =
        make_native_surface_dofs_3d(lprism, 3.0 / 32.0);
    const auto& top_left = lcloud.patches[3];
    const double ldu = 1.0 / static_cast<double>(top_left.nu);
    const auto sharp = parameter_dof_candidates_2x2(
        lprism, lcloud, 3, 0.1 * ldu, 0.5);
    require(four_unique_valid_ids(sharp, lcloud),
            "sharp-edge query must retain four candidates");
    require(all_on_patch(sharp, lcloud, 3),
            "non-G1 edge remains on the source patch");

    const auto smooth = parameter_dof_candidates_2x2(
        lprism, lcloud, 3, 1.0 - 0.1 * ldu, 0.5);
    require(four_unique_valid_ids(smooth, lcloud),
            "smooth L-cap seam must retain four candidates");
    require(contains_patch(smooth, lcloud, 4),
            "smooth L-cap seam reaches its neighboring patch");

    NativeNurbsSurface3D reversed;
    reversed.name = "reversed_test";
    reversed.description = "two coplanar patches with reversed edge parameter";
    reversed.patches.push_back(
        kfbim::geometry3d::NurbsSurfacePatch3D::make_bilinear_plane(
            {0.0, 0.0, 0.0}, {1.0, 0.0, 0.0},
            {0.0, 1.0, 0.0}, {1.0, 1.0, 0.0}));
    reversed.patches.push_back(
        kfbim::geometry3d::NurbsSurfacePatch3D::make_bilinear_plane(
            {2.0, 1.0, 0.0}, {1.0, 1.0, 0.0},
            {2.0, 0.0, 0.0}, {1.0, 0.0, 0.0}));
    reversed.patch_names = {"left", "right_reversed"};
    reversed.smooth_neighbors.resize(2);
    reversed.smooth_neighbors[0][static_cast<int>(PatchEdge3D::UMax)] =
        SmoothPatchNeighbor3D{1, PatchEdge3D::UMax, true};
    reversed.smooth_neighbors[1][static_cast<int>(PatchEdge3D::UMax)] =
        SmoothPatchNeighbor3D{0, PatchEdge3D::UMax, true};
    reversed.expected_area = 2.0;
    reversed.exact_inside = [](const Eigen::Vector3d&) { return false; };
    const SurfaceDofCloud3D reversed_cloud =
        make_native_surface_dofs_3d(reversed, 0.2);
    const auto reversed_ids = parameter_dof_candidates_2x2(
        reversed, reversed_cloud, 0, 0.99, 0.2);
    require(contains_patch(reversed_ids, reversed_cloud, 1),
            "reversed smooth seam reaches its neighbor");
    bool found_high_v = false;
    for (int id : reversed_ids) {
        const auto& dof = reversed_cloud.dofs[static_cast<std::size_t>(id)];
        if (dof.patch_id == 1 && dof.v > 0.5)
            found_high_v = true;
    }
    require(found_high_v,
            "reversed smooth seam reverses the along-edge parameter");

    kfbim::geometry3d::NurbsParamTriangle3D triangle;
    triangle.patch_index = 7;
    triangle.uv = {{{0.0, 0.0}, {1.0, 0.0}, {0.0, 2.0}}};
    const Eigen::Vector2d uv = interpolate_triangle_parameter(
        triangle, Eigen::Vector3d(0.2, 0.3, 0.5));
    require((uv - Eigen::Vector2d(0.3, 1.0)).norm() < 1.0e-14,
            "triangle barycentric UV interpolation");

    require(smooth_patch_component(torus, 0).size() == 16,
            "torus smooth component contains every native patch");
}

} // namespace

int main()
{
    try {
        test_native_models();
        test_uniform_native_dofs();
        test_parameter_candidates();
        std::cout << "native NURBS model tests passed\n";
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "native NURBS model test failure: " << error.what() << '\n';
        return 1;
    }
}
