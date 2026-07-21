#include "native_nurbs_surface_3d.hpp"

#include <cmath>
#include <iostream>
#include <stdexcept>
#include <string>

namespace {

using kfbim::app3d::GeometryKind3D;
using kfbim::app3d::NativeNurbsSurface3D;
using kfbim::app3d::make_native_nurbs_surface_3d;

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

} // namespace

int main()
{
    try {
        test_native_models();
        std::cout << "native NURBS model tests passed\n";
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "native NURBS model test failure: " << error.what() << '\n';
        return 1;
    }
}
