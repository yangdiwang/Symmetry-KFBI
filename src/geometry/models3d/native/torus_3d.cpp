#include "construction.hpp"
#include "models_3d.hpp"

#include <cmath>
#include <stdexcept>
#include <utility>

namespace kfbim::app3d {

namespace {

geometry3d::NurbsSurfacePatch3D make_torus_quarter_patch(
    int u_quarter,
    int v_quarter,
    double major_radius,
    double minor_radius,
    const Eigen::Vector3d& center)
{
    using namespace native_detail;
    const RationalQuarterArc2D u_arc =
        quarter_arc(0.5 * kPi * u_quarter);
    const RationalQuarterArc2D v_arc =
        quarter_arc(0.5 * kPi * v_quarter);

    std::vector<std::vector<Eigen::Vector3d>> controls(
        3, std::vector<Eigen::Vector3d>(3));
    std::vector<std::vector<double>> weights(3, std::vector<double>(3));
    for (int i = 0; i < 3; ++i) {
        for (int j = 0; j < 3; ++j) {
            const double rho = major_radius
                             + minor_radius * v_arc.controls[j].x();
            controls[i][j] = {
                center.x() + rho * u_arc.controls[i].x(),
                center.y() + rho * u_arc.controls[i].y(),
                center.z() + minor_radius * v_arc.controls[j].y()};
            weights[i][j] = u_arc.weights[i] * v_arc.weights[j];
        }
    }
    return geometry3d::NurbsSurfacePatch3D(
        quadratic_unit_basis(), quadratic_unit_basis(),
        std::move(controls), std::move(weights));
}

NativeNurbsSurface3D make_torus(double major_radius,
                                double minor_radius,
                                const Eigen::Vector3d& center)
{
    using namespace native_detail;
    NativeNurbsSurface3D surface;
    surface.name = "torus";
    surface.description = "exact rational-quadratic NURBS torus";
    for (int iu = 0; iu < 4; ++iu) {
        for (int iv = 0; iv < 4; ++iv) {
            append_patch(surface,
                         "torus_u" + std::to_string(iu)
                             + "_v" + std::to_string(iv),
                         make_torus_quarter_patch(
                             iu, iv, major_radius, minor_radius, center));
        }
    }
    auto patch_id = [](int iu, int iv) {
        return 4 * (iu % 4) + (iv % 4);
    };
    for (int iu = 0; iu < 4; ++iu) {
        for (int iv = 0; iv < 4; ++iv) {
            connect_smooth(surface,
                           patch_id(iu, iv), PatchEdge3D::UMax,
                           patch_id(iu + 1, iv), PatchEdge3D::UMin,
                           false);
            connect_smooth(surface,
                           patch_id(iu, iv), PatchEdge3D::VMax,
                           patch_id(iu, iv + 1), PatchEdge3D::VMin,
                           false);
        }
    }
    surface.expected_area = 4.0 * kPi * kPi * major_radius * minor_radius;
    surface.exact_inside =
        [major_radius, minor_radius, center](const Eigen::Vector3d& x) {
            const Eigen::Vector3d local = x - center;
            const double rho = std::hypot(local.x(), local.y());
            return (rho - major_radius) * (rho - major_radius)
                 + local.z() * local.z() < minor_radius * minor_radius;
        };
    return surface;
}

} // namespace

namespace native_detail {

NativeNurbsSurface3D make_default_torus_3d()
{
    return make_torus(0.55, 0.20, Eigen::Vector3d(0.07, -0.04, 0.03));
}

} // namespace native_detail

NativeNurbsSurface3D make_native_nurbs_torus_3d(
    double major_radius, double minor_radius, const Eigen::Vector3d& center)
{
    if (!std::isfinite(major_radius) || !std::isfinite(minor_radius)
        || !(major_radius > minor_radius) || !(minor_radius > 0.0)
        || !center.allFinite())
        throw std::invalid_argument("torus requires finite R > r > 0 and center");
    return make_torus(major_radius, minor_radius, center);
}

} // namespace kfbim::app3d
