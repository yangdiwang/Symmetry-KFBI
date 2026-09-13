#include "src/geometry/models3d/industrial/models_3d.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <stdexcept>
#include <utility>

namespace kfbim::geometry3d {
namespace {

using Spine = std::array<Eigen::Vector3d, 4>;

Eigen::Vector3d polar_point(double radius, double angle, double z = 0.0)
{
    return {radius * std::cos(angle), radius * std::sin(angle), z};
}

Spine straight_spine(const Eigen::Vector3d& first, const Eigen::Vector3d& last)
{
    return {{first, (2.0 * first + last) / 3.0,
             (first + 2.0 * last) / 3.0, last}};
}

Spine at_height(Spine points, double z)
{
    for (auto& point : points) point.z() = z;
    return points;
}

NurbsSurfacePatch3D swept_sector(
    const Spine& spine, double start, double end, bool reverse = false)
{
    // Complex multiplication R(u) A(v) combines a polynomial cubic spine
    // with an exact rational quadratic unit-circle arc. Each radial station
    // is circular and each flank is a swept cubic curve. These are native
    // tensor-product NURBS control nets, not sampled surface fits.
    const double middle = 0.5 * (start + end);
    const double middle_weight = std::cos(0.5 * (end - start));
    const std::array<Eigen::Vector3d, 3> arc{{
        polar_point(1.0, start), polar_point(1.0 / middle_weight, middle),
        polar_point(1.0, end)}};
    const std::array<double, 3> arc_weights{{1.0, middle_weight, 1.0}};
    std::vector<std::vector<Eigen::Vector3d>> net(4, std::vector<Eigen::Vector3d>(3));
    std::vector<std::vector<double>> weights(4, std::vector<double>(3));
    for (int i = 0; i < 4; ++i) for (int j = 0; j < 3; ++j) {
        net[i][j] = {
            spine[i].x() * arc[j].x() - spine[i].y() * arc[j].y(),
            spine[i].x() * arc[j].y() + spine[i].y() * arc[j].x(),
            spine[i].z()};
        weights[i][j] = arc_weights[j];
    }
    if (reverse) {
        std::reverse(net.begin(), net.end());
        std::reverse(weights.begin(), weights.end());
    }
    return NurbsSurfacePatch3D(
        geometry::NurbsBasis1D(3, {0, 0, 0, 0, 1, 1, 1, 1}),
        geometry::NurbsBasis1D(2, {0, 0, 0, 1, 1, 1}),
        std::move(net), std::move(weights));
}

NurbsSurfacePatch3D vane_flank(const Spine& top, double angle, bool reverse)
{
    std::vector<std::vector<Eigen::Vector3d>> net(4, std::vector<Eigen::Vector3d>(2));
    const double c = std::cos(angle), s = std::sin(angle);
    for (int i = 0; i < 4; ++i) {
        net[i][1] = {c * top[i].x() - s * top[i].y(),
                     s * top[i].x() + c * top[i].y(), top[i].z()};
        net[i][0] = net[i][1];
        net[i][0].z() = 0.0;
    }
    if (reverse) std::reverse(net.begin(), net.end());
    return NurbsSurfacePatch3D(
        geometry::NurbsBasis1D(3, {0, 0, 0, 0, 1, 1, 1, 1}),
        geometry::NurbsBasis1D(1, {0, 0, 1, 1}), std::move(net),
        std::vector<std::vector<double>>(4, std::vector<double>(2, 1.0)));
}

Spine power_coefficients(const Spine& points)
{
    return {{points[0], 3.0 * (points[1] - points[0]),
             3.0 * (points[2] - 2.0 * points[1] + points[0]),
             points[3] - 3.0 * points[2] + 3.0 * points[1] - points[0]}};
}

double vane_volume(const Spine& top, double angular_width)
{
    // In physical polar coordinates, dA = (R dot R') du dtheta.
    // Integrate h(u) (R dot R') exactly from polynomial coefficients.
    // This reference is independent of surface quadrature or tessellation.
    const auto p = power_coefficients(top);
    double integral = 0.0;
    for (int i = 0; i < 4; ++i) for (int j = 1; j < 4; ++j)
        for (int k = 0; k < 4; ++k)
            integral += p[k].z() * static_cast<double>(j)
                * p[i].head<2>().dot(p[j].head<2>())
                / static_cast<double>(i + j + k);
    return angular_width * integral;
}

double linear_annular_volume(double r0, double z0, double r1, double z1)
{
    // 2 pi integral r z(r) dr for a linear height profile.
    return 2.0 * std::acos(-1.0) * (r1 - r0)
        * (r0 * (2.0 * z0 + z1) + r1 * (z0 + 2.0 * z1)) / 6.0;
}

void verify_spine_is_radially_monotone(const Spine& spine)
{
    // The degree-five Bernstein coefficients of R dot R' are positive.
    // This certifies a positive planar Jacobian on the WHOLE parameter
    // interval and a unique radius per u. The angular sectors partition
    // a full circle, so the vanes and channels cannot cross or overlap.
    constexpr std::array<int, 4> choose3{{1, 3, 3, 1}};
    constexpr std::array<int, 3> choose2{{1, 2, 1}};
    constexpr std::array<int, 6> choose5{{1, 5, 10, 10, 5, 1}};
    std::array<double, 6> coefficients{};
    for (int i = 0; i < 4; ++i) for (int j = 0; j < 3; ++j)
        coefficients[i + j] += static_cast<double>(choose3[i] * choose2[j])
            / choose5[i + j] * spine[i].head<2>().dot(
                3.0 * (spine[j + 1] - spine[j]).head<2>());
    for (double coefficient : coefficients)
        if (!(coefficient > 0.0))
            throw std::logic_error("impeller spine lacks a positive radial Jacobian certificate");
}

} // namespace

IndustrialNurbsModel3D make_industrial_impeller_3d()
{
    constexpr int blade_count = 8;
    constexpr double bore_radius = 0.13;
    constexpr double chamfer_radius = 0.155;
    constexpr double hub_shoulder_radius = 0.25;
    constexpr double hub_radius = 0.30;
    constexpr double tip_radius = 0.88;
    constexpr double disk_face_radius = 0.94;
    constexpr double disk_radius = 0.97;
    constexpr double base_bottom = -0.16;
    constexpr double hub_top = 0.45;
    constexpr double bore_top = 0.425;
    constexpr double blade_root_top = 0.39;
    constexpr double blade_tip_top = 0.18;
    const double pi = std::acos(-1.0);
    const double pitch = 2.0 * pi / blade_count;
    const double blade_width = 6.0 * pi / 180.0;
    const double sweep = -40.0 * pi / 180.0;

    const Spine blade_top{{
        {hub_radius, 0.0, blade_root_top}, {0.51, -0.005, 0.39},
        {0.79, -0.18, 0.28}, polar_point(tip_radius, sweep, blade_tip_top)}};
    verify_spine_is_radially_monotone(blade_top);
    const Spine bottom = straight_spine(
        polar_point(bore_radius, 0.0, base_bottom),
        polar_point(disk_face_radius, sweep, base_bottom));
    verify_spine_is_radially_monotone(bottom);

    std::vector<NurbsSurfacePatch3D> patches;
    std::vector<std::string> names;
    patches.reserve(192);
    names.reserve(192);
    auto add = [&](NurbsSurfacePatch3D patch, const std::string& name) {
        patches.push_back(std::move(patch));
        names.push_back(name);
    };
    auto radial_strip = [&](double r0, double z0, double r1, double z1,
                            double a, double b, const std::string& name) {
        add(swept_sector(straight_spine({r0, 0.0, z0}, {r1, 0.0, z1}), a, b), name);
    };

    // Two angular cells per vane: its occupied footprint, then an open
    // flow channel. Shared boundaries have identical circular/polynomial
    // parameterizations, including the disk underside. Emit ONLY exposed
    // faces of one connected solid: omit all hub/vane/disk interfaces.
    for (int blade = 0; blade < blade_count; ++blade) {
        const double start = blade * pitch;
        const std::array<double, 3> angles{{start, start + blade_width, start + pitch}};
        for (int cell = 0; cell < 2; ++cell) {
            const double a = angles[cell], b = angles[cell + 1];
            const std::string suffix = "_" + std::to_string(blade + 1)
                + (cell == 0 ? "_vane" : "_channel");
            radial_strip(bore_radius, bore_top, chamfer_radius, hub_top,
                         a, b, "hub_bore_chamfer" + suffix);
            radial_strip(chamfer_radius, hub_top, hub_shoulder_radius, hub_top,
                         a, b, "hub_crown" + suffix);
            radial_strip(hub_shoulder_radius, hub_top, hub_radius, blade_root_top,
                         a, b, "hub_conical_shoulder" + suffix);
            radial_strip(bore_radius, base_bottom, bore_radius, bore_top,
                         a, b, "through_bore" + suffix);
            add(swept_sector(bottom, a, b, true), "disk_bottom" + suffix);
            radial_strip(tip_radius, 0.0, disk_face_radius, 0.0,
                         a + sweep, b + sweep, "disk_outer_land" + suffix);
            radial_strip(disk_face_radius, 0.0, disk_radius, -0.03,
                         a + sweep, b + sweep, "disk_top_chamfer" + suffix);
            radial_strip(disk_radius, -0.03, disk_radius, -0.13,
                         a + sweep, b + sweep, "disk_outer_rim" + suffix);
            radial_strip(disk_radius, -0.13, disk_face_radius, base_bottom,
                         a + sweep, b + sweep, "disk_bottom_chamfer" + suffix);
            if (cell == 0) {
                add(swept_sector(blade_top, a, b), "curved_vane_crown" + suffix);
                add(vane_flank(blade_top, a, false), "curved_vane_pressure_side" + suffix);
                add(vane_flank(blade_top, b, true), "curved_vane_suction_side" + suffix);
                radial_strip(tip_radius, blade_tip_top, tip_radius, 0.0,
                             a + sweep, b + sweep, "vane_discharge_tip" + suffix);
            } else {
                add(swept_sector(at_height(blade_top, 0.0), a, b),
                    "open_flow_channel_floor" + suffix);
                radial_strip(hub_radius, blade_root_top, hub_radius, 0.0,
                             a, b, "exposed_hub_wall" + suffix);
            }
        }
    }

    const double base_volume = pi * (
        2.0 * 0.03 / 3.0 * (disk_face_radius * disk_face_radius
            + disk_face_radius * disk_radius + disk_radius * disk_radius)
        + 0.10 * disk_radius * disk_radius
        - (-base_bottom) * bore_radius * bore_radius);
    const double hub_volume =
        linear_annular_volume(bore_radius, bore_top, chamfer_radius, hub_top)
        + linear_annular_volume(chamfer_radius, hub_top, hub_shoulder_radius, hub_top)
        + linear_annular_volume(hub_shoulder_radius, hub_top, hub_radius, blade_root_top);
    const double volume = base_volume + hub_volume
        + blade_count * vane_volume(blade_top, blade_width);
    return finalize_industrial_model(
        "impeller",
        "Open radial impeller with eight upright backwards-curved finite-thickness "
        "vanes, decreasing vane height, a raised chamfered hub and circular through "
        "bore, and a chamfered circular base disk. One connected exposed boundary "
        "of exact native rational NURBS patches; channels remain open above the "
        "disk. This is an illustrative mechanical geometry, not an aerodynamic design.",
        std::move(patches), std::move(names), volume, 1);
}

} // namespace kfbim::geometry3d
