#include "industrial_nurbs_models_3d.hpp"

#include <array>
#include <cmath>
#include <string>
#include <utility>
#include <vector>

namespace kfbim::geometry3d {
namespace {

// Quadratic rational Bezier curves, used as the two boundary rows of a
// homogeneous ruled surface. Lines are degree-elevated to the same degree.
struct FlangeCurve {
    std::array<Eigen::Vector2d, 3> points;
    std::array<double, 3> weights;
};

Eigen::Vector2d polar(double radius, double angle)
{
    return {radius * std::cos(angle), radius * std::sin(angle)};
}

FlangeCurve circular_arc(const Eigen::Vector2d& center, double radius,
                         double first_angle, double last_angle)
{
    const double mid_angle = 0.5 * (first_angle + last_angle);
    const double mid_weight = std::cos(0.5 * (last_angle - first_angle));
    return {{center + polar(radius, first_angle),
             center + polar(radius / mid_weight, mid_angle),
             center + polar(radius, last_angle)},
            {1.0, mid_weight, 1.0}};
}

FlangeCurve line(const Eigen::Vector2d& first, const Eigen::Vector2d& last)
{
    return {{first, 0.5 * (first + last), last}, {1.0, 1.0, 1.0}};
}

NurbsSurfacePatch3D loft(const FlangeCurve& first, const FlangeCurve& last,
                         double first_z, double last_z)
{
    std::vector<std::vector<Eigen::Vector3d>> points(3);
    std::vector<std::vector<double>> weights(3);
    for (std::size_t i = 0; i < 3; ++i) {
        points[i] = {{first.points[i].x(), first.points[i].y(), first_z},
                     {last.points[i].x(), last.points[i].y(), last_z}};
        weights[i] = {first.weights[i], last.weights[i]};
    }
    return {kfbim::geometry::NurbsBasis1D(2, {0, 0, 0, 1, 1, 1}),
            kfbim::geometry::NurbsBasis1D(1, {0, 0, 1, 1}),
            std::move(points), std::move(weights)};
}

// Revolve a rational quadratic meridian (radius, z). The tensor weights are
// products, so both the azimuthal circles and the meridian fillet are exact.
NurbsSurfacePatch3D revolve(const FlangeCurve& meridian,
                            double first_angle, double last_angle)
{
    const auto azimuth = circular_arc(Eigen::Vector2d::Zero(), 1.0,
                                      first_angle, last_angle);
    std::vector<std::vector<Eigen::Vector3d>> points(3);
    std::vector<std::vector<double>> weights(3);
    for (std::size_t i = 0; i < 3; ++i) {
        for (std::size_t j = 0; j < 3; ++j) {
            points[i].push_back({meridian.points[j].x() * azimuth.points[i].x(),
                                 meridian.points[j].x() * azimuth.points[i].y(),
                                 meridian.points[j].y()});
            weights[i].push_back(azimuth.weights[i] * meridian.weights[j]);
        }
    }
    return {kfbim::geometry::NurbsBasis1D(2, {0, 0, 0, 1, 1, 1}),
            kfbim::geometry::NurbsBasis1D(2, {0, 0, 0, 1, 1, 1}),
            std::move(points), std::move(weights)};
}

} // namespace

IndustrialNurbsModel3D make_industrial_flange_3d()
{
    const double pi = std::acos(-1.0);
    constexpr int bolt_count = 8;
    constexpr double outer_radius = 0.90;
    constexpr double face_outer_radius = 0.87;
    constexpr double hub_root_radius = 0.43;
    constexpr double hub_fillet_radius = 0.06;
    constexpr double neck_base_radius = hub_root_radius - hub_fillet_radius;
    constexpr double neck_radius = 0.32;
    constexpr double sealing_radius = 0.305;
    constexpr double bore_radius = 0.22;
    constexpr double bore_mouth_radius = 0.235;
    constexpr double bolt_circle_radius = 0.65;
    constexpr double bolt_hole_radius = 0.075;
    constexpr double bolt_mouth_radius = 0.090;
    constexpr double bottom = -0.14;
    constexpr double top = 0.10;
    constexpr double outer_chamfer_height = 0.03;
    constexpr double hole_chamfer_height = 0.015;
    constexpr double fillet_top = top + hub_fillet_radius;
    constexpr double neck_start = 0.31;
    constexpr double neck_top = 0.35;
    constexpr double sealing_top = 0.365;
    const double half_sector = pi / bolt_count;

    std::vector<NurbsSurfacePatch3D> patches;
    std::vector<std::string> names;
    patches.reserve(256);
    names.reserve(256);
    const auto add = [&](NurbsSurfacePatch3D patch, std::string name) {
        patches.push_back(std::move(patch));
        names.push_back(std::move(name));
    };

    // Each eighth of the bolt annulus contains one bolt hole. Four strips connect
    // its exact circular arcs to the four edges of the enclosing annular sector.
    // The hole arc endpoints face the sector corners. This gives noncollapsed
    // corner connections and keeps every arc span strictly below pi.
    std::array<double, 5> hole_angles{};
    const std::array<Eigen::Vector2d, 4> reference_corners{
        polar(hub_root_radius, -half_sector), polar(face_outer_radius, -half_sector),
        polar(face_outer_radius, half_sector), polar(hub_root_radius, half_sector)};
    for (std::size_t i = 0; i < 4; ++i) {
        const Eigen::Vector2d delta = reference_corners[i]
                                   - Eigen::Vector2d(bolt_circle_radius, 0.0);
        hole_angles[i] = std::atan2(delta.y(), delta.x());
    }
    hole_angles[4] = hole_angles[0] + 2 * pi;

    for (int sector = 0; sector < bolt_count; ++sector) {
        const double angle = sector * 2 * half_sector;
        const double lo = angle - half_sector;
        const double hi = angle + half_sector;
        const Eigen::Vector2d center = polar(bolt_circle_radius, angle);
        const Eigen::Vector2d origin = Eigen::Vector2d::Zero();
        // The sector perimeter traverses counterclockwise around its interior.
        const std::array<FlangeCurve, 4> perimeter{
            line(polar(hub_root_radius, lo), polar(face_outer_radius, lo)),
            circular_arc(origin, face_outer_radius, lo, hi),
            line(polar(face_outer_radius, hi), polar(hub_root_radius, hi)),
            circular_arc(origin, hub_root_radius, hi, lo)};

        const std::string label = "sector_" + std::to_string(sector + 1);
        for (std::size_t edge = 0; edge < 4; ++edge) {
            const auto hole = circular_arc(center, bolt_hole_radius,
                angle + hole_angles[edge], angle + hole_angles[edge + 1]);
            const auto mouth = circular_arc(center, bolt_mouth_radius,
                angle + hole_angles[edge], angle + hole_angles[edge + 1]);
            const std::string strip = label + "_strip_" + std::to_string(edge + 1);
            // Curve order or height order sets du x dv to the outward normal.
            add(loft(perimeter[edge], mouth, top, top), "top_bolt_face_" + strip);
            add(loft(mouth, perimeter[edge], bottom, bottom), "bottom_bolt_face_" + strip);
            add(loft(mouth, hole, top, top - hole_chamfer_height),
                "bolt_entry_chamfer_" + strip);
            add(loft(hole, hole, top - hole_chamfer_height, bottom + hole_chamfer_height),
                "bolt_through_bore_" + strip);
            add(loft(hole, mouth, bottom + hole_chamfer_height, bottom),
                "bolt_exit_chamfer_" + strip);
        }

        const auto ring = [&](double r0, double z0, double r1, double z1,
                              const std::string& feature) {
            add(loft(circular_arc(origin, r0, lo, hi), circular_arc(origin, r1, lo, hi),
                     z0, z1), feature + "_" + label);
        };
        ring(face_outer_radius, bottom, outer_radius, bottom + outer_chamfer_height,
             "outer_lower_chamfer");
        ring(outer_radius, bottom + outer_chamfer_height,
             outer_radius, top - outer_chamfer_height, "outer_rim");
        ring(outer_radius, top - outer_chamfer_height, face_outer_radius, top,
             "outer_upper_chamfer");

        // A concave quarter-circle blends the broad flange face into its hub.
        // The raised neck, seal face and bore continue this same boundary;
        // there is no overlapping cylinder or internal cap below the hub.
        const FlangeCurve fillet{
            {{{hub_root_radius, top}, {neck_base_radius, top},
              {neck_base_radius, fillet_top}}},
            {{1.0, std::sqrt(0.5), 1.0}}};
        add(revolve(fillet, lo, hi), "hub_root_fillet_" + label);
        ring(neck_base_radius, fillet_top, neck_radius, neck_start, "tapered_hub");
        ring(neck_radius, neck_start, neck_radius, neck_top, "raised_neck");
        ring(neck_radius, neck_top, sealing_radius, sealing_top, "seal_outer_chamfer");
        ring(sealing_radius, sealing_top, bore_mouth_radius, sealing_top,
             "raised_sealing_face");
        ring(bore_mouth_radius, sealing_top, bore_radius, sealing_top - hole_chamfer_height,
             "central_bore_entry_chamfer");
        ring(bore_radius, sealing_top - hole_chamfer_height,
             bore_radius, bottom + hole_chamfer_height, "central_through_bore");
        ring(bore_radius, bottom + hole_chamfer_height, bore_mouth_radius, bottom,
             "central_bore_exit_chamfer");
        ring(bore_mouth_radius, bottom, hub_root_radius, bottom, "bottom_hub_face");
    }

    // Independent solid-of-revolution reference: integrate the outer radial
    // profile, then subtract the central bore and eight chamfered bolt bores.
    const auto frustum = [pi](double r0, double r1, double height) {
        return pi * height * (r0 * r0 + r0 * r1 + r1 * r1) / 3;
    };
    const double flange_volume = 2 * frustum(face_outer_radius, outer_radius, outer_chamfer_height)
        + pi * outer_radius * outer_radius * (top - bottom - 2 * outer_chamfer_height);
    // r(z) = a - sqrt(b^2 - (z - top - b)^2), z in [top, top + b].
    const double fillet_volume = pi * (hub_root_radius * hub_root_radius * hub_fillet_radius
        - 0.5 * pi * hub_root_radius * hub_fillet_radius * hub_fillet_radius
        + (2.0 / 3) * std::pow(hub_fillet_radius, 3));
    const double hub_volume = fillet_volume
        + frustum(neck_base_radius, neck_radius, neck_start - fillet_top)
        + pi * neck_radius * neck_radius * (neck_top - neck_start)
        + frustum(neck_radius, sealing_radius, sealing_top - neck_top);
    const double central_bore_volume = 2 * frustum(bore_radius, bore_mouth_radius, hole_chamfer_height)
        + pi * bore_radius * bore_radius * (sealing_top - bottom - 2 * hole_chamfer_height);
    const double bolt_bore_volume = 2 * frustum(bolt_hole_radius, bolt_mouth_radius, hole_chamfer_height)
        + pi * bolt_hole_radius * bolt_hole_radius * (top - bottom - 2 * hole_chamfer_height);
    const double volume = flange_volume + hub_volume - central_bore_volume - bolt_count * bolt_bore_volume;
    return finalize_industrial_model(
        "flange",
        "Eight-bolt pipe flange with a flared raised hub, exact toroidal root fillet, "
        "raised annular sealing face, chamfered central bore and bolt holes, and beveled outer rim; "
        "one connected untrimmed native tensor NURBS boundary.",
        std::move(patches), std::move(names), volume, bolt_count + 1);
}

} // namespace kfbim::geometry3d
