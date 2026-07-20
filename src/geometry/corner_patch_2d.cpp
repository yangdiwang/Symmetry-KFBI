#include "corner_patch_2d.hpp"

#include "p2_curve_2d.hpp"
#include <algorithm>
#include <cmath>
#include <limits>
#include <sstream>
#include <stdexcept>

namespace kfbim {

namespace {

constexpr double kPi = 3.14159265358979323846;
constexpr double kTwoPi = 2.0 * kPi;

Eigen::Vector2d normalize_or_throw(const Eigen::Vector2d& v, const char* context)
{
    const double n = v.norm();
    if (n <= 1.0e-14)
        throw std::invalid_argument(context);
    return v / n;
}

double cross2(const Eigen::Vector2d& a, const Eigen::Vector2d& b)
{
    return a[0] * b[1] - a[1] * b[0];
}

double dot2(const Eigen::Vector2d& a, const Eigen::Vector2d& b)
{
    return a.dot(b);
}

double ccw_angle(const Eigen::Vector2d& from, const Eigen::Vector2d& to)
{
    double angle = std::atan2(cross2(from, to), dot2(from, to));
    if (angle < 0.0)
        angle += kTwoPi;
    return angle;
}

double normalize_opening(double angle)
{
    while (angle <= 0.0)
        angle += kTwoPi;
    while (angle >= kTwoPi)
        angle -= kTwoPi;
    return angle;
}

double corner_clearance_radius(const Interface2D& iface, double tol)
{
    if (iface.corners().empty())
        return 0.0;
    if (iface.corners().size() < 2) {
        throw std::invalid_argument(
            "corner patch radius requires at least two corners");
    }

    double min_corner_distance = std::numeric_limits<double>::infinity();
    for (int i = 0; i < static_cast<int>(iface.corners().size()); ++i) {
        const int pi = iface.corner(i).point;
        const Eigen::Vector2d xi = iface.points().row(pi).transpose();
        for (int j = i + 1; j < static_cast<int>(iface.corners().size()); ++j) {
            const int pj = iface.corner(j).point;
            const Eigen::Vector2d xj = iface.points().row(pj).transpose();
            min_corner_distance =
                std::min(min_corner_distance, (xi - xj).norm());
        }
    }

    const double radius = 0.5 * min_corner_distance - tol;
    if (!(radius > 0.0)) {
        throw std::invalid_argument(
            "corner patch automatic radius must be positive");
    }
    return radius;
}

std::string patch_context(const CornerPatch2D& patch)
{
    std::ostringstream out;
    out << "corner patch corner=" << patch.corner
        << " side=" << corner_patch_side_name(patch.side)
        << " radius=" << patch.radius;
    return out.str();
}

void validate_corner_clearance(const Interface2D& iface,
                               const CornerPatch2D& patch,
                               double tol)
{
    for (int c = 0; c < static_cast<int>(iface.corners().size()); ++c) {
        if (c == patch.corner)
            continue;
        const int point = iface.corner(c).point;
        const Eigen::Vector2d other = iface.points().row(point).transpose();
        if ((other - patch.center).norm() < patch.radius + tol) {
            throw std::invalid_argument(
                patch_context(patch) + " contains another corner " + std::to_string(c));
        }
    }
}

void validate_patch_overlap(const std::vector<CornerPatch2D>& patches,
                            double tol)
{
    for (int i = 0; i < static_cast<int>(patches.size()); ++i) {
        for (int j = i + 1; j < static_cast<int>(patches.size()); ++j) {
            if (patches[i].side != patches[j].side)
                continue;
            const double dist = (patches[i].center - patches[j].center).norm();
            if (dist < patches[i].radius + patches[j].radius + tol) {
                throw std::invalid_argument(
                    patch_context(patches[i]) + " overlaps corner patch "
                    + std::to_string(j));
            }
        }
    }
}

CornerPatch2D make_patch(const Interface2D& iface,
                         int corner_id,
                         CornerPatchSide2D side,
                         double opening,
                         double radius)
{
    const CornerData2D& corner = iface.corner(corner_id);
    const Eigen::Vector2d t_minus =
        normalize_or_throw(corner.tangent_minus,
                           "corner tangent_minus must be nonzero");
    const Eigen::Vector2d t_plus =
        normalize_or_throw(corner.tangent_plus,
                           "corner tangent_plus must be nonzero");
    const Eigen::Vector2d back_prev = -t_minus;
    const Eigen::Vector2d fwd_next = t_plus;

    CornerPatch2D patch;
    patch.corner = corner_id;
    patch.point = corner.point;
    patch.side = side;
    patch.opening_angle = opening;
    patch.radius = radius;
    patch.center = iface.points().row(corner.point).transpose();

    if (side == CornerPatchSide2D::Interior) {
        patch.ray_minus = fwd_next;
        patch.ray_plus = back_prev;
    } else {
        patch.ray_minus = back_prev;
        patch.ray_plus = fwd_next;
    }

    return patch;
}

CornerPatch2D make_circle_patch(const Interface2D& iface,
                                int corner_id,
                                double radius)
{
    const CornerData2D& corner = iface.corner(corner_id);
    CornerPatch2D patch;
    patch.corner = corner_id;
    patch.point = corner.point;
    patch.side = CornerPatchSide2D::Interior;
    patch.shape = CornerPatchShape2D::Circle;
    patch.opening_angle = kTwoPi;
    patch.radius = radius;
    patch.center = iface.points().row(corner.point).transpose();
    patch.ray_minus = Eigen::Vector2d(1.0, 0.0);
    patch.ray_plus = Eigen::Vector2d(1.0, 0.0);
    return patch;
}

} // namespace

const char* corner_patch_side_name(CornerPatchSide2D side)
{
    switch (side) {
    case CornerPatchSide2D::Interior:
        return "interior";
    case CornerPatchSide2D::Exterior:
        return "exterior";
    }
    return "unknown";
}

bool corner_patch_contains_point_2d(const CornerPatch2D& patch,
                                    const Eigen::Vector2d& point,
                                    double tolerance)
{
    const Eigen::Vector2d v = point - patch.center;
    const double r = v.norm();
    if (r > patch.radius || r <= tolerance)
        return false;

    if (patch.shape == CornerPatchShape2D::Circle)
        return true;

    return corner_patch_sector_contains_point_2d(patch, point, tolerance);
}

bool corner_patch_sector_contains_point_2d(const CornerPatch2D& patch,
                                           const Eigen::Vector2d& point,
                                           double tolerance)
{
    if (patch.shape == CornerPatchShape2D::Circle) {
        return (point - patch.center).norm() > tolerance;
    }

    const Eigen::Vector2d v = point - patch.center;
    const double r = v.norm();
    if (r <= tolerance)
        return false;

    const Eigen::Vector2d dir = v / r;
    const Eigen::Vector2d ray0 =
        normalize_or_throw(patch.ray_minus, "corner patch ray_minus must be nonzero");
    const Eigen::Vector2d ray1 =
        normalize_or_throw(patch.ray_plus, "corner patch ray_plus must be nonzero");
    const double opening = patch.opening_angle > 0.0
        ? patch.opening_angle
        : ccw_angle(ray0, ray1);
    const double angle = ccw_angle(ray0, dir);
    return angle <= opening + tolerance;
}

double corner_patch_radial_signed_distance_2d(const CornerPatch2D& patch,
                                              const Eigen::Vector2d& point)
{
    return (point - patch.center).norm() - patch.radius;
}

std::vector<CornerPatch2D> build_corner_patches_2d(
    const Interface2D& iface,
    double radius,
    CornerPatchBuildMode2D build_mode,
    const CornerPatchOptions2D& options)
{
    if (!(radius > 0.0))
        throw std::invalid_argument("corner patch radius must be positive");

    const double clearance_radius =
        corner_clearance_radius(iface, options.geometry_tol);
    if (clearance_radius == 0.0)
        return {};
    radius = std::min(radius, clearance_radius);

    std::vector<CornerPatch2D> patches;
    for (int c = 0; c < static_cast<int>(iface.corners().size()); ++c) {
        const CornerData2D& corner = iface.corner(c);
        const double interior_opening = normalize_opening(kPi - corner.turn_angle);
        const double exterior_opening = kTwoPi - interior_opening;

        if (build_mode == CornerPatchBuildMode2D::Circle
            || (build_mode
                    == CornerPatchBuildMode2D::
                           ReentrantCircleDominantOpening
                && interior_opening > kPi)) {
            patches.push_back(make_circle_patch(iface, c, radius));
        } else if (build_mode == CornerPatchBuildMode2D::BothSides) {
            patches.push_back(make_patch(iface,
                                         c,
                                         CornerPatchSide2D::Interior,
                                         interior_opening,
                                         radius));
            patches.push_back(make_patch(iface,
                                         c,
                                         CornerPatchSide2D::Exterior,
                                         exterior_opening,
                                         radius));
        } else if (interior_opening >= exterior_opening) {
            patches.push_back(make_patch(iface,
                                         c,
                                         CornerPatchSide2D::Interior,
                                         interior_opening,
                                         radius));
        } else {
            patches.push_back(make_patch(iface,
                                         c,
                                         CornerPatchSide2D::Exterior,
                                         exterior_opening,
                                         radius));
        }
    }

    for (const CornerPatch2D& patch : patches) {
        validate_corner_clearance(iface, patch, options.geometry_tol);
    }
    if (options.require_no_overlap)
        validate_patch_overlap(patches, options.geometry_tol);

    return patches;
}

Interface2D with_corner_patches_2d(const Interface2D& iface,
                                   std::vector<CornerPatch2D> patches)
{
    PanelSideGeometry2D panel_sides;
    panel_sides.point_normals = iface.panel_normals();
    panel_sides.point_tangents = iface.panel_tangents();

    return Interface2D(iface.points(),
                       iface.normals(),
                       iface.weights(),
                       iface.points_per_panel(),
                       iface.panel_point_indices(),
                       iface.panel_components(),
                       std::move(panel_sides),
                       iface.point_kind(),
                       iface.corner_index_by_point(),
                       iface.corners(),
                       iface.panel_node_layout(),
                       std::move(patches));
}

Interface2D apply_corner_patch_config_2d(const Interface2D& iface,
                                         const CornerPatchConfig2D& config)
{
    if (!config.enabled)
        return iface;

    std::vector<CornerPatch2D> patches =
        build_corner_patches_2d(
            iface, config.radius, config.build_mode, config.options);
    if (!config.radius_overrides_by_corner.empty()) {
        for (CornerPatch2D& patch : patches) {
            if (patch.corner < 0
                || patch.corner
                       >= static_cast<int>(
                           config.radius_overrides_by_corner.size())) {
                continue;
            }
            const double override_radius =
                config.radius_overrides_by_corner[
                    static_cast<std::size_t>(patch.corner)];
            if (std::isfinite(override_radius) && override_radius > 0.0)
                patch.radius = override_radius;
        }
        for (const CornerPatch2D& patch : patches) {
            validate_corner_clearance(
                iface, patch, config.options.geometry_tol);
        }
        if (config.options.require_no_overlap)
            validate_patch_overlap(patches, config.options.geometry_tol);
    }
    return with_corner_patches_2d(iface, std::move(patches));
}

} // namespace kfbim
