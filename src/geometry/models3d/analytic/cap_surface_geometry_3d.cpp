#include "cap_surface_geometry_3d.hpp"

#include <Eigen/Geometry>

#include <algorithm>
#include <cmath>
#include <stdexcept>
#include <utility>

namespace kfbim::geometry3d {
namespace {

constexpr double pi = 3.141592653589793238462643383279502884;
constexpr double rigid_transform_tolerance = 1.0e-12;

enum class AtlasPatchKind { PolarCentral, PolarRing, MeridianBelt };
enum class AtlasSide { None, North, South, East, West };

struct AtlasPatch {
    AtlasPatchKind kind = AtlasPatchKind::PolarCentral;
    AtlasSide side = AtlasSide::None;
    int hemisphere = 0;
};

struct ReferenceEvaluation {
    Eigen::Vector3d point = Eigen::Vector3d::Zero();
    Eigen::Vector3d du = Eigen::Vector3d::Zero();
    Eigen::Vector3d dv = Eigen::Vector3d::Zero();
};

AtlasPatch atlas_patch(int patch)
{
    if (patch < 0 || patch >= 14)
        throw std::out_of_range("Patch index is out of range");
    if (patch == 0)
        return {AtlasPatchKind::PolarCentral, AtlasSide::None, +1};
    if (patch >= 1 && patch <= 4) {
        constexpr AtlasSide sides[] = {
            AtlasSide::North, AtlasSide::South,
            AtlasSide::East, AtlasSide::West};
        return {AtlasPatchKind::PolarRing, sides[patch - 1], +1};
    }
    if (patch >= 5 && patch <= 8) {
        constexpr AtlasSide sides[] = {
            AtlasSide::North, AtlasSide::South,
            AtlasSide::East, AtlasSide::West};
        return {AtlasPatchKind::MeridianBelt, sides[patch - 5], 0};
    }
    if (patch == 9)
        return {AtlasPatchKind::PolarCentral, AtlasSide::None, -1};
    constexpr AtlasSide sides[] = {
        AtlasSide::North, AtlasSide::South,
        AtlasSide::East, AtlasSide::West};
    return {AtlasPatchKind::PolarRing, sides[patch - 10], -1};
}

int atlas_patch_index(AtlasPatchKind kind, AtlasSide side, int hemisphere)
{
    auto side_offset = [](AtlasSide value) {
        switch (value) {
        case AtlasSide::North: return 0;
        case AtlasSide::South: return 1;
        case AtlasSide::East: return 2;
        case AtlasSide::West: return 3;
        case AtlasSide::None: break;
        }
        return -1;
    };
    if (kind == AtlasPatchKind::PolarCentral)
        return hemisphere > 0 ? 0 : 9;
    const int offset = side_offset(side);
    if (offset < 0)
        return -1;
    if (kind == AtlasPatchKind::MeridianBelt)
        return 5 + offset;
    return (hemisphere > 0 ? 1 : 10) + offset;
}

Eigen::Vector2d square_side_point(AtlasSide side,
                                  double v,
                                  double half_width)
{
    const double s = 2.0 * v - 1.0;
    switch (side) {
    case AtlasSide::North: return {half_width * s, half_width};
    case AtlasSide::East: return {half_width, -half_width * s};
    case AtlasSide::South: return {-half_width * s, -half_width};
    case AtlasSide::West: return {-half_width, half_width * s};
    case AtlasSide::None: break;
    }
    throw std::invalid_argument("A square side is required");
}

Eigen::Vector2d square_side_derivative(AtlasSide side, double half_width)
{
    switch (side) {
    case AtlasSide::North: return {2.0 * half_width, 0.0};
    case AtlasSide::East: return {0.0, -2.0 * half_width};
    case AtlasSide::South: return {-2.0 * half_width, 0.0};
    case AtlasSide::West: return {0.0, 2.0 * half_width};
    case AtlasSide::None: break;
    }
    throw std::invalid_argument("A square side is required");
}

ReferenceEvaluation reference_geometry(
    const AnalyticCapGeometryOptions3D& options,
    int patch_index,
    double u,
    double v)
{
    const AtlasPatch patch = atlas_patch(patch_index);
    if (!std::isfinite(u) || !std::isfinite(v) || u < -1.0e-12
        || u > 1.0 + 1.0e-12 || v < -1.0e-12 || v > 1.0 + 1.0e-12)
        throw std::out_of_range("Patch coordinates lie outside [0,1]^2");
    u = std::clamp(u, 0.0, 1.0);
    v = std::clamp(v, 0.0, 1.0);

    ReferenceEvaluation result;
    if (patch.kind == AtlasPatchKind::PolarCentral) {
        const double x = options.square_half_width * (2.0 * u - 1.0);
        const double y = options.square_half_width * (2.0 * v - 1.0);
        const double z_abs = std::sqrt(std::max(0.0, 1.0 - x*x - y*y));
        if (!(z_abs > 0.0))
            throw std::runtime_error("Degenerate central cap Jacobian");
        const double sign = static_cast<double>(patch.hemisphere);
        const double xu = 2.0 * options.square_half_width;
        const double yv = 2.0 * options.square_half_width;
        result.point = {x, y, sign * z_abs};
        result.du = {xu, 0.0, -sign * x * xu / z_abs};
        result.dv = {0.0, yv, -sign * y * yv / z_abs};
        return result;
    }

    const Eigen::Vector2d q0 =
        square_side_point(patch.side, v, options.square_half_width);
    const Eigen::Vector2d dq0 =
        square_side_derivative(patch.side, options.square_half_width);
    const double r0 = q0.norm();
    const Eigen::Vector2d direction = q0 / r0;
    const Eigen::Vector2d direction_v =
        dq0 / r0 - q0 * (q0.dot(dq0) / (r0*r0*r0));

    if (patch.kind == AtlasPatchKind::PolarRing) {
        const Eigen::Vector2d q1 = options.polar_radius * direction;
        const Eigen::Vector2d q = (1.0 - u) * q0 + u * q1;
        const Eigen::Vector2d qu = q1 - q0;
        const Eigen::Vector2d qv =
            (1.0 - u) * dq0 + u * options.polar_radius * direction_v;
        const double z_abs =
            std::sqrt(std::max(0.0, 1.0 - q.squaredNorm()));
        if (!(z_abs > 0.0))
            throw std::runtime_error("Degenerate polar ring Jacobian");
        const double sign = static_cast<double>(patch.hemisphere);
        result.point = {q.x(), q.y(), sign * z_abs};
        result.du = {qu.x(), qu.y(), -sign * q.dot(qu) / z_abs};
        result.dv = {qv.x(), qv.y(), -sign * q.dot(qv) / z_abs};
        return result;
    }

    const double theta_c = std::asin(options.polar_radius);
    const double theta_span = pi - 2.0 * theta_c;
    const double theta = theta_c + u * theta_span;
    const double st = std::sin(theta);
    const double ct = std::cos(theta);
    result.point = {st * direction.x(), st * direction.y(), ct};
    result.du = {theta_span * ct * direction.x(),
                 theta_span * ct * direction.y(), -theta_span * st};
    result.dv = {st * direction_v.x(), st * direction_v.y(), 0.0};
    return result;
}

} // namespace

AnalyticCapGeometry3D::AnalyticCapGeometry3D(
    AnalyticCapGeometryOptions3D options)
    : options_(std::move(options))
{
    if (!(options_.square_half_width > 0.0)
        || std::sqrt(2.0) * options_.square_half_width >= 1.0)
        throw std::invalid_argument(
            "square_half_width must make the central square lie inside the unit disk");
    if (!(options_.polar_radius
              > std::sqrt(2.0) * options_.square_half_width)
        || !(options_.polar_radius < 1.0))
        throw std::invalid_argument(
            "polar_radius must enclose the square corners and be less than one");
    if ((options_.ellipsoid_axes.array() <= 0.0).any()
        || !options_.ellipsoid_axes.allFinite())
        throw std::invalid_argument("Ellipsoid axes must be finite and positive");
    if (!std::isfinite(options_.flower_epsilon)
        || !std::isfinite(options_.flower_eta)
        || 1.0 - std::abs(options_.flower_epsilon)
                   - 2.0 * std::abs(options_.flower_eta) <= 0.0)
        throw std::invalid_argument(
            "Flower parameters do not guarantee a positive radial graph");
    if (!options_.rigid_rotation.allFinite()
        || !options_.rigid_center.allFinite()
        || !options_.rigid_translation.allFinite())
        throw std::invalid_argument(
            "Rigid rotation, center, and translation must be finite");
    const Eigen::Matrix3d orthogonality_error =
        options_.rigid_rotation.transpose() * options_.rigid_rotation
        - Eigen::Matrix3d::Identity();
    if (orthogonality_error.cwiseAbs().maxCoeff()
        > rigid_transform_tolerance)
        throw std::invalid_argument("Rigid rotation must be orthogonal");
    if (std::abs(options_.rigid_rotation.determinant() - 1.0)
        > rigid_transform_tolerance)
        throw std::invalid_argument(
            "Rigid rotation must have determinant one");
}

bool AnalyticCapGeometry3D::has_nontrivial_rigid_map() const noexcept
{
    return !options_.rigid_rotation.isIdentity(0.0)
        || !options_.rigid_translation.isZero(0.0);
}

Eigen::Vector3d AnalyticCapGeometry3D::forward_rigid_point(
    const Eigen::Vector3d& reference) const
{
    if (!has_nontrivial_rigid_map())
        return reference;
    return options_.rigid_center
         + options_.rigid_rotation * (reference - options_.rigid_center)
         + options_.rigid_translation;
}

Eigen::Vector3d AnalyticCapGeometry3D::inverse_rigid_point(
    const Eigen::Vector3d& world) const
{
    if (!has_nontrivial_rigid_map())
        return world;
    return options_.rigid_center
         + options_.rigid_rotation.transpose()
               * (world - options_.rigid_center - options_.rigid_translation);
}

Eigen::Vector3d AnalyticCapGeometry3D::forward_rigid_vector(
    const Eigen::Vector3d& reference) const
{
    if (!has_nontrivial_rigid_map())
        return reference;
    return options_.rigid_rotation * reference;
}

double AnalyticCapGeometry3D::flower_radius(
    const Eigen::Vector3d& direction) const
{
    const double x = direction.x();
    const double y = direction.y();
    const double z = direction.z();
    const double h4 = x*x*x*x - 6.0*x*x*y*y + y*y*y*y;
    return 1.0 + options_.flower_epsilon * h4
           + options_.flower_eta * (3.0*z*z - 1.0);
}

Eigen::Vector3d AnalyticCapGeometry3D::flower_radius_gradient(
    const Eigen::Vector3d& direction) const
{
    const double x = direction.x();
    const double y = direction.y();
    const double z = direction.z();
    return {options_.flower_epsilon * (4.0*x*x*x - 12.0*x*y*y),
            options_.flower_epsilon * (-12.0*x*x*y + 4.0*y*y*y),
            6.0 * options_.flower_eta * z};
}

double AnalyticCapGeometry3D::level_set(const Eigen::Vector3d& point) const
{
    if (!point.allFinite())
        throw std::invalid_argument("Level-set point must be finite");
    const Eigen::Vector3d reference = inverse_rigid_point(point);
    if (options_.shape == AnalyticCapShape3D::Ellipsoid)
        return reference.cwiseQuotient(options_.ellipsoid_axes).squaredNorm()
               - 1.0;
    const double radius = reference.norm();
    if (!(radius > 1.0e-14))
        return -1.0;
    return radius - flower_radius(reference / radius);
}

Eigen::Vector3d AnalyticCapGeometry3D::level_set_gradient(
    const Eigen::Vector3d& point) const
{
    if (!point.allFinite())
        throw std::invalid_argument("Level-set point must be finite");
    const Eigen::Vector3d reference = inverse_rigid_point(point);
    Eigen::Vector3d reference_gradient;
    if (options_.shape == AnalyticCapShape3D::Ellipsoid) {
        reference_gradient = 2.0 * reference.cwiseQuotient(
            options_.ellipsoid_axes.cwiseProduct(options_.ellipsoid_axes));
    } else {
        const double radius = reference.norm();
        if (!(radius > 1.0e-14)) {
            reference_gradient = Eigen::Vector3d::UnitX();
        } else {
            const Eigen::Vector3d direction = reference / radius;
            const Eigen::Vector3d grad_r = flower_radius_gradient(direction);
            const Eigen::Vector3d tangential =
                grad_r - grad_r.dot(direction) * direction;
            reference_gradient = direction - tangential / radius;
        }
    }
    return forward_rigid_vector(reference_gradient);
}

Eigen::Vector3d AnalyticCapGeometry3D::outward_normal(
    const Eigen::Vector3d& point) const
{
    const Eigen::Vector3d gradient = level_set_gradient(point);
    const double norm = gradient.norm();
    if (!(norm > 0.0) || !std::isfinite(norm))
        throw std::runtime_error("Level-set gradient has no normal direction");
    return gradient / norm;
}

bool AnalyticCapGeometry3D::inside(const Eigen::Vector3d& point) const
{
    return level_set(point) < 0.0;
}

Eigen::Vector3d AnalyticCapGeometry3D::surface_point_from_direction(
    const Eigen::Vector3d& supplied_direction) const
{
    const double norm = supplied_direction.norm();
    if (!(norm > 1.0e-14) || !std::isfinite(norm)
        || !supplied_direction.allFinite())
        throw std::invalid_argument(
            "A nonzero finite reference direction is required");
    const Eigen::Vector3d direction = supplied_direction / norm;
    Eigen::Vector3d reference;
    if (options_.shape == AnalyticCapShape3D::Ellipsoid)
        reference = options_.ellipsoid_axes.cwiseProduct(direction);
    else
        reference = flower_radius(direction) * direction;
    return forward_rigid_point(reference);
}

AnalyticCapSurfaceEvaluation3D AnalyticCapGeometry3D::evaluate(
    int patch, double u, double v) const
{
    const ReferenceEvaluation ref = reference_geometry(options_, patch, u, v);
    AnalyticCapSurfaceEvaluation3D result;
    if (options_.shape == AnalyticCapShape3D::Ellipsoid) {
        result.point = options_.ellipsoid_axes.cwiseProduct(ref.point);
        result.tangents.col(0) =
            options_.ellipsoid_axes.cwiseProduct(ref.du);
        result.tangents.col(1) =
            options_.ellipsoid_axes.cwiseProduct(ref.dv);
        result.normal =
            (2.0 * result.point.cwiseQuotient(
                options_.ellipsoid_axes.cwiseProduct(options_.ellipsoid_axes)))
                .normalized();
    } else {
        const double radius = flower_radius(ref.point);
        const Eigen::Vector3d grad_r = flower_radius_gradient(ref.point);
        result.point = radius * ref.point;
        result.tangents.col(0) =
            radius * ref.du + grad_r.dot(ref.du) * ref.point;
        result.tangents.col(1) =
            radius * ref.dv + grad_r.dot(ref.dv) * ref.point;
        const Eigen::Vector3d tangential_gradient =
            grad_r - grad_r.dot(ref.point) * ref.point;
        result.normal =
            (ref.point - tangential_gradient / radius).normalized();
    }
    result.area_element =
        result.tangents.col(0).cross(result.tangents.col(1)).norm();
    if (!(result.area_element > 0.0) || !std::isfinite(result.area_element)
        || !result.point.allFinite() || !result.normal.allFinite())
        throw std::runtime_error("Invalid cap-atlas surface geometry");
    if (has_nontrivial_rigid_map()) {
        result.point = forward_rigid_point(result.point);
        result.tangents.col(0) = forward_rigid_vector(result.tangents.col(0));
        result.tangents.col(1) = forward_rigid_vector(result.tangents.col(1));
        result.normal = forward_rigid_vector(result.normal);
    }
    return result;
}

AnalyticCapPatchLocation3D AnalyticCapGeometry3D::locate(
    const Eigen::Vector3d& physical) const
{
    AnalyticCapPatchLocation3D result;
    if (!physical.allFinite())
        return result;
    const Eigen::Vector3d reference = inverse_rigid_point(physical);
    Eigen::Vector3d direction;
    if (options_.shape == AnalyticCapShape3D::Ellipsoid)
        direction = reference.cwiseQuotient(options_.ellipsoid_axes);
    else
        direction = reference;
    const double norm = direction.norm();
    if (!(norm > 1.0e-14))
        return result;
    direction /= norm;

    const double x = direction.x();
    const double y = direction.y();
    const double theta = std::acos(std::clamp(direction.z(), -1.0, 1.0));
    const double theta_c = std::asin(options_.polar_radius);
    const double rho = std::hypot(x, y);
    const double maximum = std::max(std::abs(x), std::abs(y));

    AtlasSide side;
    double side_parameter = 0.0;
    if (std::abs(y) >= std::abs(x)) {
        if (y >= 0.0) {
            side = AtlasSide::North;
            side_parameter = 0.5 * (x / std::max(y, 1.0e-15) + 1.0);
        } else {
            side = AtlasSide::South;
            side_parameter = 0.5 * (1.0 - x / std::max(-y, 1.0e-15));
        }
    } else if (x >= 0.0) {
        side = AtlasSide::East;
        side_parameter = 0.5 * (1.0 - y / std::max(x, 1.0e-15));
    } else {
        side = AtlasSide::West;
        side_parameter = 0.5 * (y / std::max(-x, 1.0e-15) + 1.0);
    }
    side_parameter = std::clamp(side_parameter, 0.0, 1.0);

    const bool north = theta < theta_c - 1.0e-11;
    const bool south = theta > pi - theta_c + 1.0e-11;
    if (north || south) {
        const int hemisphere = north ? +1 : -1;
        if (maximum <= options_.square_half_width + 1.0e-11) {
            result.patch = atlas_patch_index(
                AtlasPatchKind::PolarCentral, AtlasSide::None, hemisphere);
            result.u = 0.5 * (x / options_.square_half_width + 1.0);
            result.v = 0.5 * (y / options_.square_half_width + 1.0);
        } else {
            result.patch = atlas_patch_index(
                AtlasPatchKind::PolarRing, side, hemisphere);
            const Eigen::Vector2d q0 = square_side_point(
                side, side_parameter, options_.square_half_width);
            result.u = (rho - q0.norm())
                       / (options_.polar_radius - q0.norm());
            result.v = side_parameter;
        }
    } else {
        result.patch = atlas_patch_index(
            AtlasPatchKind::MeridianBelt, side, 0);
        result.u = (theta - theta_c) / (pi - 2.0 * theta_c);
        result.v = side_parameter;
    }
    result.u = std::clamp(result.u, 0.0, 1.0);
    result.v = std::clamp(result.v, 0.0, 1.0);
    result.valid = result.patch >= 0;
    return result;
}

} // namespace kfbim::geometry3d
