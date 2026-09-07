#include "closest_point_trace_extension_3d.hpp"

#include <algorithm>
#include <cmath>
#include <limits>

namespace kfbim::app3d {
namespace {

bool has_internal_c0_knot(const geometry3d::NurbsSurfacePatch3D& patch)
{
    const auto has_feature = [](const kfbim::geometry::NurbsBasis1D& basis) {
        const auto& knots = basis.knots();
        for (std::size_t first = 0; first < knots.size();) {
            std::size_t end = first + 1;
            while (end < knots.size() && knots[end] == knots[first]) ++end;
            if (knots[first] > basis.domain_start() && knots[first] < basis.domain_end()
                && end - first >= static_cast<std::size_t>(basis.degree()))
                return true;
            first = end;
        }
        return false;
    };
    return has_feature(patch.basis_u()) || has_feature(patch.basis_v());
}

bool touches_feature(const NativeNurbsSurface3D& surface, int patch_index,
                     double u, double v, double u_tolerance, double v_tolerance)
{
    const auto& patch = surface.patches[static_cast<std::size_t>(patch_index)];
    const auto touches = [&](const geometry3d::NurbsPatchEdgeInterval3D& side) {
        if (side.patch != patch_index) return false;
        double across = 0.0, boundary = 0.0, along = 0.0;
        double across_tolerance = 0.0, along_tolerance = 0.0;
        using Edge = geometry3d::NurbsPatchEdge3D;
        switch (side.edge) {
        case Edge::UMin:
        case Edge::UMax:
            across = u; along = v;
            boundary = side.edge == Edge::UMin ? patch.domain_start_u() : patch.domain_end_u();
            across_tolerance = u_tolerance; along_tolerance = v_tolerance;
            break;
        case Edge::VMin:
        case Edge::VMax:
            across = v; along = u;
            boundary = side.edge == Edge::VMin ? patch.domain_start_v() : patch.domain_end_v();
            across_tolerance = v_tolerance; along_tolerance = u_tolerance;
            break;
        }
        return std::abs(across - boundary) <= across_tolerance
            && along >= side.begin - along_tolerance
            && along <= side.end + along_tolerance;
    };
    for (const auto& connection : surface.geometric_connections)
        if (!connection.g1 && (touches(connection.first) || touches(connection.second)))
            return true;

    // Legacy metadata cannot certify an unlisted boundary as smooth.
    if (surface.geometric_connections.empty()) {
        const auto boundary_is_smooth = [&](int edge) {
            return surface.smooth_neighbors.size() == surface.patches.size()
                && surface.smooth_neighbors[static_cast<std::size_t>(patch_index)]
                                           [static_cast<std::size_t>(edge)].has_value();
        };
        if (std::abs(u - patch.domain_start_u()) <= u_tolerance && !boundary_is_smooth(0)) return true;
        if (std::abs(u - patch.domain_end_u()) <= u_tolerance && !boundary_is_smooth(1)) return true;
        if (std::abs(v - patch.domain_start_v()) <= v_tolerance && !boundary_is_smooth(2)) return true;
        if (std::abs(v - patch.domain_end_v()) <= v_tolerance && !boundary_is_smooth(3)) return true;
    }
    return false;
}
} // namespace

const char* closest_point_trace_anchor_status_name_3d(ClosestPointTraceAnchorStatus3D status) noexcept
{
    switch (status) {
    case ClosestPointTraceAnchorStatus3D::Accepted: return "accepted";
    case ClosestPointTraceAnchorStatus3D::InvalidInput: return "invalid_input";
    case ClosestPointTraceAnchorStatus3D::UnresolvedProjection: return "unresolved_projection";
    case ClosestPointTraceAnchorStatus3D::AmbiguousProjection: return "ambiguous_projection";
    case ClosestPointTraceAnchorStatus3D::IncompatibleSheet: return "incompatible_sheet";
    case ClosestPointTraceAnchorStatus3D::FeatureContact: return "feature_contact";
    case ClosestPointTraceAnchorStatus3D::NonNormalProjection: return "non_normal_projection";
    case ClosestPointTraceAnchorStatus3D::TooFar: return "too_far";
    }
    return "invalid_status";
}

ClosestPointTraceAnchor3D select_closest_point_trace_anchor_3d(
    const NativeNurbsSurface3D& surface,
    const geometry3d::NurbsSurfaceClosestPointResult3D& projection,
    const Eigen::Vector3d& support_point, const Eigen::Vector3d& trace_point,
    int target_patch, const Eigen::Vector3d& target_normal, double maximum_distance)
{
    using Status = ClosestPointTraceAnchorStatus3D;
    ClosestPointTraceAnchor3D result;
    const auto refuse = [&](Status status) {
        result.status = status;
        return result;
    };
    const int count = static_cast<int>(surface.patches.size());
    if (!support_point.allFinite() || !trace_point.allFinite() || !target_normal.allFinite()
        || target_patch < 0 || target_patch >= count
        || surface.patch_components.size() != surface.patches.size()
        || !std::isfinite(maximum_distance) || maximum_distance <= 0.0
        || std::abs(target_normal.norm() - 1.0) > 1e-8)
        return refuse(Status::InvalidInput);
    if (projection.status != geometry3d::NurbsSurfaceClosestPointStatus3D::Bounded
        || !projection.localized || !projection.normal_valid)
        return refuse(Status::UnresolvedProjection);
    if (projection.separated_near_ties)
        return refuse(Status::AmbiguousProjection);
    if (projection.patch_index < 0 || projection.patch_index >= count
        || !projection.point.allFinite() || !projection.normal.allFinite()
        || !std::isfinite(projection.u) || !std::isfinite(projection.v)
        || !std::isfinite(projection.distance_lower) || projection.distance_lower < 0.0
        || !std::isfinite(projection.distance_upper)
        || projection.distance_upper < projection.distance_lower
        || !std::isfinite(projection.localization_radius_upper)
        || projection.localization_radius_upper < 0.0)
        return refuse(Status::InvalidInput);

    const double scale = std::max({1.0, support_point.norm(), trace_point.norm(), maximum_distance});
    if (!std::isfinite(scale))
        return refuse(Status::InvalidInput);
    const double roundoff = 256.0 * std::numeric_limits<double>::epsilon() * scale;
    const double radius = projection.localization_radius_upper;
    if (radius > std::max(1e-8 * scale, 1e-3 * maximum_distance))
        return refuse(Status::UnresolvedProjection);
    const double location_tolerance = std::max(roundoff, 4.0 * radius);
    result.localization_radius = radius;
    result.distance = (support_point - projection.point).norm();
    if (!std::isfinite(result.distance))
        return refuse(Status::InvalidInput);
    if (projection.distance_upper > maximum_distance + roundoff
        || projection.distance_upper > (support_point - trace_point).norm() + roundoff
        || result.distance > projection.distance_upper + location_tolerance
        || result.distance + location_tolerance < projection.distance_lower)
        return refuse(Status::TooFar);

    const int owner_patch = projection.patch_index;
    // Patch adjacency does not describe creases at internal repeated knots.
    // Without parameter-side connectivity, conservatively refuse the whole
    // target/owner patch, even if this particular Q lies away from its crease.
    if (has_internal_c0_knot(surface.patches[static_cast<std::size_t>(target_patch)])
        || has_internal_c0_knot(surface.patches[static_cast<std::size_t>(owner_patch)]))
        return refuse(Status::FeatureContact);
    if (projection.component != surface.patch_components[static_cast<std::size_t>(owner_patch)]
        || projection.component != surface.patch_components[static_cast<std::size_t>(target_patch)])
        return refuse(Status::IncompatibleSheet);
    const auto sheet = smooth_patch_component(surface, target_patch);
    if (std::find(sheet.begin(), sheet.end(), owner_patch) == sheet.end())
        return refuse(Status::IncompatibleSheet);
    const auto& patch = surface.patches[static_cast<std::size_t>(owner_patch)];
    const double parameter_tolerance = 64.0 * std::max(patch.basis_u().tolerance(), patch.basis_v().tolerance());
    if (projection.u < patch.domain_start_u() || projection.u > patch.domain_end_u()
        || projection.v < patch.domain_start_v() || projection.v > patch.domain_end_v())
        return refuse(Status::InvalidInput);
    const auto derivatives = patch.evaluate_with_derivatives(projection.u, projection.v);
    const double jacobian = derivatives.du.cross(derivatives.dv).norm();
    if (!std::isfinite(jacobian) || jacobian <= roundoff * roundoff)
        return refuse(Status::NonNormalProjection);
    const auto normal = (derivatives.du.cross(derivatives.dv) / jacobian).eval();
    const double surface_residual = (derivatives.point - projection.point).norm();
    if (surface_residual > std::max(roundoff, 1e-10 * scale)
        || std::abs(projection.normal.norm() - 1.0) > 1e-8
        || normal.dot(projection.normal) < 1.0 - 1e-8)
        return refuse(Status::InvalidInput);
    if (normal.dot(target_normal) <= 0.5)
        return refuse(Status::IncompatibleSheet);

    // The reciprocal tangent metric gives a first-order parameter neighborhood
    // for the feature refusal check, including skew patches.
    const double u_tolerance = parameter_tolerance
        + location_tolerance * derivatives.dv.norm() / jacobian;
    const double v_tolerance = parameter_tolerance
        + location_tolerance * derivatives.du.norm() / jacobian;
    if (touches_feature(surface, owner_patch, projection.u, projection.v, u_tolerance, v_tolerance))
        return refuse(Status::FeatureContact);
    const Eigen::Vector3d displacement = support_point - projection.point;
    const double tangent_residual = (displacement - displacement.dot(normal) * normal).norm();
    if (tangent_residual > std::max(1e-9 * scale, location_tolerance))
        return refuse(Status::NonNormalProjection);

    for (const auto& candidate : projection.candidates) {
        if (candidate.distance_upper <= projection.distance_upper + roundoff
            && (candidate.point - projection.point).norm() > 2.0 * location_tolerance)
            return refuse(Status::AmbiguousProjection);
    }
    result.owner.nurbs_patch_index = owner_patch;
    result.owner.nurbs_parameter = {projection.u, projection.v};
    result.owner.crossing_point = projection.point;
    result.owner.crossing_normal = normal;
    result.owner.surface_component = projection.component;
    result.owner.crossing_residual = surface_residual;
    result.owner.edge_parameter = 1.0; // endpoint of A -> Q, not a parameter on A -> P
    result.owner.status = P2CrossingOwnerStatus3D::ExactIntersection;
    return refuse(Status::Accepted);
}

} // namespace kfbim::app3d
