#include "edge_patch_3d.hpp"

#include <cmath>
#include <stdexcept>

namespace kfbim::geometry3d {

namespace {

bool finite_vector(const Eigen::Vector3d& value)
{
    return value.allFinite();
}

double radius_for_edge(const EdgePatchConfig3D& config, int edge)
{
    double radius = config.radius;
    if (edge < static_cast<int>(config.radius_overrides_by_edge.size())) {
        const double candidate =
            config.radius_overrides_by_edge[static_cast<std::size_t>(edge)];
        if (std::isfinite(candidate) && candidate > 0.0)
            radius = candidate;
    }
    return radius;
}

} // namespace

std::vector<EdgePatch3D> build_edge_patches_3d(
    const std::vector<EdgePatchSource3D>& edges,
    const EdgePatchConfig3D& config)
{
    if (!config.enabled)
        return {};
    if (!std::isfinite(config.radius) || config.radius <= 0.0)
        throw std::invalid_argument("edge patch radius must be positive");
    if (!std::isfinite(config.options.geometry_tol)
        || config.options.geometry_tol < 0.0) {
        throw std::invalid_argument(
            "edge patch geometry tolerance must be finite and nonnegative");
    }

    std::vector<EdgePatch3D> patches;
    patches.reserve(edges.size());
    for (int edge = 0; edge < static_cast<int>(edges.size()); ++edge) {
        const EdgePatchSource3D& source =
            edges[static_cast<std::size_t>(edge)];
        if (!finite_vector(source.start) || !finite_vector(source.end))
            throw std::invalid_argument("edge patch endpoints must be finite");
        const Eigen::Vector3d span = source.end - source.start;
        const double length = span.norm();
        if (!(length > config.options.geometry_tol))
            throw std::invalid_argument("edge patch edge length must be positive");
        const double radius = radius_for_edge(config, edge);
        if (!std::isfinite(radius) || radius <= 0.0)
            throw std::invalid_argument("edge patch radius must be positive");

        patches.push_back({edge,
                           source.start,
                           source.end,
                           span / length,
                           length,
                           radius,
                           source.incident_patches});
    }
    return patches;
}

EdgePatchCoordinates3D edge_patch_coordinates_3d(
    const EdgePatch3D& patch,
    const Eigen::Vector3d& point)
{
    if (!(patch.length > 0.0) || !patch.tangent.allFinite()
        || !point.allFinite()) {
        throw std::invalid_argument("invalid edge patch coordinate query");
    }
    const Eigen::Vector3d offset = point - patch.start;
    const double s = offset.dot(patch.tangent);
    const Eigen::Vector3d axis_point = patch.start + s * patch.tangent;
    return {s, (point - axis_point).norm(), axis_point};
}

bool edge_patch_contains_point_3d(const EdgePatch3D& patch,
                                  const Eigen::Vector3d& point,
                                  double tolerance)
{
    if (!std::isfinite(tolerance) || tolerance < 0.0)
        throw std::invalid_argument(
            "edge patch query tolerance must be finite and nonnegative");
    if (!(patch.radius > 0.0))
        throw std::invalid_argument("edge patch radius must be positive");
    const EdgePatchCoordinates3D coordinates =
        edge_patch_coordinates_3d(patch, point);
    return coordinates.s >= -tolerance
        && coordinates.s <= patch.length + tolerance
        && coordinates.rho > tolerance
        && coordinates.rho <= patch.radius + tolerance;
}

double edge_patch_radial_signed_distance_3d(
    const EdgePatch3D& patch,
    const Eigen::Vector3d& point)
{
    if (!(patch.radius > 0.0))
        throw std::invalid_argument("edge patch radius must be positive");
    return edge_patch_coordinates_3d(patch, point).rho - patch.radius;
}

} // namespace kfbim::geometry3d
