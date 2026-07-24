#include "crossing_owner_restrict_3d.hpp"

#include <algorithm>
#include <cmath>
#include <limits>
#include <stdexcept>

namespace kfbim::app3d {
namespace {

constexpr double kMinimumNormalAlignment = 0.50;

void validate_cloud(const NativeNurbsSurface3D& surface,
                    const SurfaceDofCloud3D& cloud)
{
    if (surface.patches.empty()
        || surface.patches.size() != cloud.patches.size()
        || surface.smooth_neighbors.size() != surface.patches.size()) {
        throw std::invalid_argument(
            "crossing-owner selection has incompatible surface DOF cloud");
    }
}

void validate_root(const geometry3d::NurbsSurfaceCrossing3D& root,
                   const NativeNurbsSurface3D& surface)
{
    if (root.patch_index < 0
        || root.patch_index >= static_cast<int>(surface.patches.size())) {
        throw std::invalid_argument(
            "crossing-owner selection has an invalid crossing patch");
    }
    if (!std::isfinite(root.u) || !std::isfinite(root.v)
        || !std::isfinite(root.edge_parameter) || !root.point.allFinite()
        || !root.normal.allFinite() || !std::isfinite(root.residual)
        || !std::isfinite(root.transversality)
        || root.normal.squaredNorm() == 0.0 || root.residual < 0.0
        || root.transversality < 0.0) {
        throw std::invalid_argument(
            "crossing-owner selection has nonfinite or invalid crossing data");
    }
}

double reliable_transversality_tolerance(
    const NativeNurbsSurface3D& surface,
    double segment_length)
{
    const double diameter =
        surface.geometry_model().control_bounds().diameter();
    const double geometry_tolerance =
        std::max(1.0e-12 * diameter, 1.0e-14);
    const double geometry_scale =
        std::max({diameter, segment_length, geometry_tolerance});
    return std::max(
        32.0 * std::sqrt(std::numeric_limits<double>::epsilon()),
        std::sqrt(8.0 * geometry_tolerance / geometry_scale));
}

RestrictOwnerDecision3D target_decision(
    int target_dof,
    RestrictOwnerDecisionKind3D kind,
    const geometry3d::NurbsSurfaceCrossing3D* root = nullptr)
{
    RestrictOwnerDecision3D decision;
    decision.owner_dof = target_dof;
    decision.kind = kind;
    if (root != nullptr) {
        decision.crossing_patch = root->patch_index;
        decision.segment_parameter = root->edge_parameter;
        decision.residual = root->residual;
        decision.transversality = root->transversality;
    }
    return decision;
}

bool is_target_or_g1_patch(const SurfaceDofCloud3D& cloud,
                           int target_patch,
                           int crossing_patch)
{
    const std::vector<int>& smooth =
        cloud.patches[static_cast<std::size_t>(target_patch)].smooth_patch_ids;
    return crossing_patch == target_patch
        || std::find(smooth.begin(), smooth.end(), crossing_patch)
               != smooth.end();
}

int nearest_crossing_candidate(
    const NativeNurbsSurface3D& surface,
    const SurfaceDofCloud3D& cloud,
    const geometry3d::NurbsSurfaceCrossing3D& root)
{
    const std::array<int, 4> candidates = parameter_dof_candidates_2x2(
        surface, cloud, root.patch_index, root.u, root.v);
    const Eigen::Vector3d reference_normal = root.normal.normalized();
    int nearest = -1;
    double best_distance = std::numeric_limits<double>::infinity();
    for (int pass = 0; pass < 2 && nearest < 0; ++pass) {
        for (int candidate : candidates) {
            if (candidate < 0
                || candidate >= static_cast<int>(cloud.dofs.size())) {
                throw std::invalid_argument(
                    "crossing-owner candidate DOF is invalid");
            }
            const SurfaceDof3D& dof =
                cloud.dofs[static_cast<std::size_t>(candidate)];
            if (!dof.point.allFinite() || !dof.normal.allFinite()) {
                throw std::invalid_argument(
                    "crossing-owner candidate DOF is nonfinite");
            }
            if (pass == 0
                && dof.normal.dot(reference_normal)
                    < kMinimumNormalAlignment) {
                continue;
            }
            const double distance = (dof.point - root.point).squaredNorm();
            if (distance < best_distance) {
                best_distance = distance;
                nearest = candidate;
            }
        }
    }
    if (nearest < 0) {
        throw std::runtime_error(
            "failed to associate crossing with parameter DOFs");
    }
    return nearest;
}

} // namespace

RestrictOwnerDecision3D select_restrict_correction_owner_3d(
    int target_dof,
    const Eigen::Vector3d& query,
    const Eigen::Vector3d& support,
    const NativeNurbsSurface3D& surface,
    const SurfaceDofCloud3D& cloud,
    const geometry3d::NurbsSurfaceIntersectionResult3D& intersection)
{
    validate_cloud(surface, cloud);
    if (target_dof < 0 || target_dof >= static_cast<int>(cloud.dofs.size())) {
        throw std::invalid_argument(
            "crossing-owner selection target DOF is invalid");
    }
    if (!query.allFinite() || !support.allFinite()) {
        throw std::invalid_argument(
            "crossing-owner selection segment endpoints must be finite");
    }
    const double segment_length = (support - query).norm();
    const double geometry_tolerance = std::max(
        1.0e-12 * surface.geometry_model().control_bounds().diameter(),
        1.0e-14);
    if (!std::isfinite(segment_length)
        || segment_length <= geometry_tolerance) {
        throw std::invalid_argument(
            "crossing-owner selection segment must have nonzero length");
    }
    const SurfaceDof3D& target =
        cloud.dofs[static_cast<std::size_t>(target_dof)];
    if (target.patch_id < 0
        || target.patch_id >= static_cast<int>(cloud.patches.size())) {
        throw std::invalid_argument(
            "crossing-owner selection target patch is invalid");
    }
    for (const auto& root : intersection.crossings)
        validate_root(root, surface);

    if (intersection.overlap_detected) {
        return target_decision(
            target_dof,
            RestrictOwnerDecisionKind3D::DegenerateCrossingFallback);
    }
    if (intersection.diagnostics.unresolved_candidates > 0
        || intersection.diagnostics.ambiguous_root_clusters > 0) {
        return target_decision(
            target_dof,
            RestrictOwnerDecisionKind3D::AmbiguousEdgeFallback);
    }
    if (intersection.crossings.empty()) {
        return target_decision(
            target_dof,
            RestrictOwnerDecisionKind3D::NoCrossingFallback);
    }
    if (intersection.crossings.size() != 1) {
        return target_decision(
            target_dof,
            RestrictOwnerDecisionKind3D::MultipleCrossingFallback);
    }

    const auto& root = intersection.crossings.front();
    if (root.feature_edge_contact) {
        return target_decision(
            target_dof,
            RestrictOwnerDecisionKind3D::AmbiguousEdgeFallback, &root);
    }
    if (root.transversality
        <= reliable_transversality_tolerance(surface, segment_length)) {
        return target_decision(
            target_dof,
            RestrictOwnerDecisionKind3D::DegenerateCrossingFallback, &root);
    }
    if (is_target_or_g1_patch(
            cloud, target.patch_id, root.patch_index)) {
        return target_decision(
            target_dof,
            RestrictOwnerDecisionKind3D::TargetOrG1SingleCrossing, &root);
    }

    return target_decision(
        nearest_crossing_candidate(surface, cloud, root),
        RestrictOwnerDecisionKind3D::ForeignNonG1SingleCrossing, &root);
}

} // namespace kfbim::app3d
