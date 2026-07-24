#include "restrict_owner_geometry_preprocessor_3d.hpp"

#include <algorithm>
#include <cmath>
#include <limits>
#include <stdexcept>
#include <utility>

namespace kfbim::app3d {
namespace {

using DecisionKind = RestrictOwnerDecisionKind3D;
using FallbackCause = RestrictOwnerFallbackCause3D;
using NormalizedClass = RestrictOwnerNormalizedClass3D;
using QueryPath = RestrictOwnerQueryPath3D;

void validate_cloud_topology(
    const NativeNurbsSurface3D& surface,
    const SurfaceDofCloud3D& cloud)
{
    const std::size_t patch_count = surface.patches.size();
    if (patch_count == 0
        || cloud.patches.empty()
        || cloud.dofs.empty()
        || cloud.patches.size() != patch_count
        || surface.smooth_neighbors.size() != patch_count
        || surface.patch_components.size() != patch_count) {
        throw std::invalid_argument(
            "restrict-owner preprocessor has empty or incompatible "
            "surface DOF cloud topology");
    }

    std::size_t expected_first = 0;
    for (std::size_t patch = 0; patch < cloud.patches.size(); ++patch) {
        const SurfaceDofPatch3D& dof_patch = cloud.patches[patch];
        if (dof_patch.nu <= 0 || dof_patch.nv <= 0
            || dof_patch.first_dof < 0
            || static_cast<std::size_t>(dof_patch.first_dof)
                   != expected_first) {
            throw std::invalid_argument(
                "restrict-owner preprocessor has empty or incompatible "
                "surface DOF cloud topology");
        }
        const std::size_t nu =
            static_cast<std::size_t>(dof_patch.nu);
        const std::size_t nv =
            static_cast<std::size_t>(dof_patch.nv);
        if (nu > std::numeric_limits<std::size_t>::max() / nv
            || expected_first
                   > std::numeric_limits<std::size_t>::max() - nu * nv) {
            throw std::invalid_argument(
                "restrict-owner preprocessor surface DOF topology "
                "overflows its index range");
        }
        expected_first += nu * nv;
        for (int smooth_patch : dof_patch.smooth_patch_ids) {
            if (smooth_patch < 0
                || smooth_patch >= static_cast<int>(patch_count)) {
                throw std::invalid_argument(
                    "restrict-owner preprocessor has empty or incompatible "
                    "surface DOF cloud topology");
            }
        }
    }
    if (expected_first != cloud.dofs.size()) {
        throw std::invalid_argument(
            "restrict-owner preprocessor has empty or incompatible "
            "surface DOF cloud topology");
    }
    for (std::size_t patch = 0; patch < cloud.patches.size(); ++patch) {
        const auto& dof_patch = cloud.patches[patch];
        for (int i = 0; i < dof_patch.nu; ++i) {
            for (int j = 0; j < dof_patch.nv; ++j) {
                const int index = dof_patch.dof_index(i, j);
                if (index < 0
                    || index >= static_cast<int>(cloud.dofs.size())) {
                    throw std::invalid_argument(
                        "restrict-owner preprocessor has empty or "
                        "incompatible surface DOF cloud topology");
                }
                const SurfaceDof3D& dof =
                    cloud.dofs[static_cast<std::size_t>(index)];
                if (dof.patch_id != static_cast<int>(patch)
                    || dof.i != i || dof.j != j) {
                    throw std::invalid_argument(
                        "restrict-owner preprocessor has empty or "
                        "incompatible surface DOF cloud topology");
                }
            }
        }
    }
}

geometry3d::NurbsSurfaceIntersector3D make_intersector(
    const NativeNurbsSurface3D& surface,
    const SurfaceDofCloud3D& cloud,
    double grid_spacing,
    const RestrictOwnerPreprocessOptions3D& options)
{
    if (!std::isfinite(grid_spacing) || grid_spacing <= 0.0) {
        throw std::invalid_argument(
            "restrict-owner preprocessor requires positive finite grid "
            "spacing");
    }
    if (options.maximum_independent_crossings != 2) {
        throw std::invalid_argument(
            "restrict-owner preprocessor requires exactly two maximum "
            "independent crossings");
    }
    validate_cloud_topology(surface, cloud);

    geometry3d::NurbsSurfaceIntersectorOptions3D intersector_options;
    intersector_options.maximum_element_extent = 2.0 * grid_spacing;
    intersector_options.local_max_subdivision_depth = 4;
    return geometry3d::NurbsSurfaceIntersector3D(
        surface.geometry_model(), intersector_options);
}

std::size_t path_index(QueryPath path)
{
    return static_cast<std::size_t>(path);
}

std::size_t fallback_index(FallbackCause cause)
{
    return static_cast<std::size_t>(cause);
}

} // namespace

const char* restrict_owner_preprocess_mode_name_3d(
    RestrictOwnerPreprocessMode3D mode)
{
    switch (mode) {
    case RestrictOwnerPreprocessMode3D::FullIntersectionReference:
        return "full_intersection_reference";
    case RestrictOwnerPreprocessMode3D::OptimizedIntersection:
        return "optimized_intersection";
    case RestrictOwnerPreprocessMode3D::RegionClosestHybrid:
        return "region_closest_hybrid";
    }
    throw std::invalid_argument(
        "invalid restrict-owner preprocessing mode");
}

RestrictOwnerPreprocessMode3D
parse_restrict_owner_preprocess_mode_3d(const std::string& text)
{
    if (text == "full_intersection_reference")
        return RestrictOwnerPreprocessMode3D::FullIntersectionReference;
    if (text == "optimized_intersection")
        return RestrictOwnerPreprocessMode3D::OptimizedIntersection;
    if (text == "region_closest_hybrid")
        return RestrictOwnerPreprocessMode3D::RegionClosestHybrid;
    throw std::invalid_argument(
        "invalid restrict-owner preprocessing mode: " + text);
}

RestrictOwnerGeometryPreprocessor3D::
RestrictOwnerGeometryPreprocessor3D(
    const NativeNurbsSurface3D& surface,
    const SurfaceDofCloud3D& cloud,
    double grid_spacing,
    RestrictOwnerPreprocessMode3D mode,
    RestrictOwnerPreprocessOptions3D options)
    : surface_(surface)
    , cloud_(cloud)
    , intersector_(
          make_intersector(surface, cloud, grid_spacing, options))
    , mode_(mode)
    , options_(options)
{
    switch (mode_) {
    case RestrictOwnerPreprocessMode3D::FullIntersectionReference:
    case RestrictOwnerPreprocessMode3D::OptimizedIntersection:
    case RestrictOwnerPreprocessMode3D::RegionClosestHybrid:
        return;
    }
    throw std::invalid_argument(
        "invalid restrict-owner preprocessing mode");
}

bool RestrictOwnerGeometryPreprocessor3D::is_compatible_patch(
    int target_patch,
    int patch) const
{
    if (target_patch < 0
        || target_patch >= static_cast<int>(cloud_.patches.size())
        || patch < 0
        || patch >= static_cast<int>(cloud_.patches.size())) {
        throw std::invalid_argument(
            "restrict-owner preprocessor patch index is invalid");
    }
    const auto& compatible =
        cloud_.patches[static_cast<std::size_t>(target_patch)]
            .smooth_patch_ids;
    return patch == target_patch
        || std::find(
               compatible.begin(), compatible.end(), patch)
               != compatible.end();
}

RestrictOwnerGeometryPreprocessor3D::CandidatePartition
RestrictOwnerGeometryPreprocessor3D::partition_candidates(
    int target_patch,
    std::vector<Candidate> candidates) const
{
    CandidatePartition partition;
    partition.compatible.reserve(candidates.size());
    partition.foreign.reserve(candidates.size());
    for (Candidate& candidate : candidates) {
        if (is_compatible_patch(
                target_patch, candidate.patch_index())) {
            partition.compatible.push_back(std::move(candidate));
        } else {
            partition.foreign.push_back(std::move(candidate));
        }
    }
    return partition;
}

RestrictOwnerPreprocessResult3D
RestrictOwnerGeometryPreprocessor3D::make_target_result(
    int target_dof,
    QueryPath path) const
{
    RestrictOwnerPreprocessResult3D result;
    result.owner_dof = target_dof;
    result.owner_class = NormalizedClass::Target;
    result.query_path = path;
    return result;
}

RestrictOwnerPreprocessResult3D
RestrictOwnerGeometryPreprocessor3D::make_fail_closed_target_result(
    int target_dof,
    QueryPath path,
    FallbackCause cause) const
{
    RestrictOwnerPreprocessResult3D result;
    result.owner_dof = target_dof;
    result.owner_class = NormalizedClass::FailClosedTarget;
    result.query_path = path;
    result.fallback_cause = cause;
    return result;
}

RestrictOwnerPreprocessResult3D
RestrictOwnerGeometryPreprocessor3D::make_foreign_result(
    int owner_dof,
    const geometry3d::NurbsSurfaceCrossing3D& crossing,
    QueryPath path) const
{
    RestrictOwnerPreprocessResult3D result;
    result.owner_dof = owner_dof;
    result.owner_class = NormalizedClass::UniqueForeign;
    result.query_path = path;
    result.foreign_crossing = crossing;
    return result;
}

FallbackCause
RestrictOwnerGeometryPreprocessor3D::classify_fallback_cause(
    const geometry3d::NurbsSurfaceIntersectionResult3D& intersection,
    const RestrictOwnerDecision3D& decision) const
{
    if (intersection.overlap_detected)
        return FallbackCause::Overlap;
    if (intersection.diagnostics.unresolved_candidates > 0
        || intersection.diagnostics.ambiguous_root_clusters > 0) {
        return FallbackCause::Unresolved;
    }
    if (!intersection.crossings.empty()
        && intersection.crossings.front().feature_edge_contact) {
        return FallbackCause::FeatureContact;
    }
    if (!intersection.crossings.empty()
        && intersection.crossings.front().transversality
               <= intersection.crossings.front()
                      .reliable_transversality_tolerance) {
        return FallbackCause::NearTangent;
    }
    if (decision.kind == DecisionKind::DegenerateCrossingFallback)
        return FallbackCause::NearTangent;
    return FallbackCause::Unresolved;
}

RestrictOwnerPreprocessResult3D
RestrictOwnerGeometryPreprocessor3D::normalize_intersection_result(
    int target_dof,
    const Eigen::Vector3d& query,
    const Eigen::Vector3d& support,
    const geometry3d::NurbsSurfaceIntersectionResult3D& intersection,
    QueryPath path) const
{
    const RestrictOwnerDecision3D decision =
        select_restrict_correction_owner_3d(
            target_dof, query, support, surface_, cloud_, intersection);
    RestrictOwnerPreprocessResult3D result;
    switch (decision.kind) {
    case DecisionKind::ForeignNonG1SingleCrossing:
        result = make_foreign_result(
            decision.owner_dof, intersection.crossings.front(), path);
        break;
    case DecisionKind::MultipleCrossingFallback:
        result = make_fail_closed_target_result(
            target_dof, path, FallbackCause::MultipleCrossings);
        break;
    case DecisionKind::DegenerateCrossingFallback:
    case DecisionKind::AmbiguousEdgeFallback:
        result = make_fail_closed_target_result(
            target_dof, path,
            classify_fallback_cause(intersection, decision));
        break;
    case DecisionKind::TargetSideNode:
    case DecisionKind::TargetOrG1SingleCrossing:
    case DecisionKind::NoCrossingFallback:
        result = make_target_result(target_dof, path);
        break;
    default:
        throw std::logic_error(
            "invalid restrict-owner decision kind");
    }
    result.legacy_decision_kind = decision.kind;
    return result;
}

RestrictOwnerPreprocessResult3D
RestrictOwnerGeometryPreprocessor3D::full_reference_result(
    int target_dof,
    const Eigen::Vector3d& query,
    const Eigen::Vector3d& support,
    QueryPath path)
{
    try {
        const auto intersection =
            intersector_.intersect_segment(query, support);
        return normalize_intersection_result(
            target_dof, query, support, intersection, path);
    } catch (const geometry3d::
                 UnresolvedNurbsIntersectionCandidate3D&) {
        return make_fail_closed_target_result(
            target_dof, path, FallbackCause::Unresolved);
    } catch (const std::runtime_error& error) {
        if (std::string(error.what())
            == "coincident roots on unrelated NURBS patches") {
            return make_fail_closed_target_result(
                target_dof, path, FallbackCause::Coincidence);
        }
        throw;
    }
}

RestrictOwnerPreprocessResult3D
RestrictOwnerGeometryPreprocessor3D::optimized_intersection_result(
    int,
    const Eigen::Vector3d&,
    const Eigen::Vector3d&)
{
    throw std::logic_error(
        "optimized restrict-owner preprocessing is not implemented");
}

RestrictOwnerSampleResult3D
RestrictOwnerGeometryPreprocessor3D::preprocess_sample(
    const RestrictOwnerSampleInput3D& input)
{
    if (input.target_dof < 0
        || input.target_dof >= static_cast<int>(cloud_.dofs.size())) {
        throw std::invalid_argument(
            "restrict-owner preprocessing target DOF is outside the "
            "surface cloud");
    }
    if (!input.query.allFinite()) {
        throw std::invalid_argument(
            "restrict-owner preprocessing sample points must be finite");
    }
    for (std::size_t q = 0; q < input.support_points.size(); ++q) {
        if (!input.support_points[q].allFinite()) {
            throw std::invalid_argument(
                "restrict-owner preprocessing sample points must be finite");
        }
        if (input.wrong_side[q]
            && input.support_points[q] == input.query) {
            throw std::invalid_argument(
                "restrict-owner preprocessing wrong-side segment must "
                "have nonzero length");
        }
    }

    RestrictOwnerSampleResult3D sample_result;
    const std::uint64_t before_queries =
        diagnostics_.wrong_side_queries;
    std::uint64_t classified = 0;
    for (std::size_t q = 0; q < input.wrong_side.size(); ++q) {
        if (!input.wrong_side[q])
            continue;

        RestrictOwnerPreprocessResult3D result;
        switch (mode_) {
        case RestrictOwnerPreprocessMode3D::FullIntersectionReference:
            result = full_reference_result(
                input.target_dof, input.query,
                input.support_points[q], QueryPath::FullIntersection);
            break;
        case RestrictOwnerPreprocessMode3D::OptimizedIntersection:
        case RestrictOwnerPreprocessMode3D::RegionClosestHybrid:
            result = optimized_intersection_result(
                input.target_dof, input.query,
                input.support_points[q]);
            break;
        default:
            throw std::logic_error(
                "invalid restrict-owner preprocessing mode");
        }

        const std::size_t result_path = path_index(result.query_path);
        const std::size_t result_fallback =
            fallback_index(result.fallback_cause);
        if (result_path >= diagnostics_.path_counts.size()
            || result_fallback >= diagnostics_.fallback_counts.size()) {
            throw std::logic_error(
                "restrict-owner preprocessing produced an invalid "
                "diagnostic classification");
        }
        ++diagnostics_.wrong_side_queries;
        ++diagnostics_.path_counts[result_path];
        ++diagnostics_.fallback_counts[result_fallback];
        switch (result.owner_class) {
        case NormalizedClass::Target:
            ++diagnostics_.target_decisions;
            break;
        case NormalizedClass::UniqueForeign:
            ++diagnostics_.foreign_decisions;
            break;
        case NormalizedClass::FailClosedTarget:
            ++diagnostics_.fail_closed_target_decisions;
            break;
        default:
            throw std::logic_error(
                "restrict-owner preprocessing produced an invalid "
                "normalized class");
        }
        sample_result.nodes[q] = std::move(result);
        ++classified;
    }

    if (classified
        != diagnostics_.wrong_side_queries - before_queries) {
        throw std::logic_error(
            "restrict-owner preprocessing did not classify every "
            "wrong-side node");
    }
    return sample_result;
}

RestrictOwnerPreprocessMode3D
RestrictOwnerGeometryPreprocessor3D::mode() const noexcept
{
    return mode_;
}

const RestrictOwnerPreprocessDiagnostics3D&
RestrictOwnerGeometryPreprocessor3D::diagnostics() const noexcept
{
    return diagnostics_;
}

void RestrictOwnerGeometryPreprocessor3D::reset_diagnostics()
{
    diagnostics_ = {};
}

} // namespace kfbim::app3d
