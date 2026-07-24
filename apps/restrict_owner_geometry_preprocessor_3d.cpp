#include "restrict_owner_geometry_preprocessor_3d.hpp"

#include <algorithm>
#include <chrono>
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
using CertificateKind =
    geometry3d::NurbsElementSegmentCertificateKind3D;

class OptionalStageTimer {
public:
    OptionalStageTimer(bool enabled, double& accumulator)
        : enabled_(enabled)
        , accumulator_(accumulator)
    {
        if (enabled_)
            start_ = Clock::now();
    }

    ~OptionalStageTimer()
    {
        if (enabled_) {
            accumulator_ += std::chrono::duration<double>(
                Clock::now() - start_).count();
        }
    }

private:
    using Clock = std::chrono::steady_clock;

    bool enabled_ = false;
    double& accumulator_;
    Clock::time_point start_{};
};

void accumulate_certificate_diagnostics(
    RestrictOwnerPreprocessDiagnostics3D& diagnostics,
    const geometry3d::NurbsSurfaceCandidateCertificate3D& certificate)
{
    const auto& work = certificate.diagnostics;
    if (work.conservative_rejections > 0
        && work.closest_point_attempts == 0) {
        ++diagnostics.control_hull_rejections;
    }
    diagnostics.closest_point_attempts +=
        static_cast<std::uint64_t>(work.closest_point_attempts);
    diagnostics.closest_point_converged +=
        static_cast<std::uint64_t>(
            work.roots_recovered_by_closest_point
            + work.terminal_misses_by_closest_point);
    diagnostics.closest_point_iterations +=
        static_cast<std::uint64_t>(work.closest_point_iterations);
}

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

double segment_parameter_tolerance(
    double geometry_tolerance,
    double segment_length)
{
    return std::max(
        16.0e-12, 16.0 * geometry_tolerance / segment_length);
}

double patch_parameter_tolerance(
    const geometry3d::NurbsSurfacePatch3D& patch)
{
    const double scale = std::max({
        1.0,
        patch.domain_end_u() - patch.domain_start_u(),
        patch.domain_end_v() - patch.domain_start_v()});
    return 16.0e-12 * scale;
}

std::optional<FallbackCause> filtered_fallback_cause(
    const NativeNurbsSurface3D& surface,
    double geometry_tolerance,
    const Eigen::Vector3d& query,
    const Eigen::Vector3d& support,
    const geometry3d::NurbsSurfaceIntersectionResult3D& intersection)
{
    if (intersection.overlap_detected)
        return FallbackCause::Overlap;
    if (intersection.diagnostics.unresolved_candidates > 0
        || intersection.diagnostics.ambiguous_root_clusters > 0) {
        return FallbackCause::Unresolved;
    }
    const double edge_tolerance = segment_parameter_tolerance(
        geometry_tolerance, (support - query).norm());
    for (const auto& root : intersection.crossings) {
        if (root.edge_parameter <= edge_tolerance
            || root.edge_parameter >= 1.0 - edge_tolerance) {
            return FallbackCause::Endpoint;
        }
        if (root.feature_edge_contact)
            return FallbackCause::FeatureContact;
        if (root.transversality
            <= root.reliable_transversality_tolerance) {
            return FallbackCause::NearTangent;
        }
    }
    if (intersection.diagnostics.seam_deduplications > 0)
        return FallbackCause::Seam;
    for (const auto& root : intersection.crossings) {
        const auto& patch =
            surface.patches[static_cast<std::size_t>(root.patch_index)];
        const double tolerance = patch_parameter_tolerance(patch);
        if (root.u <= patch.domain_start_u() + tolerance
            || root.u >= patch.domain_end_u() - tolerance
            || root.v <= patch.domain_start_v() + tolerance
            || root.v >= patch.domain_end_v() - tolerance) {
            return FallbackCause::ParameterBoundary;
        }
    }
    return std::nullopt;
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
    int target_dof,
    const Eigen::Vector3d& query,
    const Eigen::Vector3d& support)
{
    OptionalStageTimer optimized_timer(
        options_.collect_stage_timings,
        diagnostics_.optimized_intersection_seconds);

    const int target_patch =
        cloud_.dofs[static_cast<std::size_t>(target_dof)].patch_id;
    CandidatePartition partition = partition_candidates(
        target_patch,
        intersector_.conservative_segment_candidates(query, support));
    diagnostics_.compatible_aabb_candidates +=
        static_cast<std::uint64_t>(partition.compatible.size());
    diagnostics_.foreign_aabb_candidates +=
        static_cast<std::uint64_t>(partition.foreign.size());

    if (partition.foreign.empty()) {
        return make_target_result(
            target_dof, QueryPath::SegmentTargetOnly);
    }

    std::vector<Candidate> ordered_candidates;
    ordered_candidates.reserve(
        partition.foreign.size() + partition.compatible.size());
    for (Candidate& candidate : partition.foreign)
        ordered_candidates.push_back(std::move(candidate));
    for (Candidate& candidate : partition.compatible)
        ordered_candidates.push_back(std::move(candidate));

    const auto full_fallback = [&](FallbackCause cause) {
        OptionalStageTimer fallback_timer(
            options_.collect_stage_timings,
            diagnostics_.full_fallback_seconds);
        ++diagnostics_.full_fallback_calls;
        RestrictOwnerPreprocessResult3D result = full_reference_result(
            target_dof, query, support,
            QueryPath::FullIntersectionFallback);
        result.query_path = QueryPath::FullIntersectionFallback;
        result.fallback_cause = cause;
        return result;
    };

    ++diagnostics_.optimized_intersection_calls;
    try {
        geometry3d::NurbsSurfaceFilteredIntersectionOptions3D
            filtered_options;
        filtered_options.maximum_independent_crossings =
            options_.maximum_independent_crossings;
        const auto filtered = intersector_.intersect_segment_candidates(
            query, support, ordered_candidates, filtered_options);

        if (filtered.independent_crossing_limit_reached) {
            return make_fail_closed_target_result(
                target_dof, QueryPath::OptimizedIntersection,
                FallbackCause::MultipleCrossings);
        }
        if (!filtered.all_candidates_processed) {
            return full_fallback(FallbackCause::Unresolved);
        }
        if (const auto cause = filtered_fallback_cause(
                surface_, intersector_.geometry_tolerance(),
                query, support, filtered.intersection)) {
            return full_fallback(*cause);
        }

        return normalize_intersection_result(
            target_dof, query, support, filtered.intersection,
            QueryPath::OptimizedIntersection);
    } catch (const geometry3d::
                 UnresolvedNurbsIntersectionCandidate3D&) {
        return full_fallback(FallbackCause::Unresolved);
    } catch (const std::runtime_error& error) {
        if (std::string(error.what())
            == "coincident roots on unrelated NURBS patches") {
            return full_fallback(FallbackCause::Coincidence);
        }
        throw;
    }
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

    const int target_patch =
        cloud_.dofs[static_cast<std::size_t>(input.target_dof)].patch_id;
    bool hybrid_sweep_target = false;
    if (mode_ == RestrictOwnerPreprocessMode3D::RegionClosestHybrid) {
        geometry3d::NurbsAabb3D sweep{input.query, input.query};
        for (const Eigen::Vector3d& point : input.support_points) {
            sweep.lower = sweep.lower.cwiseMin(point);
            sweep.upper = sweep.upper.cwiseMax(point);
        }
        CandidatePartition sweep_partition;
        {
            OptionalStageTimer region_timer(
                options_.collect_stage_timings,
                diagnostics_.region_seconds);
            sweep_partition = partition_candidates(
                target_patch,
                intersector_.conservative_candidates(sweep));
        }
        diagnostics_.compatible_aabb_candidates +=
            static_cast<std::uint64_t>(
                sweep_partition.compatible.size());
        diagnostics_.foreign_aabb_candidates +=
            static_cast<std::uint64_t>(sweep_partition.foreign.size());
        hybrid_sweep_target = sweep_partition.foreign.empty();
    }

    const auto hybrid_segment_result =
        [&](const Eigen::Vector3d& support)
            -> RestrictOwnerPreprocessResult3D {
        CandidatePartition partition = partition_candidates(
            target_patch,
            intersector_.conservative_segment_candidates(
                input.query, support));
        diagnostics_.compatible_aabb_candidates +=
            static_cast<std::uint64_t>(partition.compatible.size());
        diagnostics_.foreign_aabb_candidates +=
            static_cast<std::uint64_t>(partition.foreign.size());
        if (partition.foreign.empty()) {
            return make_target_result(
                input.target_dof, QueryPath::SegmentTargetOnly);
        }

        struct CandidateCertificate {
            bool foreign = false;
            geometry3d::NurbsSurfaceCandidateCertificate3D certificate;
        };
        std::vector<CandidateCertificate> certificates;
        certificates.reserve(
            partition.foreign.size() + partition.compatible.size());
        const auto certify_partition =
            [&](const std::vector<Candidate>& candidates, bool foreign) {
            for (const Candidate& candidate : candidates) {
                geometry3d::NurbsSurfaceCandidateCertificate3D
                    certificate;
                {
                    OptionalStageTimer closest_timer(
                        options_.collect_stage_timings,
                        diagnostics_.closest_point_seconds);
                    certificate =
                        intersector_.certify_candidate_segment(
                            candidate, input.query, support);
                }
                accumulate_certificate_diagnostics(
                    diagnostics_, certificate);
                switch (certificate.kind) {
                case CertificateKind::CertifiedMiss:
                    ++diagnostics_.closest_certified_misses;
                    break;
                case CertificateKind::CertifiedUniqueTransverseRoot:
                    ++diagnostics_.closest_certified_roots;
                    break;
                case CertificateKind::Unresolved:
                    ++diagnostics_.closest_unresolved;
                    break;
                }
                certificates.push_back(
                    {foreign, std::move(certificate)});
            }
        };
        certify_partition(partition.foreign, true);
        certify_partition(partition.compatible, false);

        const bool all_miss = std::all_of(
            certificates.begin(), certificates.end(),
            [](const CandidateCertificate& work) {
                return work.certificate.kind
                    == CertificateKind::CertifiedMiss;
            });
        if (all_miss) {
            const bool used_closest = std::any_of(
                certificates.begin(), certificates.end(),
                [](const CandidateCertificate& work) {
                    return work.certificate.diagnostics
                               .closest_point_attempts > 0;
                });
            return make_target_result(
                input.target_dof,
                used_closest
                    ? QueryPath::ClosestCertifiedMiss
                    : QueryPath::SegmentTargetOnly);
        }

        const geometry3d::NurbsSurfaceCrossing3D* safe_root = nullptr;
        bool safe_state = true;
        for (const CandidateCertificate& work : certificates) {
            const auto& certificate = work.certificate;
            const bool root_is_safe =
                work.foreign
                && certificate.kind
                       == CertificateKind::CertifiedUniqueTransverseRoot
                && certificate.crossing.has_value()
                && certificate.segment_parameter_strictly_interior
                && certificate.element_parameter_strictly_interior
                && certificate.patch_parameter_strictly_interior
                && !certificate.crossing->feature_edge_contact
                && certificate.crossing->transversality
                       > certificate.crossing
                             ->reliable_transversality_tolerance;
            if (root_is_safe && safe_root == nullptr) {
                safe_root = &*certificate.crossing;
            } else if (certificate.kind != CertificateKind::CertifiedMiss) {
                safe_state = false;
            }
        }
        if (safe_state && safe_root != nullptr) {
            geometry3d::NurbsSurfaceIntersectionResult3D intersection;
            intersection.crossings.push_back(*safe_root);
            const auto result = normalize_intersection_result(
                input.target_dof, input.query, support, intersection,
                QueryPath::ClosestCertifiedRoot);
            if (result.owner_class == NormalizedClass::UniqueForeign)
                return result;
        }
        return optimized_intersection_result(
            input.target_dof, input.query, support);
    };

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
            result = optimized_intersection_result(
                input.target_dof, input.query,
                input.support_points[q]);
            break;
        case RestrictOwnerPreprocessMode3D::RegionClosestHybrid:
            if (hybrid_sweep_target) {
                result = make_target_result(
                    input.target_dof, QueryPath::SweepTargetOnly);
            } else {
                result = hybrid_segment_result(
                    input.support_points[q]);
            }
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
