#pragma once

#include "crossing_owner_restrict_3d.hpp"

#include <Eigen/Dense>

#include <array>
#include <cstddef>
#include <cstdint>
#include <optional>
#include <string>
#include <vector>

namespace kfbim::app3d {

constexpr std::size_t kRestrictOwnerTricubicSupportCount3D = 64;

enum class RestrictOwnerPreprocessMode3D {
    FullIntersectionReference,
    OptimizedIntersection,
    RegionClosestHybrid
};

enum class RestrictOwnerNormalizedClass3D {
    Target,
    UniqueForeign,
    FailClosedTarget
};

enum class RestrictOwnerQueryPath3D {
    FullIntersection,
    SweepTargetOnly,
    SegmentTargetOnly,
    ClosestCertifiedMiss,
    ClosestCertifiedRoot,
    OptimizedIntersection,
    FullIntersectionFallback,
    Count
};

enum class RestrictOwnerFallbackCause3D {
    None,
    MultipleCrossings,
    Unresolved,
    Overlap,
    Endpoint,
    NearTangent,
    FeatureContact,
    ParameterBoundary,
    Seam,
    Coincidence,
    Count
};

const char* restrict_owner_preprocess_mode_name_3d(
    RestrictOwnerPreprocessMode3D mode);

RestrictOwnerPreprocessMode3D
parse_restrict_owner_preprocess_mode_3d(const std::string& text);

struct RestrictOwnerSampleInput3D {
    int target_dof = -1;
    Eigen::Vector3d query = Eigen::Vector3d::Zero();
    std::array<Eigen::Vector3d,
               kRestrictOwnerTricubicSupportCount3D> support_points{};
    std::array<bool,
               kRestrictOwnerTricubicSupportCount3D> wrong_side{};
};

struct RestrictOwnerPreprocessResult3D {
    int owner_dof = -1;
    RestrictOwnerNormalizedClass3D owner_class =
        RestrictOwnerNormalizedClass3D::Target;
    RestrictOwnerQueryPath3D query_path =
        RestrictOwnerQueryPath3D::FullIntersection;
    RestrictOwnerFallbackCause3D fallback_cause =
        RestrictOwnerFallbackCause3D::None;
    std::optional<geometry3d::NurbsSurfaceCrossing3D> foreign_crossing;
    std::optional<RestrictOwnerDecisionKind3D>
        legacy_decision_kind;
};

struct RestrictOwnerSampleResult3D {
    std::array<std::optional<RestrictOwnerPreprocessResult3D>,
               kRestrictOwnerTricubicSupportCount3D> nodes;
};

struct RestrictOwnerPreprocessOptions3D {
    bool collect_stage_timings = false;
    int maximum_independent_crossings = 2;
};

struct RestrictOwnerPreprocessDiagnostics3D {
    std::uint64_t wrong_side_queries = 0;
    std::array<std::uint64_t,
        static_cast<std::size_t>(
            RestrictOwnerQueryPath3D::Count)> path_counts{};
    std::array<std::uint64_t,
        static_cast<std::size_t>(
            RestrictOwnerFallbackCause3D::Count)> fallback_counts{};
    std::uint64_t compatible_aabb_candidates = 0;
    std::uint64_t foreign_aabb_candidates = 0;
    std::uint64_t control_hull_rejections = 0;
    std::uint64_t closest_point_attempts = 0;
    std::uint64_t closest_point_converged = 0;
    std::uint64_t closest_point_iterations = 0;
    std::uint64_t closest_certified_misses = 0;
    std::uint64_t closest_certified_roots = 0;
    std::uint64_t closest_unresolved = 0;
    std::uint64_t optimized_intersection_calls = 0;
    std::uint64_t full_fallback_calls = 0;
    std::uint64_t target_decisions = 0;
    std::uint64_t foreign_decisions = 0;
    std::uint64_t fail_closed_target_decisions = 0;
    double region_seconds = 0.0;
    double closest_point_seconds = 0.0;
    double optimized_intersection_seconds = 0.0;
    double full_fallback_seconds = 0.0;
};

class RestrictOwnerGeometryPreprocessor3D {
public:
    RestrictOwnerGeometryPreprocessor3D(
        const NativeNurbsSurface3D& surface,
        const SurfaceDofCloud3D& cloud,
        double grid_spacing,
        RestrictOwnerPreprocessMode3D mode,
        RestrictOwnerPreprocessOptions3D options = {});

    RestrictOwnerSampleResult3D preprocess_sample(
        const RestrictOwnerSampleInput3D& input);

    RestrictOwnerPreprocessMode3D mode() const noexcept;
    const RestrictOwnerPreprocessDiagnostics3D&
    diagnostics() const noexcept;
    void reset_diagnostics();

private:
    using Candidate = geometry3d::NurbsSurfaceCandidate3D;

    struct CandidatePartition {
        std::vector<Candidate> compatible;
        std::vector<Candidate> foreign;
    };

    bool is_compatible_patch(int target_patch, int patch) const;
    CandidatePartition partition_candidates(
        int target_patch,
        std::vector<Candidate> candidates) const;
    RestrictOwnerPreprocessResult3D make_target_result(
        int target_dof,
        RestrictOwnerQueryPath3D path) const;
    RestrictOwnerPreprocessResult3D make_fail_closed_target_result(
        int target_dof,
        RestrictOwnerQueryPath3D path,
        RestrictOwnerFallbackCause3D cause) const;
    RestrictOwnerPreprocessResult3D make_foreign_result(
        int owner_dof,
        const geometry3d::NurbsSurfaceCrossing3D& crossing,
        RestrictOwnerQueryPath3D path) const;
    RestrictOwnerFallbackCause3D classify_fallback_cause(
        const geometry3d::NurbsSurfaceIntersectionResult3D& intersection,
        const RestrictOwnerDecision3D& decision) const;
    RestrictOwnerPreprocessResult3D normalize_intersection_result(
        int target_dof,
        const Eigen::Vector3d& query,
        const Eigen::Vector3d& support,
        const geometry3d::NurbsSurfaceIntersectionResult3D& intersection,
        RestrictOwnerQueryPath3D path) const;
    RestrictOwnerPreprocessResult3D full_reference_result(
        int target_dof,
        const Eigen::Vector3d& query,
        const Eigen::Vector3d& support,
        RestrictOwnerQueryPath3D path);
    RestrictOwnerPreprocessResult3D optimized_intersection_result(
        int target_dof,
        const Eigen::Vector3d& query,
        const Eigen::Vector3d& support);

    const NativeNurbsSurface3D& surface_;
    const SurfaceDofCloud3D& cloud_;
    geometry3d::NurbsSurfaceIntersector3D intersector_;
    RestrictOwnerPreprocessMode3D mode_;
    RestrictOwnerPreprocessOptions3D options_;
    RestrictOwnerPreprocessDiagnostics3D diagnostics_;
};

} // namespace kfbim::app3d
