#pragma once

#include "rational_bezier_element_3d.hpp"

#include <Eigen/Dense>
#include <array>
#include <optional>
#include <stdexcept>

#include <vector>

namespace kfbim::geometry3d {

class NurbsPatchPolarEvaluator3D;

struct NurbsElementRoot3D {
    int patch_index = -1;
    int component = -1;
    double u = 0.0;
    double v = 0.0;
    double t = 0.0;
    Eigen::Vector3d point = Eigen::Vector3d::Zero();
    Eigen::Vector3d normal = Eigen::Vector3d::Zero();
    double residual = 0.0;
    double transversality = 0.0;
    double reliable_transversality_tolerance = 0.0;
};

struct NurbsElementParameterSeed3D {
    double u = 0.0;
    double v = 0.0;
    double t = 0.0;
};

struct NurbsElementIntersectionDiagnostics3D {
    int subdivision_boxes = 0;
    int conservative_rejections = 0;
    int triangle_seed_hits = 0;
    int newton_attempts = 0;
    int newton_iterations = 0;
    int early_unique_certificate_attempts = 0;
    int early_unique_certificate_successes = 0;
    int planar_analytic_hits = 0;
    int planar_analytic_misses = 0;
    int planar_analytic_fallbacks = 0;
    int closest_point_prefilter_attempts = 0;
    int closest_point_prefilter_certified_hits = 0;
    int closest_point_prefilter_certified_misses = 0;
    int closest_point_prefilter_fallbacks = 0;
    int certified_fallback_elements = 0;
    int roots_recovered_without_triangle_seed = 0;
    int unresolved_boxes = 0;
    int maximum_subdivision_depth_reached = 0;
    int terminal_certificate_boxes = 0;
    int maximum_terminal_certificate_depth_reached = 0;
    int closest_point_attempts = 0;
    int closest_point_iterations = 0;
    int roots_recovered_by_closest_point = 0;
    int terminal_misses_by_closest_point = 0;
    int closest_point_failures = 0;
    int supplied_seed_attempts = 0;
    int roots_recovered_by_supplied_seed = 0;
    int maximum_supplied_seed_count = 0;
    int high_degree_control_hull_fallbacks = 0;
    // Outward-rounded conservative longitudinal intervals (physical distance
    // from segment start) of terminal boxes whose root content could not be
    // certified.
    // These let callers prove that an unresolved box lies beyond the part of
    // a segment relevant to a nearest-root query.
    std::vector<std::array<double, 2>> unresolved_longitudinal_intervals;
};

struct NurbsElementIntersectionResult3D {
    std::vector<NurbsElementRoot3D> roots;
    NurbsElementIntersectionDiagnostics3D diagnostics;
    bool overlap_detected = false;
};

enum class NurbsElementSegmentCertificateKind3D {
    CertifiedMiss,
    CertifiedUniqueTransverseRoot,
    Unresolved
};

struct NurbsElementSegmentCertificate3D {
    NurbsElementSegmentCertificateKind3D kind =
        NurbsElementSegmentCertificateKind3D::Unresolved;
    std::optional<NurbsElementRoot3D> root;
    NurbsElementIntersectionDiagnostics3D diagnostics;
    bool overlap_detected = false;
};

class UnresolvedNurbsIntersectionCandidate3D : public std::runtime_error {
public:
    explicit UnresolvedNurbsIntersectionCandidate3D(
        NurbsElementIntersectionResult3D partial_result);

    const NurbsElementIntersectionResult3D& partial_result() const noexcept;

private:
    NurbsElementIntersectionResult3D partial_result_;
};

inline constexpr int kDefaultTerminalSeparationSubdivisionDepth3D = 6;
inline constexpr int kMaximumTerminalSeparationSubdivisionDepth3D = 16;

struct NurbsElementIntersectionOptions3D {
    double geometry_tolerance = 1e-12;
    double parameter_tolerance = 1e-12;
    int max_subdivision_depth = 4;
    int terminal_separation_subdivision_depth =
        kDefaultTerminalSeparationSubdivisionDepth3D;
    int max_newton_iterations = 24;
    bool use_triangle_seed = true;
    bool use_early_unique_root_certificate = false;
    bool use_affine_planar_fast_path = false;
    bool use_closest_point_prefilter = false;
    std::vector<NurbsElementParameterSeed3D> parameter_seeds;
    // Optional immutable evaluator for this patch. Must outlive this query.
    const NurbsPatchPolarEvaluator3D* surface_evaluator = nullptr;
};

NurbsElementSegmentCertificate3D certify_nurbs_bezier_element_segment_3d(
    const RationalBezierElement3D& element,
    const NurbsSurfacePatch3D& patch,
    const Eigen::Vector3d& segment_start,
    const Eigen::Vector3d& segment_end,
    const NurbsElementIntersectionOptions3D& options,
    std::optional<Eigen::Vector2d> preferred_seed = std::nullopt);

NurbsElementIntersectionResult3D intersect_nurbs_bezier_element_3d(
    const RationalBezierElement3D& element,
    const NurbsSurfacePatch3D& patch,
    const Eigen::Vector3d& segment_start,
    const Eigen::Vector3d& segment_end,
    const NurbsElementIntersectionOptions3D& options);

} // namespace kfbim::geometry3d
