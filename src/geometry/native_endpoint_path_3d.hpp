#pragma once

#include "nurbs_surface_intersector_3d.hpp"

#include <cstdint>
#include <optional>
#include <string>
#include <vector>

namespace kfbim::geometry3d {

struct ExactNativeBezierElement3D;

enum class RootProofKind3D { Candidate, ExistsUnique, KnownEndpointUnique };
enum class RootRelation3D { SameRoot, DistinctRoots, Unresolved };
enum class PathStatus3D {
    Certified, Unresolved, BudgetExceeded, UnsupportedEndpoint,
    DegenerateSegment, InvalidInput
};

struct RootInterval3D { double lower = 0.0; double upper = 0.0; };
struct RootParameterBox3D { RootInterval3D u; RootInterval3D v; };

struct NativeSurfaceEndpoint3D {
    int patch_index = -1;
    double u = 0.0;
    double v = 0.0;
};

struct RootCertificate3D {
    RootProofKind3D kind = RootProofKind3D::Candidate;
    std::uint64_t proof_id = 0;
    std::uint64_t event_id = 0;
    int patch_index = -1;
    std::size_t source_element = 0;
    RootParameterBox3D uniqueness_box;
    RootParameterBox3D root_enclosure;
    RootInterval3D t_interval;
    double contraction_bound = 0.0;
    unsigned precision_bits = 53;
    bool existence_proved = false;
    bool uniqueness_proved = false;
};

struct NativeRootTransition3D {
    int sign = 0; // +1 enters, -1 leaves; never inferred from desired side.
    int incoming_inside = -1;
    int outgoing_inside = -1;
    double oriented_dot_lower = 0.0;
    double oriented_dot_upper = 0.0;
    bool certified = false;
};

struct NativeRootAccuracy3D {
    double point_error_bound = 0.0;
    double parameter_error_bound = 0.0;
    double normal_error_bound = 0.0;
    bool certified = false;
};

struct CertifiedNativePathRoot3D {
    NurbsSurfaceCrossing3D representative;
    RootCertificate3D certificate;
    NativeRootTransition3D transition;
    NativeRootAccuracy3D accuracy;
};

struct PathCertificationBudget3D {
    std::uint64_t max_candidate_regions = 4096;
    std::uint64_t max_subdivision_nodes = 32768;
    unsigned max_subdivision_depth = 60;
    std::uint64_t max_newton_steps = 16384;
    std::uint64_t max_closest_point_evaluations = 0;
    std::uint64_t max_relation_refinements = 1024;
    unsigned max_precision_escalations = 64;
    unsigned max_precision_bits = 512;
};

struct NativeEndpointQueryOptions3D {
    double point_tolerance = 2.0e-12;
    double parameter_tolerance = 2.0e-12;
    double normal_tolerance = 2.0e-10;
    unsigned initial_precision_bits = 53;
    bool enable_precision_escalation = true;
    // The first implementation is a closed native path; extension is explicit
    // and currently unsupported, rather than silently changing query geometry.
    double endpoint_extension_ratio = 0.0;
    std::optional<bool> expected_start_inside;
};

struct PathCoverage3D {
    std::uint64_t candidate_regions = 0;
    std::uint64_t visited_regions = 0;
    std::uint64_t excluded_regions = 0;
    std::uint64_t root_regions = 0;
    std::uint64_t split_regions = 0;
    std::uint64_t unresolved_regions = 0;
    bool complete = false;
    bool order_certified = false;
    bool transitions_certified = false;
    bool representatives_certified = false;
    int certified_start_inside = -1;
};

struct PathCertificationDiagnostics3D {
    std::uint64_t newton_steps = 0;
    std::uint64_t closest_point_evaluations = 0;
    std::uint64_t numerical_candidates = 0;
    std::uint64_t known_endpoint_proofs = 0;
    std::uint64_t ordinary_root_proofs = 0;
    std::uint64_t relation_refinements = 0;
    std::uint64_t same_root_merges = 0;
    unsigned precision_escalations = 0;
    unsigned highest_precision_bits = 53;
    unsigned maximum_depth = 0;
    std::string reason;
};

struct NativeEndpointPathResult3D {
    PathStatus3D status = PathStatus3D::Unresolved;
    std::vector<CertifiedNativePathRoot3D> open_roots;
    std::optional<CertifiedNativePathRoot3D> endpoint;
    std::vector<CertifiedNativePathRoot3D> post_endpoint_roots;
    PathCoverage3D coverage;
    PathCertificationDiagnostics3D diagnostics;
    std::string message;
    // Round-trip numerical metadata and exact source/query fingerprints are
    // included in this bounded, deterministic replay/audit payload.
    std::string dump;
};

NativeEndpointPathResult3D certify_native_endpoint_path_3d(
    const NurbsSurfaceModel3D& model,
    const std::vector<ExactNativeBezierElement3D>& exact_geometry,
    const Eigen::Vector3d& start,
    const NativeSurfaceEndpoint3D& endpoint,
    const NativeEndpointQueryOptions3D& options = {},
    const PathCertificationBudget3D& budget = {},
    const std::vector<NurbsAabb3D>* certified_bounds = nullptr);

const char* native_endpoint_path_status_name_3d(PathStatus3D status) noexcept;

} // namespace kfbim::geometry3d
