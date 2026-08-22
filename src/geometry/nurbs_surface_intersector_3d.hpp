#pragma once

#include "nurbs_bezier_intersection_3d.hpp"
#include "nurbs_surface_model_3d.hpp"
#include "rational_bezier_element_3d.hpp"

#include <Eigen/Dense>

#include <array>
#include <cstddef>
#include <limits>
#include <vector>

namespace kfbim::geometry3d {

class NurbsSurfaceIntersector3D;

class NurbsSurfaceCandidate3D {
public:
    int patch_index() const noexcept;
    int component() const noexcept;
    const NurbsAabb3D& bounds() const noexcept;

private:
    friend class NurbsSurfaceIntersector3D;
    NurbsSurfaceCandidate3D(
        const NurbsSurfaceIntersector3D* source,
        std::size_t query_element,
        int patch_index,
        int component,
        NurbsAabb3D bounds);

    const NurbsSurfaceIntersector3D* source_ = nullptr;
    std::size_t query_element_ = 0;
    int patch_index_ = -1;
    int component_ = -1;
    NurbsAabb3D bounds_;
};

// One parametric owner of a physical line/surface root.  A root on a
// declared patch seam can have more than one owner even though it is one
// geometric event.  The legacy scalar fields on NurbsSurfaceCrossing3D keep
// the deterministic representative used by existing callers; owners retains
// the complete canonical owner set for event-aware consumers.
struct NurbsSurfaceRootOwner3D {
    int patch_index = -1;
    double u = 0.0;
    double v = 0.0;
    Eigen::Vector3d point = Eigen::Vector3d::Zero();
    Eigen::Vector3d normal = Eigen::Vector3d::Zero();
    double residual = 0.0;
    double transversality = 0.0;
    bool feature_edge_contact = false;
    double reliable_transversality_tolerance = 0.0;
};

struct NurbsSurfaceCrossing3D {
    int patch_index = -1;
    int component = -1;
    double u = 0.0;
    double v = 0.0;
    double edge_parameter = 0.0;
    Eigen::Vector3d point = Eigen::Vector3d::Zero();
    Eigen::Vector3d normal = Eigen::Vector3d::Zero();
    double residual = 0.0;
    double transversality = 0.0;
    bool feature_edge_contact = false;
    double reliable_transversality_tolerance = 0.0;
    std::vector<NurbsSurfaceRootOwner3D> owners;
};

struct NurbsSurfaceIntersectionDiagnostics3D {
    int candidate_elements = 0;
    int bvh_candidate_elements = 0;
    int mapped_candidate_elements = 0;
    int maximum_candidate_elements_per_edge = 0;
    int triangle_seed_hits = 0;
    int triangle_seed_misses_recovered = 0;
    int subdivision_boxes = 0;
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
    int same_patch_deduplications = 0;
    int seam_deduplications = 0;
    int unresolved_candidates = 0;
    int maximum_subdivision_depth_reached = 0;
    int terminal_certificate_boxes = 0;
    int maximum_terminal_certificate_depth_reached = 0;
    int closest_point_attempts = 0;
    int closest_point_iterations = 0;
    int roots_recovered_by_closest_point = 0;
    int terminal_misses_by_closest_point = 0;
    int closest_point_failures = 0;
    int sample_seed_candidates = 0;
    int sample_seeds_accepted = 0;
    int roots_recovered_by_sample_seed = 0;
    int maximum_sample_seeds_per_element = 0;
    int stationary_solve_attempts = 0;
    int stationary_solve_converged = 0;
    int stationary_witnesses = 0;
    int root_pairs_protected_by_stationary_witness = 0;
    int ambiguous_root_clusters = 0;
    int non_g1_topology_merges = 0;
    int high_degree_fallbacks = 0;
    std::vector<std::array<double, 2>> unresolved_longitudinal_intervals;
};

struct NurbsSurfaceIntersectionResult3D {
    std::vector<NurbsSurfaceCrossing3D> crossings;
    NurbsSurfaceIntersectionDiagnostics3D diagnostics;
    bool overlap_detected = false;
};

struct NurbsSurfaceCandidateCertificate3D {
    NurbsElementSegmentCertificateKind3D kind =
        NurbsElementSegmentCertificateKind3D::Unresolved;
    std::optional<NurbsSurfaceCrossing3D> crossing;
    NurbsElementIntersectionDiagnostics3D diagnostics;
    bool overlap_detected = false;
    bool segment_parameter_strictly_interior = false;
    bool element_parameter_strictly_interior = false;
    bool patch_parameter_strictly_interior = false;
};

struct NurbsSurfaceFilteredIntersectionOptions3D {
    int maximum_independent_crossings = 2;
};

struct NurbsSurfaceFilteredIntersectionResult3D {
    NurbsSurfaceIntersectionResult3D intersection;
    bool all_candidates_processed = true;
    bool independent_crossing_limit_reached = false;
};

struct NurbsCartesianEdgeQuery3D {
    int axis = -1;
    int i = -1;
    int j = -1;
    int k = -1;
    Eigen::Vector3d start = Eigen::Vector3d::Zero();
    Eigen::Vector3d end = Eigen::Vector3d::Zero();
};

struct AmbiguousRootCluster3D {
    int component = -1;
    double edge_parameter_begin = 0.0;
    double edge_parameter_end = 1.0;
    std::vector<NurbsSurfaceCrossing3D> candidates;
};

struct NurbsCartesianEdgeIntersections3D {
    std::vector<NurbsSurfaceCrossing3D> crossings;
    std::vector<AmbiguousRootCluster3D> ambiguous_clusters;
    std::vector<int> toggled_components;
    NurbsSurfaceIntersectionDiagnostics3D diagnostics;
    int confirmed_transverse_count = 0;
    bool root_count_known = true;
    bool parity_known_from_roots = true;
    bool has_near_tangent_candidate = false;
    bool changes_inside_outside = false;
    bool changes_component_membership = false;
};

struct NurbsQueryElementSample3D {
    double u = 0.0;
    double v = 0.0;
    Eigen::Vector3d point = Eigen::Vector3d::Zero();
};

struct NurbsQueryElementDescriptor3D {
    std::size_t id = 0;
    NurbsAabb3D bounds;
    int patch_index = -1;
    int component = -1;
};

enum class NurbsCartesianEdgeQueryRoute3D {
    Configured,
    OptimizedCertified
};

struct NurbsCartesianEdgeQueryOptions3D {
    int local_max_subdivision_depth = -1;
    NurbsCartesianEdgeQueryRoute3D route =
        NurbsCartesianEdgeQueryRoute3D::Configured;
};

struct NurbsSurfaceIntersectorOptions3D {
    bool use_triangle_seeds = true;
    // Preserve every declared non-G1 patch owner of a coincident physical
    // root.  The default keeps the historical canonical one-root behavior;
    // one-sided C0 restrict routes opt in so they can choose a G1 sheet using
    // their own geometric criterion.
    bool preserve_non_g1_root_owners = false;
    // Continue across all candidate elements after an unresolved terminal
    // box and return the aggregate certified roots plus conservative
    // longitudinal intervals.  The default preserves the historical throw.
    bool collect_unresolved_regions = false;
    bool use_early_unique_root_certificate = false;
    bool use_affine_planar_fast_path = false;
    bool use_closest_point_prefilter = false;
    int bvh_leaf_size = 8;
    double maximum_element_extent =
        std::numeric_limits<double>::infinity();
    int local_max_subdivision_depth = 4;
    int terminal_separation_subdivision_depth =
        kDefaultTerminalSeparationSubdivisionDepth3D;
};

class NurbsSurfaceIntersector3D {
public:
    explicit NurbsSurfaceIntersector3D(
        NurbsSurfaceModel3D model,
        NurbsSurfaceIntersectorOptions3D options = {});

    const NurbsSurfaceModel3D& model() const;
    const NurbsAabb3D& bounds() const;
    double geometry_tolerance() const;
    std::size_t query_element_count() const noexcept;
    const std::vector<NurbsQueryElementDescriptor3D>&
    query_elements() const noexcept;
    double maximum_query_element_extent() const noexcept;
    const std::array<NurbsQueryElementSample3D, 16>&
    query_element_samples(std::size_t element) const;

    std::vector<NurbsSurfaceCandidate3D> conservative_candidates(
        const NurbsAabb3D& query_bounds) const;
    std::vector<NurbsSurfaceCandidate3D> conservative_segment_candidates(
        const Eigen::Vector3d& start,
        const Eigen::Vector3d& end) const;
    NurbsSurfaceCandidateCertificate3D certify_candidate_segment(
        const NurbsSurfaceCandidate3D& candidate,
        const Eigen::Vector3d& start,
        const Eigen::Vector3d& end) const;
    NurbsSurfaceFilteredIntersectionResult3D intersect_segment_candidates(
        const Eigen::Vector3d& start,
        const Eigen::Vector3d& end,
        const std::vector<NurbsSurfaceCandidate3D>& ordered_candidates,
        const NurbsSurfaceFilteredIntersectionOptions3D& options = {}) const;

    NurbsSurfaceIntersectionResult3D intersect_segment(
        const Eigen::Vector3d& start,
        const Eigen::Vector3d& end) const;
    NurbsCartesianEdgeIntersections3D intersect_cartesian_edge(
        const NurbsCartesianEdgeQuery3D& edge) const;
    NurbsCartesianEdgeIntersections3D intersect_cartesian_edge(
        const NurbsCartesianEdgeQuery3D& edge,
        const std::vector<std::size_t>& candidate_element_ids,
        NurbsCartesianEdgeQueryOptions3D options = {}) const;

    std::vector<int> containing_components(
        const Eigen::Vector3d& point) const;
    std::vector<RationalBezierElement3D> acceleration_leaves(
        double maximum_extent) const;

private:
    struct BvhNode {
        NurbsAabb3D bounds;
        int left = -1;
        int right = -1;
        int begin = 0;
        int end = 0;
    };

    int build_bvh_node(int begin, int end);
    void collect_candidate_elements(
        int node,
        const NurbsAabb3D& segment_bounds,
        std::vector<int>& candidates) const;
    void validate_segment(
        const Eigen::Vector3d& start,
        const Eigen::Vector3d& end) const;
    void validate_candidate_owner(
        const NurbsSurfaceCandidate3D& candidate) const;
    NurbsSurfaceIntersectionResult3D intersect_segment_impl(
        const Eigen::Vector3d& start,
        const Eigen::Vector3d& end,
        const NurbsCartesianEdgeQuery3D* cartesian_edge,
        const std::vector<int>* mapped_candidates,
        int local_max_subdivision_depth,
        NurbsCartesianEdgeQueryRoute3D route) const;
    NurbsSurfaceIntersectionResult3D intersect_segment_candidate_indices(
        const Eigen::Vector3d& start,
        const Eigen::Vector3d& end,
        const std::vector<std::size_t>& candidates,
        const NurbsCartesianEdgeQuery3D* cartesian_edge,
        int maximum_independent_crossings,
        bool* all_candidates_processed,
        bool* independent_crossing_limit_reached,
        int local_max_subdivision_depth,
        NurbsCartesianEdgeQueryRoute3D route,
        bool provided_candidates) const;
    NurbsCartesianEdgeIntersections3D intersect_cartesian_edge_impl(
        const NurbsCartesianEdgeQuery3D& edge,
        const std::vector<int>* mapped_candidates,
        NurbsCartesianEdgeQueryOptions3D options) const;

    NurbsSurfaceModel3D model_;
    NurbsSurfaceIntersectorOptions3D options_;
    NurbsAabb3D bounds_;
    double geometry_tolerance_ = 1e-14;
    double maximum_query_element_extent_ = 0.0;
    std::vector<RationalBezierElement3D> elements_;
    std::vector<bool> element_touches_non_g1_feature_;
    std::vector<NurbsQueryElementDescriptor3D> query_elements_;
    std::vector<std::array<NurbsQueryElementSample3D, 16>> element_samples_;
    std::vector<int> element_order_;
    std::vector<BvhNode> bvh_nodes_;
    int bvh_root_ = -1;
};

} // namespace kfbim::geometry3d
