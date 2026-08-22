#pragma once

#include "nurbs_surface_intersector_3d.hpp"
#include "../grid/cartesian_grid_3d.hpp"

#include <array>
#include <cstddef>
#include <functional>
#include <limits>
#include <memory>
#include <vector>

namespace kfbim::geometry3d {

class NurbsCartesianDomain3D;

enum class NurbsCartesianPreprocessStrategy3D {
    CertifiedBaseline,
    OptimizedIntersection,
    Hybrid
};

struct GridEdgeEventId3D {
    int first_node = -1;
    int second_node = -1;
    int ordinal = -1;
};

inline bool operator==(const GridEdgeEventId3D& first,
                       const GridEdgeEventId3D& second) noexcept
{
    return first.first_node == second.first_node
        && first.second_node == second.second_node
        && first.ordinal == second.ordinal;
}

inline bool operator!=(const GridEdgeEventId3D& first,
                       const GridEdgeEventId3D& second) noexcept
{
    return !(first == second);
}

enum class GridEdgeEventCertification3D {
    CertifiedTransverse,
    IncompleteRootSet,
    UnknownParity,
    NearTangentEdge,
    FeatureContact,
    UnreliableTransversality,
};

// Canonical, direction-independent physical event on an undirected Cartesian
// grid edge.  first_node < second_node and ordinal follows increasing
// canonical_parameter.  canonical_sign is sign(n dot (x_second-x_first)); a
// directional view flips both parameter order and sign when queried from the
// other endpoint.
struct GridEdgeEvent3D {
    GridEdgeEventId3D id;
    double canonical_parameter = 0.0;
    int canonical_sign = 0;
    int component = -1;
    Eigen::Vector3d point = Eigen::Vector3d::Zero();
    Eigen::Vector3d normal = Eigen::Vector3d::Zero();
    double residual = 0.0;
    double transversality = 0.0;
    bool feature_edge_contact = false;
    GridEdgeEventCertification3D certification =
        GridEdgeEventCertification3D::IncompleteRootSet;
    std::vector<NurbsSurfaceRootOwner3D> owners;

    bool certified_transverse() const noexcept
    {
        return certification
            == GridEdgeEventCertification3D::CertifiedTransverse;
    }
};

class GridEdgeEventView3D {
public:
    const GridEdgeEvent3D& canonical_event() const;
    const GridEdgeEventId3D& id() const;
    int first_node() const noexcept;
    int second_node() const noexcept;
    double edge_parameter() const noexcept;
    int sign() const noexcept;
    int component() const noexcept;
    GridEdgeEventCertification3D certification() const noexcept;
    bool certified_transverse() const noexcept;
    const std::vector<NurbsSurfaceRootOwner3D>& owners() const;

private:
    friend class NurbsCartesianDomain3D;
    friend std::vector<GridEdgeEventView3D> grid_edge_event_views_3d(
        const std::vector<GridEdgeEvent3D>&, int, int);

    GridEdgeEventView3D(const GridEdgeEvent3D* event,
                        int first_node,
                        int second_node,
                        bool reversed) noexcept;

    const GridEdgeEvent3D* event_ = nullptr;
    int first_node_ = -1;
    int second_node_ = -1;
    bool reversed_ = false;
};

class NurbsSurfaceCrossingRange3D {
public:
    NurbsSurfaceCrossingRange3D() noexcept = default;

    const NurbsSurfaceCrossing3D* begin() const noexcept;
    const NurbsSurfaceCrossing3D* end() const noexcept;
    std::size_t size() const noexcept;
    const NurbsSurfaceCrossing3D& operator[](std::size_t index) const;

private:
    friend class NurbsCartesianDomain3D;
    NurbsSurfaceCrossingRange3D(
        const NurbsSurfaceCrossing3D* data, std::size_t size) noexcept;

    const NurbsSurfaceCrossing3D* data_ = nullptr;
    std::size_t size_ = 0;
};

struct NurbsCartesianDomainOptions3D {
    bool use_triangle_seeds = true;
    double maximum_element_extent_cap =
        std::numeric_limits<double>::infinity();
    NurbsCartesianPreprocessStrategy3D strategy =
        NurbsCartesianPreprocessStrategy3D::CertifiedBaseline;
};
struct NurbsCartesianEdgeClassification3D {
    bool queried = false;
    bool has_confirmed_interface = false;
    bool changes_component_membership = false;
    bool root_count_known = true;
    bool parity_known_from_roots = true;
    bool has_near_tangent_candidate = false;
    bool used_targeted_retry = false;
    // True only when the final complete root set has one certified oriented
    // physical event for every retained crossing.  Unlike correction_safe,
    // this also describes same-label and multi-event Cartesian edges.
    bool physical_events_certified = false;
    // Legacy unique-crossing query safety.  The all-event PDE path uses
    // physical_events_certified and the event catalog instead.
    bool correction_safe = false;
    std::size_t confirmed_crossing_count = 0;
    std::size_t ambiguous_cluster_count = 0;
    int confirmed_transverse_count = 0;
};

// Exact component-membership query used only when a feature/multi-owner
// crossing cannot be oriented consistently from its incident normals.  The
// second argument is NurbsSurfaceCrossing3D::component.
using GridEdgeComponentContainment3D =
    std::function<bool(const Eigen::Vector3d&, int)>;

// A candidate edge is retried whenever its complete discrete root set has not
// been certified, independently of the labels at its endpoints.  In
// particular, an unknown zero-root result and an ambiguous same-label double
// crossing both require a retry.
bool grid_edge_root_set_requires_targeted_retry_3d(
    const NurbsCartesianEdgeIntersections3D& intersections) noexcept;

// Pure canonicalization helper used by NurbsCartesianDomain3D and focused
// event tests.  Crossings are interpreted in the supplied node_a -> node_b
// direction and are returned in the canonical first_node < second_node order.
std::vector<GridEdgeEvent3D> build_canonical_grid_edge_events_3d(
    int node_a,
    int node_b,
    const Eigen::Vector3d& point_a,
    const Eigen::Vector3d& point_b,
    const std::vector<NurbsSurfaceCrossing3D>& crossings,
    const NurbsCartesianEdgeClassification3D& classification,
    const GridEdgeComponentContainment3D& component_contains = {});

// Produces query-oriented views for one canonical edge.  Event ids stay
// invariant; a reverse query returns increasing query parameters and negated
// signs.
std::vector<GridEdgeEventView3D> grid_edge_event_views_3d(
    const std::vector<GridEdgeEvent3D>& canonical_events,
    int node_a,
    int node_b);

struct NurbsCartesianDomainDiagnostics3D {
    int nurbs_patch_count = 0;
    int bezier_element_count = 0;
    int acceleration_leaf_count = 0;
    std::size_t candidate_grid_edge_count = 0;
    std::size_t candidate_element_incidence_count = 0;
    double maximum_query_element_extent = 0.0;
    std::array<std::size_t, 3> barrier_edge_counts{{0, 0, 0}};
    std::array<std::size_t, 3> interface_edge_counts{{0, 0, 0}};
    std::size_t multi_crossing_edge_count = 0;
    std::size_t even_parity_interface_edge_count = 0;
    std::size_t odd_parity_interface_edge_count = 0;
    std::size_t ambiguous_parity_edge_count = 0;
    std::size_t ambiguous_label_changing_edge_count = 0;
    std::size_t targeted_retry_count = 0;
    std::size_t targeted_retry_resolved_count = 0;
    std::size_t targeted_retry_unsafe_count = 0;
    std::size_t correction_safe_edge_count = 0;
    std::size_t unsafe_label_changing_edge_count = 0;
    // Candidate Cartesian edges whose final root set/parity/transversality is
    // not certified.  This includes the zero-confirmed-root case, which has
    // no event object through which downstream setup could otherwise fail.
    std::size_t uncertified_grid_edge_count = 0;
    std::size_t canonical_grid_edge_event_count = 0;
    std::size_t certified_transverse_event_count = 0;
    std::size_t endpoint_parity_fallback_count = 0;
    std::size_t endpoint_classification_query_count = 0;
    std::size_t component_parity_toggle_count = 0;
    int grid_component_count = 0;
    int box_exterior_component_count = 0;
    int representative_query_count = 0;
    std::vector<std::size_t> component_sizes;
    NurbsSurfaceIntersectionDiagnostics3D intersections;
    NurbsSurfaceIntersectionDiagnostics3D targeted_retry_intersections;
    double maximum_root_residual = 0.0;
    double intersector_build_seconds = 0.0;
    double candidate_enumeration_seconds = 0.0;
    double edge_intersection_seconds = 0.0;
    double edge_materialization_seconds = 0.0;
    double flood_labeling_seconds = 0.0;
    double representative_classification_seconds = 0.0;
    double invariant_verification_seconds = 0.0;
    double total_construction_seconds = 0.0;
};

class NurbsCartesianDomain3D {
public:
    NurbsCartesianDomain3D(
        const CartesianGrid3D& grid,
        NurbsSurfaceModel3D model,
        NurbsCartesianDomainOptions3D options = {});

    int label(int node) const;
    bool has_barrier_between(int node_a, int node_b) const;
    bool has_interface_between(int node_a, int node_b) const;
    NurbsSurfaceCrossingRange3D crossings_between(
        int node_a, int node_b) const;
    std::vector<GridEdgeEventView3D> grid_edge_events_between(
        int node_a, int node_b) const;
    const std::vector<GridEdgeEvent3D>& grid_edge_event_catalog() const;
    const NurbsSurfaceCrossing3D& crossing_between(
        int node_a, int node_b) const;
    NurbsCartesianEdgeClassification3D edge_classification_between(
        int node_a, int node_b) const;
    const NurbsSurfaceCrossing3D& correction_crossing_between(
        int node_a, int node_b) const;
    const std::vector<int>& labels() const;
    const NurbsCartesianDomainDiagnostics3D& diagnostics() const;
    const NurbsAabb3D& surface_bounds() const;
    double geometry_tolerance() const;
    // Canonical transitive G1 component for every NURBS patch.  Component
    // identifiers are the minimum patch index in each smooth sheet and are
    // therefore deterministic under connection ordering.
    const std::vector<int>& patch_g1_components() const;
    bool is_compatible_grid(const CartesianGrid3D& grid) const noexcept;

private:
    struct Impl;
    std::shared_ptr<const Impl> impl_;
};

} // namespace kfbim::geometry3d
