#pragma once

#include <Eigen/Dense>
#include <memory>
#include <vector>
#include "../grid/cartesian_grid_3d.hpp"
#include "../interface/interface_3d.hpp"
#include "p2_projection_3d.hpp"

namespace kfbim {

namespace geometry3d {
class NurbsCartesianDomain3D;
struct NurbsCartesianDomainDiagnostics3D;
struct GridEdgeEventId3D;
struct GridEdgeEvent3D;
} // namespace geometry3d

enum class P2CrossingOwnerStatus3D {
    ExactIntersection,
    GapFallback,
    EndpointNearestCenter
};

struct P2CrossingOwner3D {
    int center_index = -1;
    // Correction/collocation panel owning center_index.
    int panel_index = -1;
    double edge_parameter = 0.5;
    // Barycentric coordinate on panel_index when geometry and correction
    // surfaces coincide; otherwise this is the selected center coordinate.
    Eigen::Vector3d barycentric = Eigen::Vector3d::Zero();
    // Approximate hit on the complete flat corner-triangle geometry. These
    // legacy fields remain unset for native NURBS owners and endpoint-only
    // center selection.
    int geometry_panel_index = -1;
    Eigen::Vector3d geometry_barycentric = Eigen::Vector3d::Zero();
    int nurbs_patch_index = -1;
    Eigen::Vector2d nurbs_parameter = Eigen::Vector2d::Zero();
    Eigen::Vector3d crossing_point = Eigen::Vector3d::Zero();
    Eigen::Vector3d crossing_normal = Eigen::Vector3d::Zero();
    int surface_component = -1;
    double crossing_residual = 0.0;
    P2CrossingOwnerStatus3D status =
        P2CrossingOwnerStatus3D::EndpointNearestCenter;
};

// One P2 expansion-center candidate with exact source-patch metadata.  Native
// event ownership uses this metadata to prevent a C0-adjacent center and a
// NURBS root owner from being combined into one correction jet.
struct P2NativeSheetCenterCandidate3D {
    int center_index = -1;
    int panel_index = -1;
    int nurbs_patch_index = -1;
    Eigen::Vector3d point = Eigen::Vector3d::Zero();
};

struct P2NativeEventOwnerCenterSelection3D {
    int owner_index = -1;
    int center_candidate_index = -1;
};

// Select the closest legal owner/center pair.  A pair is legal only when the
// event owner and the P2 center source patch belong to the same transitive G1
// component.  Distance ties prefer the more transverse owner, then stable
// patch/parameter/center identifiers.
P2NativeEventOwnerCenterSelection3D
select_p2_native_event_owner_center_3d(
    const geometry3d::GridEdgeEvent3D& event,
    const std::vector<int>& patch_g1_components,
    const std::vector<P2NativeSheetCenterCandidate3D>& center_candidates,
    double distance_tie_tolerance);

// Owns the spatial structures relating a CartesianGrid3D and an Interface3D.
// Built once at setup; queries are read-mostly after construction.
class GridPair3D {
public:
    GridPair3D(const CartesianGrid3D& grid, const Interface3D& interface);
    GridPair3D(const CartesianGrid3D& grid,
               const Interface3D& correction_interface,
               const Interface3D& crossing_geometry);
    // A non-null domain must have been built from this exact Cartesian grid.
    // The correction/crossing interfaces and domain must also derive from the
    // same native NURBS surface model.
    GridPair3D(
        const CartesianGrid3D& grid,
        const Interface3D& correction_interface,
        const Interface3D& crossing_geometry,
        std::shared_ptr<const geometry3d::NurbsCartesianDomain3D> nurbs_domain);
    // Strict native-event constructor. correction_panel_nurbs_patches is
    // panel-major and must align one-to-one with correction_interface panels.
    // It lets event ownership constrain expansion centers to the selected
    // root owner's physical G1 sheet.
    GridPair3D(
        const CartesianGrid3D& grid,
        const Interface3D& correction_interface,
        const Interface3D& crossing_geometry,
        std::shared_ptr<const geometry3d::NurbsCartesianDomain3D> nurbs_domain,
        std::vector<int> correction_panel_nurbs_patches);
    ~GridPair3D();
    GridPair3D(const GridPair3D&)            = delete;
    GridPair3D& operator=(const GridPair3D&) = delete;

    // interface point index → nearest bulk node
    int closest_bulk_node(int interface_pt_idx) const;

    // bulk node index → nearest interface point (lazy compatibility query)
    int closest_interface_point(int bulk_node_idx) const;

    // bulk node index → nearest active P2 expansion center. Available only for
    // QuadraticLagrange six-node triangular patches.
    int nearest_p2_expansion_center(int bulk_node_idx) const;
    // Select between the two endpoint-owned nearest P2 expansion centers by
    // minimizing the sum of distances from the center to both endpoints.
    P2CrossingOwner3D p2_crossing_owner_between(int bulk_node_a,
                                                int bulk_node_b) const;
    // Resolve one immutable native NURBS grid-edge event.  Unlike the legacy
    // edge-only query above, this remains unambiguous when a Cartesian edge
    // intersects the surface more than once.
    P2CrossingOwner3D p2_crossing_owner_for_grid_edge_event(
        const geometry3d::GridEdgeEventId3D& event_id) const;
    int nearest_p2_expansion_center_between(int bulk_node_a,
                                             int bulk_node_b) const;
    int nearest_p2_expansion_center_for_interface_point(
        int interface_point) const;

    // domain label: 0 = Ω⁻ (exterior), 1,2,... = Ω⁺ (interior) of each component
    int domain_label(int bulk_node_idx) const;
    bool has_nurbs_domain() const noexcept;
    const geometry3d::NurbsCartesianDomain3D*
    nurbs_cartesian_domain() const noexcept;
    const geometry3d::NurbsCartesianDomainDiagnostics3D&
        nurbs_domain_diagnostics() const;

    bool is_near_interface(int bulk_node_idx, double radius) const;

    // all bulk node indices within radius of active geometry samples
    // (P2 expansion centers for QuadraticLagrange patches)
    std::vector<int> near_interface_nodes(double radius) const;

    // all interface point indices within radius of a given bulk node
    // (may span multiple components; used by Corrector to accumulate all contributions)
    std::vector<int> near_interface_points(int bulk_node_idx, double radius) const;

    // P2 curved-surface projections for all grid nodes in a narrow band.
    // Projection results include the parent panel and barycentric coordinates
    // for later surface interpolation/differentiation.
    NarrowBandProjection3D project_near_interface_nodes(double radius) const;

    // P2 curved-surface projections for an explicit grid-node support set.
    // Uses the nearest generated P2 expansion center to choose a parent panel,
    // initializes with the closest point on the flat vertex triangle, then
    // applies curved-patch Newton. If Newton does not converge, the flat
    // triangle parameter is returned with converged=false.
    NarrowBandProjection3D project_grid_nodes_to_interface(
        const std::vector<int>& bulk_node_indices) const;

    const CartesianGrid3D& grid()      const { return grid_; }
    const Interface3D&     interface() const { return interface_; }
    const Interface3D& crossing_geometry() const { return crossing_geometry_; }

private:
    const CartesianGrid3D& grid_;
    const Interface3D&     interface_;
    const Interface3D&     crossing_geometry_;
    std::shared_ptr<const geometry3d::NurbsCartesianDomain3D> nurbs_domain_;

    // CGAL structures built in constructor (pimpl to avoid leaking CGAL headers)
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

} // namespace kfbim
