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
    int nearest_p2_expansion_center_between(int bulk_node_a,
                                             int bulk_node_b) const;
    int nearest_p2_expansion_center_for_interface_point(
        int interface_point) const;

    // domain label: 0 = Ω⁻ (exterior), 1,2,... = Ω⁺ (interior) of each component
    int domain_label(int bulk_node_idx) const;
    bool has_nurbs_domain() const noexcept;
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
