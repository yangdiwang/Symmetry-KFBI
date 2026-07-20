#pragma once

#include <Eigen/Dense>

namespace kfbim {

enum class PanelNodeLayout3D {
    Raw,
    QuadraticLagrange
};

// Triangulated surface interface in 3D.
//
// Geometry and DOFs are kept separate:
//   - vertices + panels define the coarse triangle mesh used by legacy/raw
//     callers and as a source mesh for diagnostics
//   - points, normals, weights are the quadrature/DOF locations
//   - panel_point_indices maps each panel-local DOF to a row in points
//
// The legacy constructor builds panel-major point connectivity. The explicit
// constructor supports shared nodes, including QuadraticLagrange P2 triangles
// with local order [v0, v1, v2, e01, e12, e20].
//
// Supports multiple disconnected components via panel_components.
class Interface3D {
public:
    // vertices:         Nv x 3, geometric vertex coordinates
    // panels:           Np x 3, triangle connectivity (indices into vertices)
    // points:           Nq x 3, quadrature/DOF coordinates, panel-major order
    // normals:          Nq x 3, outward unit normals at each quadrature point
    // weights:          Nq,     quadrature weights (area-weighted)
    // points_per_panel: k,      uniform; Np = Nq / k
    // panel_components: Np,     0-indexed component id; all-zeros for single interface
    Interface3D(Eigen::MatrixX3d vertices,
                Eigen::MatrixX3i panels,
                Eigen::MatrixX3d points,
                Eigen::MatrixX3d normals,
                Eigen::VectorXd  weights,
                int              points_per_panel,
                Eigen::VectorXi  panel_components,
                PanelNodeLayout3D panel_node_layout = PanelNodeLayout3D::Raw);

    // Explicit-connectivity constructor with a separate smooth-surface group
    // per panel. panel_components describes closed connected components for
    // inside/outside labeling; panel_smooth_groups distinguishes C1 surface
    // sheets across which one local correction may be continued. At a sharp
    // edge the incident panels have the same component but different smooth
    // groups.
    Interface3D(Eigen::MatrixX3d vertices,
                Eigen::MatrixX3i panels,
                Eigen::MatrixX3d points,
                Eigen::MatrixX3d normals,
                Eigen::VectorXd  weights,
                int              points_per_panel,
                Eigen::MatrixXi  panel_point_indices,
                Eigen::VectorXi  panel_components,
                Eigen::VectorXi  panel_smooth_groups,
                PanelNodeLayout3D panel_node_layout = PanelNodeLayout3D::Raw);

    // Explicit-connectivity constructor. panel_point_indices is Np x k and
    // maps each panel-local point to a row in points/normals/weights.
    Interface3D(Eigen::MatrixX3d vertices,
                Eigen::MatrixX3i panels,
                Eigen::MatrixX3d points,
                Eigen::MatrixX3d normals,
                Eigen::VectorXd  weights,
                int              points_per_panel,
                Eigen::MatrixXi  panel_point_indices,
                Eigen::VectorXi  panel_components,
                PanelNodeLayout3D panel_node_layout = PanelNodeLayout3D::Raw);

    int num_vertices()     const { return static_cast<int>(vertices_.rows()); }
    int num_panels()       const { return static_cast<int>(panels_.rows()); }
    int num_points()       const { return static_cast<int>(points_.rows()); }
    int points_per_panel() const { return points_per_panel_; }
    int num_components()   const { return panel_components_.maxCoeff() + 1; }

    int point_index(int panel, int local_q) const {
        return panel_point_indices_(panel, local_q);
    }

    const Eigen::MatrixX3d& vertices()         const { return vertices_; }
    const Eigen::MatrixX3i& panels()           const { return panels_; }
    const Eigen::MatrixX3d& points()           const { return points_; }
    const Eigen::MatrixX3d& normals()          const { return normals_; }
    const Eigen::VectorXd&  weights()          const { return weights_; }
    const Eigen::MatrixXi&  panel_point_indices() const { return panel_point_indices_; }
    const Eigen::VectorXi&  panel_components() const { return panel_components_; }
    const Eigen::VectorXi&  panel_smooth_groups() const { return panel_smooth_groups_; }
    PanelNodeLayout3D       panel_node_layout() const { return panel_node_layout_; }

private:
    Eigen::MatrixX3d vertices_;
    Eigen::MatrixX3i panels_;
    Eigen::MatrixX3d points_;
    Eigen::MatrixX3d normals_;
    Eigen::VectorXd  weights_;
    int              points_per_panel_;
    Eigen::MatrixXi  panel_point_indices_;
    Eigen::VectorXi  panel_components_;
    Eigen::VectorXi  panel_smooth_groups_;
    PanelNodeLayout3D panel_node_layout_;
};

} // namespace kfbim
