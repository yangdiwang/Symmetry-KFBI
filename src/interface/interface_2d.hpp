#pragma once

#include <Eigen/Dense>
#include <vector>

namespace kfbim {

enum class PanelNodeLayout2D {
    LegacyGaussLegendre,
    QuadraticLagrange,
    Raw,

    // Backward-compatible alias for older active 2D call sites.
    ChebyshevLobatto = QuadraticLagrange,

    // Backward-compatible alias for older call sites.
    GaussLegendre = LegacyGaussLegendre
};

enum class InterfacePointKind2D {
    Smooth,
    Corner
};

struct CornerData2D {
    int point = -1;
    int prev_panel = -1;
    int next_panel = -1;

    Eigen::Vector2d tangent_minus = Eigen::Vector2d::Zero();
    Eigen::Vector2d tangent_plus = Eigen::Vector2d::Zero();
    Eigen::Vector2d normal_minus = Eigen::Vector2d::Zero();
    Eigen::Vector2d normal_plus = Eigen::Vector2d::Zero();

    // Signed angle from tangent_minus to tangent_plus, in radians.
    double turn_angle = 0.0;
};

enum class CornerPatchSide2D {
    Interior,
    Exterior
};

enum class CornerPatchShape2D {
    Sector,
    Circle
};

struct CornerPatch2D {
    int corner = -1;
    int point = -1;
    CornerPatchSide2D side = CornerPatchSide2D::Interior;
    CornerPatchShape2D shape = CornerPatchShape2D::Sector;
    double opening_angle = 0.0;
    double radius = 0.0;
    Eigen::Vector2d center = Eigen::Vector2d::Zero();
    Eigen::Vector2d ray_minus = Eigen::Vector2d::Zero();
    Eigen::Vector2d ray_plus = Eigen::Vector2d::Zero();
};

struct PanelSideGeometry2D {
    // Flattened as rows = num_panels * points_per_panel, row index
    // panel * points_per_panel + local_q.
    Eigen::MatrixX2d point_normals;
    Eigen::MatrixX2d point_tangents;
};

// Isoparametric piecewise-polynomial curve interface in 2D.
//
// The interface is divided into Np panels. Each panel has k = points_per_panel
// quadratic Lagrange, legacy Gauss-Legendre, or raw points that serve as both
// the geometric nodes (defining a degree k-1 polynomial curve segment) and the
// DOF/quadrature locations.
//
// Panels reference point rows through panel_point_indices. The legacy
// constructor builds panel-major connectivity, while quadratic Lagrange panels
// may share endpoint point rows between adjacent panels.
//
// Supports multiple disconnected components (e.g. several circles) via
// panel_components: one integer component id per panel.
class Interface2D {
public:
    // points:           Nq x 2, all quadrature/DOF coordinates
    // normals:          Nq x 2, outward unit normals at each point
    // weights:          Nq,     quadrature weights (arc-length-weighted)
    // points_per_panel: k,      uniform across all panels; Np inferred from Nq/k here
    // panel_components: Np,     0-indexed component id; all-zeros for single interface
    Interface2D(Eigen::MatrixX2d points,
                Eigen::MatrixX2d normals,
                Eigen::VectorXd  weights,
                int              points_per_panel,
                Eigen::VectorXi  panel_components,
                PanelNodeLayout2D panel_node_layout = PanelNodeLayout2D::LegacyGaussLegendre);

    // Explicit-connectivity constructor. panel_point_indices is Np x k and
    // maps each panel-local point to a row in points/normals/weights.
    Interface2D(Eigen::MatrixX2d points,
                Eigen::MatrixX2d normals,
                Eigen::VectorXd  weights,
                int              points_per_panel,
                Eigen::MatrixXi  panel_point_indices,
                Eigen::VectorXi  panel_components,
                PanelNodeLayout2D panel_node_layout = PanelNodeLayout2D::LegacyGaussLegendre);

    // Explicit panel-side geometry and corner metadata constructor.
    // panel_side_geometry stores per-panel-side limiting tangent/normal data;
    // the legacy point normals remain available through normals().
    Interface2D(Eigen::MatrixX2d points,
                Eigen::MatrixX2d normals,
                Eigen::VectorXd  weights,
                int              points_per_panel,
                Eigen::MatrixXi  panel_point_indices,
                Eigen::VectorXi  panel_components,
                PanelSideGeometry2D panel_side_geometry,
                std::vector<InterfacePointKind2D> point_kind,
                std::vector<int> corner_index_by_point,
                std::vector<CornerData2D> corners,
                PanelNodeLayout2D panel_node_layout = PanelNodeLayout2D::LegacyGaussLegendre,
                std::vector<CornerPatch2D> corner_patches = {});

    int num_points()       const { return static_cast<int>(points_.rows()); }
    int points_per_panel() const { return points_per_panel_; }
    int num_panels()       const { return static_cast<int>(panel_point_indices_.rows()); }
    int num_components()   const { return panel_components_.maxCoeff() + 1; }

    // global index of the q-th local point on panel p
    int point_index(int panel, int local_q) const {
        return panel_point_indices_(panel, local_q);
    }

    Eigen::Vector2d panel_normal(int panel, int local_q) const {
        return panel_normals_.row(panel_local_row(panel, local_q)).transpose();
    }

    Eigen::Vector2d panel_tangent(int panel, int local_q) const {
        return panel_tangents_.row(panel_local_row(panel, local_q)).transpose();
    }

    bool is_corner_point(int point) const {
        return point_kind_[point] == InterfacePointKind2D::Corner;
    }

    const CornerData2D& corner(int corner_id) const {
        return corners_[corner_id];
    }

    int corner_index(int point) const {
        return corner_index_by_point_[point];
    }

    const Eigen::MatrixX2d& points()           const { return points_; }
    const Eigen::MatrixX2d& normals()          const { return normals_; }
    const Eigen::VectorXd&  weights()          const { return weights_; }
    const Eigen::MatrixXi&  panel_point_indices() const { return panel_point_indices_; }
    const Eigen::VectorXi&  panel_components() const { return panel_components_; }
    const Eigen::MatrixX2d& panel_normals()     const { return panel_normals_; }
    const Eigen::MatrixX2d& panel_tangents()    const { return panel_tangents_; }
    const std::vector<InterfacePointKind2D>& point_kind() const { return point_kind_; }
    const std::vector<int>& corner_index_by_point() const { return corner_index_by_point_; }
    const std::vector<CornerData2D>& corners() const { return corners_; }
    const std::vector<CornerPatch2D>& corner_patches() const { return corner_patches_; }
    PanelNodeLayout2D       panel_node_layout() const { return panel_node_layout_; }

private:
    int panel_local_row(int panel, int local_q) const {
        return panel * points_per_panel_ + local_q;
    }

    void initialize_default_panel_side_geometry();
    void initialize_default_corner_metadata();
    void validate_topology() const;
    void validate() const;

    Eigen::MatrixX2d points_;
    Eigen::MatrixX2d normals_;
    Eigen::VectorXd  weights_;
    int              points_per_panel_;
    Eigen::MatrixXi  panel_point_indices_;
    Eigen::VectorXi  panel_components_;
    Eigen::MatrixX2d panel_normals_;
    Eigen::MatrixX2d panel_tangents_;
    std::vector<InterfacePointKind2D> point_kind_;
    std::vector<int> corner_index_by_point_;
    std::vector<CornerData2D> corners_;
    std::vector<CornerPatch2D> corner_patches_;
    PanelNodeLayout2D panel_node_layout_;
};

} // namespace kfbim
