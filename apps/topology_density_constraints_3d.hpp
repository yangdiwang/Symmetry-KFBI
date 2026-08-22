#pragma once

#include "direct_coefficient_cauchy_3d.hpp"
#include "native_nurbs_density_space_3d.hpp"
#include "topology_affine_reduction_3d.hpp"

#include <vector>

namespace kfbim::app3d {

// Homogeneous topology layer for the BaseOnly native density space.  Feature
// jet right-hand sides deliberately live in a separate boundary-condition
// layer; this object assembles only C*y0 = 0 topology equations.
struct TopologyDensityConstraintOptions3D {
    int gauss_order = 5;
    int moment_degree = 3;
    int star_layers = 1;
    int star_coordinate_digits = 11;
    double coefficient_drop_tolerance = 1.0e-15;
    double zero_row_tolerance = 1.0e-13;
    double rhs_tolerance = 1.0e-12;
    bool merge_blocks_by_support = false;
};

struct TopologyDensityConstraintPlan3D {
    ConstraintSystem3D system;
    std::vector<ConstraintBlock3D> blocks;
    int connection_count = 0;
    int overlay_cell_count = 0;
    int c0_row_count = 0;
    int smooth_c1_row_count = 0;
    int vertex_block_count = 0;
    int edge_block_count = 0;
};

// Return a deterministic physical smooth-sheet id for every patch.  Sheets
// are the connected components of the patch graph containing G1 connections
// only; ids are assigned by the lexicographically sorted component signature
// and therefore do not depend on connection traversal order or geometry pose.
[[nodiscard]] std::vector<int> physical_smooth_sheet_ids_3d(
    const NativeNurbsSurface3D& surface);

// Sparse affine sharp-edge jump-jet operators for the topology route.  The
// two matrices encode
//
//        value_matrix * value_c0 = normal_target_matrix * normal_c0.
//
// Both spaces must use the BaseOnly/TopologyBase backend.  In particular,
// assembly consumes the at-most-16-entry C0 value/directional stencils and
// never forms a mortar-row-by-global-DOF dense matrix.  The legacy
// NativeFeatureEdgeJumpJetConstraints3D class remains available unchanged to
// the legacy solve route.
struct TopologyFeatureJumpJetOperators3D {
    NativeFeatureEdgeJumpJetOptions3D options;
    SparseMatrixCSR3D value_matrix;
    SparseMatrixCSR3D normal_target_matrix;
    std::vector<ConstraintMeta3D> meta;
    std::vector<ConstraintBlock3D> blocks;
    std::vector<NativeFeatureEdgeJumpJetInfo3D> edges;
    Eigen::VectorXd row_scalings;
    double maximum_point_mismatch = 0.0;
    double minimum_abs_dihedral_sine = 1.0;
    double constant_value_residual = 0.0;

    [[nodiscard]] int feature_edge_count() const noexcept
    {
        return static_cast<int>(edges.size());
    }
    [[nodiscard]] int constraint_count() const noexcept
    {
        return static_cast<int>(value_matrix.rows());
    }
    [[nodiscard]] Eigen::VectorXd normal_target(
        const Eigen::Ref<const Eigen::VectorXd>& normal_c0) const;
    [[nodiscard]] Eigen::VectorXd residual(
        const Eigen::Ref<const Eigen::VectorXd>& value_c0,
        const Eigen::Ref<const Eigen::VectorXd>& normal_c0) const;
    [[nodiscard]] ConstraintSystem3D bind_normal_target(
        const Eigen::Ref<const Eigen::VectorXd>& normal_c0) const;
};

// Opt-in affine sharp-edge compatibility for the Dirichlet formulation.
// NormalTrace remains broken in NativeNurbsDensitySpace3D; these equations
// are an additional boundary-condition layer whose unknown is the two-sided
// normal density J1.  If t is the oriented edge tangent, m_i=n_i x t,
// c=n_a.n_b and s=n_b.m_a, ambient-gradient continuity gives
//
//   (J1_b-c J1_a)/s = grad_Gamma(g_D)_a.m_a,
//   (c J1_b-J1_a)/s = grad_Gamma(g_D)_b.m_b.
//
// Each equation is integrated against cellwise Legendre P0..P3 moments.
// The default is disabled so constructing the ordinary topology plan cannot
// silently change the legacy Dirichlet normal-density space.
struct TopologyDirichletFeatureConstraintOptions3D {
    bool enabled = false;
    int mortar_gauss_order = 5;
    double minimum_abs_dihedral_sine = 1.0e-8;
    bool normalize_rows = true;
    // The edge-tangent derivative cannot be repaired by either J1 value.
    // Reject analytically inconsistent two-sided g_D data relative to
    // max(1, |grad_a|, |grad_b|).
    double tangential_compatibility_tolerance = 2.0e-9;
};

struct TopologyDirichletFeatureConstraintPlan3D {
    TopologyDirichletFeatureConstraintOptions3D options;
    ConstraintSystem3D system;
    std::vector<ConstraintBlock3D> blocks;
    std::vector<NativeFeatureEdgeJumpJetInfo3D> edges;
    Eigen::VectorXd row_scalings;
    double maximum_point_mismatch = 0.0;
    double minimum_abs_dihedral_sine = 1.0;
    double maximum_tangential_derivative_mismatch = 0.0;

    [[nodiscard]] int feature_edge_count() const noexcept
    {
        return static_cast<int>(edges.size());
    }
    [[nodiscard]] int constraint_count() const noexcept
    {
        return static_cast<int>(system.C.rows());
    }
};

// Assemble cellwise Legendre P0..P3 moments over the union of the two sides'
// knot images.  ValueTrace couples C0 across features but not C1;
// NormalTrace leaves features completely broken.  Both fields use C0 plus a
// world-space common-conormal C1 equation on G1 connections.
[[nodiscard]] ConstraintSystem3D assemble_topology_density_constraints_3d(
    const NativeNurbsDensitySpace3D& base_space,
    TopologyDensityConstraintOptions3D options = {});

// Rows whose endpoint-cell metadata share a rounded physical point form one
// Vertex/T-star block.  Remaining rows form connection-local edge blocks.
// The returned order is always all vertex blocks followed by all edge blocks.
[[nodiscard]] std::vector<ConstraintBlock3D>
build_topology_density_constraint_blocks_3d(
    const ConstraintSystem3D& system,
    bool merge_by_support = false,
    double support_tolerance = 0.0);

[[nodiscard]] TopologyDensityConstraintPlan3D
make_topology_density_constraint_plan_3d(
    const NativeNurbsDensitySpace3D& base_space,
    TopologyDensityConstraintOptions3D options = {});

// Assemble feature-edge jump-jet moments directly in sparse Base/C0
// coordinates.  Every knot-overlay cell owns Legendre P0..P3 rows, ordered as
// first-side then second-side for each mode.  Endpoint-cell rows enter the
// physical Vertex/T-star block; only interior-cell rows remain edge-local.
[[nodiscard]] TopologyFeatureJumpJetOperators3D
make_topology_feature_jump_jet_operators_3d(
    const NativeNurbsDensitySpace3D& value_base_space,
    const NativeNurbsDensitySpace3D& normal_base_space,
    NativeFeatureEdgeJumpJetOptions3D options = {},
    TopologyDensityConstraintOptions3D topology_options = {},
    double coefficient_drop_tolerance = 1.0e-15);

// Assemble only the affine feature rows C_feature*c=d_feature for a
// BaseOnly NormalTrace space.  The right-hand side is evaluated directly from
// the analytic known-Dirichlet callback at mortar quadrature points: no
// coefficient fit or reachable-target projection is performed.  Callers may
// append this system to make_topology_density_constraint_plan_3d(...), shift
// the returned block row ids by the topology row count, and pass the combined
// system to affine_eliminate_local_svd_3d to obtain c=c_p+Gz.
[[nodiscard]] TopologyDirichletFeatureConstraintPlan3D
make_topology_dirichlet_feature_constraint_plan_3d(
    const NativeNurbsDensitySpace3D& normal_base_space,
    const KnownDirichletValueGradientHessianCallback3D& known_dirichlet,
    TopologyDirichletFeatureConstraintOptions3D options = {},
    TopologyDensityConstraintOptions3D topology_options = {});

} // namespace kfbim::app3d
