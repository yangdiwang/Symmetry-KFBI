#pragma once

#include "src/support/cauchy/direct_coefficient_cubic_cauchy_3d.hpp"
#include <Eigen/SparseCore>

namespace kfbim::app3d {
struct Trace93Case3D;

struct Trace93DensityOptions3D {
    double factor = 8.0;
    double edge_star_weight = 0.1;
    double vertex_star_weight = 0.0;
    double vertex_star_power = 0.0;
};

struct Trace93PolarDiagnostics3D {
    int essential_rank = 0, polar_rank = 0, kept = 0, defect = 0;
    double gap_factor = 1.0, left_null_rhs = 0.0;
    double maximum_condition = 0.0, maximum_beta = 0.0;
    double maximum_point_mismatch = 0.0;
    Eigen::VectorXd singular_value_ratios;
};

// The Python analysis parameters, not native rational-NURBS parameters,
// define this coefficient space. In particular cylinder angle splines are
// kept exactly; they cannot be converted to rational-parameter cubic splines
// by finite knot insertion. Coefficient order is patch-major j*ncu+i.
struct Trace93DensityLayout3D {
    using Sparse = Eigen::SparseMatrix<double, Eigen::RowMajor>;
    const Trace93Case3D* problem = nullptr; // Must outlive this layout/catalog setup.
    bool neumann = false;
    double h = 0.0;
    int reference_raw_dofs = 0, rank = 0;
    std::vector<std::array<int, 2>> patch_spans;
    std::vector<int> patch_offsets;
    Eigen::MatrixXd Z, C;
    Eigen::VectorXd particular, d;
    std::vector<std::string> constraint_kinds;
    double constraint_residual = 0.0, nullspace_residual = 0.0;
    Trace93PolarDiagnostics3D polar;
    std::vector<NativeDensityGaussPoint3D> traces;
    Eigen::VectorXd weights;
    Eigen::RowVectorXd mean_row;
    Sparse trace_basis;
    // Extra projector terms: M += (edge_basis*Z)^T diag(edge_weights)
    // (edge_basis*Z), rhs(t) += (edge_basis*Z)^T diag(edge_weights)
    // edge_trace*t; likewise vertex_*. Weights include configured scales.
    Sparse edge_trace, edge_basis, vertex_trace, vertex_basis;
    Eigen::VectorXd edge_weights, vertex_weights;

    [[nodiscard]] int coefficient_count() const { return reference_raw_dofs; }
    [[nodiscard]] std::array<NativeDensityC0Stencil3D,10>
    parameter_cubic_jet_stencils(int patch, double u, double v) const;
    [[nodiscard]] NativeDensityC0Stencil3D basis_stencil(
        int patch, double u, double v) const;
    [[nodiscard]] Eigen::RowVectorXd basis_row(int patch,double u,double v) const;
    [[nodiscard]] Eigen::RowVectorXd directional_row(int patch,double u,double v,
        const Eigen::Vector3d& direction) const;
    [[nodiscard]] NativeSurfaceCubicParameterJet3D geometry_jet(
        int patch,double u,double v) const;
    [[nodiscard]] DirectCoefficientCubicCauchyPlan3D cauchy_plan(
        int patch,double u,double v) const;
};

// Planar: exact coefficient C0/C1 + known Dirichlet edge interpolation;
// Neumann: essential constraints + weak Polar-Star spectral selection.
// Cylinder: metric-matched dyadic spans, smooth Greville constraints,
// junction-defect Dirichlet / Polar-Star Neumann. No unknown-density fitting.
[[nodiscard]] Trace93DensityLayout3D build_trace93_density_layout_3d(
    const Trace93Case3D& problem,double h,bool neumann,
    Trace93DensityOptions3D options = {});
} // namespace kfbim::app3d
