#pragma once

#include <memory>
#include <vector>

#include <Eigen/Dense>

#include "../bulk_solvers/laplace_zfft_bulk_solver_2d.hpp"
#include "../geometry/grid_pair_2d.hpp"
#include "../transfer/laplace_nurbs_density_trace_state_2d.hpp"
#include "../transfer/laplace_restrict_2d.hpp"
#include "../transfer/laplace_spread_2d.hpp"
#include "i_kfbi_operator.hpp"

namespace kfbim {

// NURBS--Spline-Coefficient Exterior-Trace KFBI (NSC-ET-KFBI).
//
// The trial vector contains reduced P3(phi) or P2(psi) B-spline
// coefficients.  Exterior traces are recovered at physical Gauss points on
// every density span and projected back into coefficient coordinates with one
// cached weighted-SVD map.  No boundary point value is a density unknown.
enum class LaplaceNscEtBvpType2D {
    InteriorDirichlet,
    InteriorNeumann
};

struct LaplaceNscEtOptions2D {
    double density_spacing_over_h = 1.5;
    // Production densities are splines in physical arclength. The independent
    // chart-aligned NURBS-parameter coordinate remains available for
    // covariant-difference and reproducible A/B convergence studies.
    NurbsDensityCoordinate2D density_coordinate =
        NurbsDensityCoordinate2D::PhysicalArclength;
    // Crossing jets are obtained from neighbouring density-value samples in
    // physical arclength by default. AnalyticBasis and the independent
    // NURBS-parameter covariant-difference path remain available for
    // controlled comparisons.
    NurbsDensityDerivativeScheme2D density_derivative_scheme =
        NurbsDensityDerivativeScheme2D::SampledFiniteDifference;
    // Physical-arclength sampled differences use the previously calibrated
    // half-span step. Parameter-covariant differences need a shorter step to
    // control the P2 psi_xi truncation error on nonuniform rational charts.
    double density_difference_step_over_span = 0.5;
    double covariant_parameter_difference_step_over_span = 0.125;
    int restrict_stencil_radius = 2;
    LaplaceCorrectionMethod2D correction_method =
        LaplaceCorrectionMethod2D::CrossingOwner;
    LaplaceP2JointPolynomialRestrictOptions2D restrict_options =
        make_laplace_p2_dof_cauchy_exterior_restrict_options_2d();

    // Numerical continuity at every exact NURBS branch connection.  The
    // phi vector must use at most C2 and psi at most C1.
    std::vector<NurbsDensityContinuity2D> phi_connections;
    std::vector<NurbsDensityContinuity2D> psi_connections;

    // Legacy collocation controls retained for source compatibility.  The
    // production Gauss coefficient projection does not select these rows or
    // apply the historical collocation right preconditioner.
    std::vector<int> trace_candidates;
    double collocation_rank_tolerance = 1.0e-11;

    // Deprecated legacy switch.  The Gauss coefficient projection already
    // returns residuals in density coefficient coordinates, so no separate
    // collocation right preconditioner is applied.
    bool use_density_collocation_preconditioner = false;

    // Gauss-Legendre nodes per smooth integration cell obtained by splitting
    // each nonzero density span at ordinary NURBS knots.  Four is the
    // production default (the natural P3 mass order); 2, 3, and 8 are also
    // supported for coverage/validation studies.
    int trace_gauss_order = 4;

    // Preserve every fitted exterior-trace coefficient in a bordered
    // Neumann system and remove its constant-density nullspace with an exact
    // mean constraint plus one Lagrange multiplier.  The legacy N-1
    // orthogonal trial reduction remains available for comparisons only.
    bool use_full_neumann_bordered_system = true;

    // Use an orthonormal Householder basis for the Neumann mean-zero
    // coefficient subspace.  The legacy single-pivot elimination is retained
    // only for controlled reproduction studies.
    bool use_orthogonal_neumann_gauge = true;
};

struct LaplaceNscEtSolveResult2D {
    Eigen::VectorXd u_bulk;
    Eigen::VectorXd unknown_reduced_coefficients;
    Eigen::VectorXd unknown_raw_coefficients;
    Eigen::VectorXd unknown_at_trace_points;
    Eigen::VectorXd exterior_trace;
    Eigen::VectorXd projected_exterior_trace;
    Eigen::VectorXd physical_trace;
    Eigen::VectorXd physical_normal_trace;
    std::vector<double> residuals;
    int iterations = 0;
    bool gmres_converged = false;
    double exterior_trace_relative = 0.0;
    double exterior_trace_inf = 0.0;
    double projected_exterior_trace_relative = 0.0;
    double bordered_residual_relative = 0.0;
    double exterior_trace_weighted_mean = 0.0;
    double density_mean = 0.0;
    double neumann_gauge_multiplier = 0.0;
};

class LaplaceNscEtBvp2D final : public IKFBIOperator {
public:
    LaplaceNscEtBvp2D(
        const CartesianGrid2D& grid,
        const Interface2D& iface,
        LaplaceNscEtBvpType2D type,
        LaplaceNscEtOptions2D options = {});

    void apply(const Eigen::VectorXd& x, Eigen::VectorXd& y) const override;
    int problem_size() const override;

    // boundary_function is g_D for Dirichlet and g_N for Neumann.  This
    // first production path solves harmonic manufactured/interior problems;
    // the fixed density is fitted once, outside every GMRES matvec.
    LaplaceNscEtSolveResult2D solve_harmonic(
        const NurbsDensitySpace2D::Function& boundary_function,
        int max_iter = 120,
        double tolerance = 1.0e-9,
        int restart = 60) const;

    const GridPair2D& grid_pair() const { return grid_pair_; }
    LaplaceNscEtBvpType2D type() const { return type_; }
    const NurbsDensitySpace2D& phi_space() const { return *phi_space_; }
    const NurbsDensitySpace2D& psi_space() const { return *psi_space_; }
    const NurbsDensitySpace2D& unknown_space() const;
    const std::vector<int>& trace_points() const { return trace_points_; }
    const std::vector<LaplaceNurbsTraceReferencePoint2D>&
    trace_reference_points() const { return trace_reference_points_; }
    const Eigen::VectorXd& trace_quadrature_weights() const
    {
        return trace_quadrature_weights_;
    }
    const Eigen::MatrixXd& gauge_reduction() const { return gauge_reduction_; }
    const Eigen::MatrixXd& density_collocation_matrix() const
    {
        return density_collocation_matrix_;
    }
    const Eigen::MatrixXd& trace_design_matrix() const
    {
        return density_collocation_matrix_;
    }
    const Eigen::MatrixXd& trace_projection_matrix() const
    {
        return trace_projection_matrix_;
    }
    double trace_projection_identity_residual() const
    {
        return trace_projection_identity_residual_;
    }
    double trace_projection_condition() const
    {
        return trace_projection_condition_;
    }
    int trace_projection_build_count() const
    {
        return trace_projection_build_count_;
    }
    double neumann_border_row_norm() const
    {
        return neumann_border_mass_functional_.norm();
    }
    double neumann_border_column_norm() const
    {
        return neumann_stabilization_direction_.norm();
    }
    int density_unknown_count() const;
    int raw_density_unknown_count() const;
    const LaplaceP2JointPolynomialRestrictDiagnostics2D&
    restrict_diagnostics() const { return restrict_.diagnostics(); }

private:
    struct FieldResult {
        Eigen::VectorXd u_bulk;
        Eigen::VectorXd exterior_value;
        Eigen::VectorXd exterior_normal;
        Eigen::VectorXd interior_value;
        Eigen::VectorXd interior_normal;
    };

    Eigen::VectorXd expand_solver_coordinates(
        const Eigen::VectorXd& solver_coordinates) const;
    int coefficient_solver_size() const;
    Eigen::VectorXd density_values_at_interface(
        const NurbsDensitySpace2D& space,
        const Eigen::VectorXd& coefficients) const;
    Eigen::VectorXd density_values_at_trace(
        const NurbsDensitySpace2D& space,
        const Eigen::VectorXd& coefficients) const;
    FieldResult evaluate_field(
        const Eigen::VectorXd& phi_coefficients,
        const Eigen::VectorXd& psi_coefficients,
        LaplaceCrossingTraceStencil2D trace_stencil,
        bool recover_physical_trace = false) const;
    Eigen::VectorXd exterior_equation_trace(const FieldResult& field) const;
    Eigen::VectorXd project_exterior_trace(
        const Eigen::VectorXd& trace) const;
    Eigen::VectorXd make_mass_row(const NurbsDensitySpace2D& space) const;
    void build_gauss_trace_references(int gauss_order);
    void build_trace_projection(double rank_tolerance);
    // Retained implementation for legacy A/B studies; not called by the
    // production Gauss-reference constructor.
    void build_trace_collocation(const std::vector<int>& candidates,
                                 double rank_tolerance);

    LaplaceNscEtBvpType2D type_;
    GridPair2D grid_pair_;
    LaplaceQuadraticPanelCenterSpread2D spread_;
    LaplaceFftBulkSolverZfft2D bulk_solver_;
    LaplaceP2CrossingOwnerJointPolynomialRestrict2D restrict_;
    std::shared_ptr<const NurbsDensitySpace2D> phi_space_;
    std::shared_ptr<const NurbsDensitySpace2D> psi_space_;
    Eigen::MatrixXd gauge_reduction_;
    Eigen::MatrixXd density_collocation_matrix_;
    Eigen::MatrixXd density_collocation_right_preconditioner_;
    Eigen::MatrixXd trace_projection_matrix_;
    Eigen::VectorXd trace_quadrature_weights_;
    Eigen::VectorXd neumann_mass_functional_;
    Eigen::VectorXd neumann_border_mass_functional_;
    Eigen::VectorXd neumann_stabilization_direction_;
    bool use_full_neumann_bordered_system_ = false;
    double trace_projection_identity_residual_ = 0.0;
    double trace_projection_condition_ = 0.0;
    int trace_projection_build_count_ = 0;
    int trace_gauss_order_ = 4;
    NurbsDensityDerivativeScheme2D density_derivative_scheme_ =
        NurbsDensityDerivativeScheme2D::SampledFiniteDifference;
    double density_difference_step_over_span_ = 0.5;
    std::vector<int> trace_points_;
    std::vector<LaplaceNurbsTraceReferencePoint2D>
        trace_reference_points_;
    Eigen::VectorXd zero_phi_;
    Eigen::VectorXd zero_psi_;
};

} // namespace kfbim
