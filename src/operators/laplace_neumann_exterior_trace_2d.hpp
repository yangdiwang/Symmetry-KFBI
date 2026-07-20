#pragma once

#include <vector>

#include <Eigen/Dense>

#include "../bulk_solvers/laplace_zfft_bulk_solver_2d.hpp"
#include "../geometry/grid_pair_2d.hpp"
#include "../grid/cartesian_grid_2d.hpp"
#include "../interface/interface_2d.hpp"
#include "../transfer/laplace_restrict_2d.hpp"
#include "../transfer/laplace_spread_2d.hpp"
#include "i_kfbi_operator.hpp"

namespace kfbim {

// Options for the harmonic interior Neumann formulation whose unknown is the
// interior Dirichlet trace f and whose physical residual is the exterior
// Dirichlet trace.  This first 2D implementation intentionally has no corner
// patch, singular fitting, S_D lifting, volume forcing, or nonzero box data.
struct LaplaceNeumannExteriorTraceOptions2D {
    int restrict_stencil_radius = 2;
    LaplaceCorrectionMethod2D correction_method =
        LaplaceCorrectionMethod2D::NearestExpansionCenter;

    // Empty means every interface point.  If a subset is supplied, every P2
    // panel point must be active and every inactive metadata point must have
    // zero quadrature weight.
    std::vector<int> active_interface_points;

    // Top-right column z of [A z; w^T 0].  Empty selects the constant vector.
    Eigen::VectorXd border_column;
};

// Explicitly constructed right-hand side for A f = b.  trace_rhs is the
// production Python-form RHS -R_ext H_N(g); the explicitly named exterior
// field is retained as an equal-valued compatibility alias for diagnostics.
struct LaplaceNeumannExteriorTraceRhs2D {
    Eigen::VectorXd compatible_neumann_data;
    Eigen::VectorXd trace_rhs;
    Eigen::VectorXd trace_rhs_exterior_virtual;
    Eigen::VectorXd fixed_u_bulk;
    Eigen::VectorXd fixed_normal_trace_interior;
    double compatibility_mean_removed = 0.0;
    double compatibility_residual = 0.0;
    double trace_route_mismatch_inf = 0.0;
};

// One combined interface solve, recovered independently through the fixed
// interior-virtual and exterior-virtual trace routes.
struct LaplaceExteriorTraceField2D {
    Eigen::VectorXd u_bulk;
    Eigen::VectorXd trace_interior;
    Eigen::VectorXd trace_exterior_from_jump;
    Eigen::VectorXd trace_exterior_virtual;
    Eigen::VectorXd normal_trace_interior;
    Eigen::VectorXd normal_trace_exterior_from_jump;
    Eigen::VectorXd normal_trace_exterior_virtual;
    double trace_route_mismatch_inf = 0.0;
};

struct LaplaceNeumannExteriorTraceSolveResult2D {
    // Reconstructed total field H_D f + H_N g on the Cartesian box.
    Eigen::VectorXd u_bulk;

    // Active-interface data and traces.
    Eigen::VectorXd dirichlet_trace;
    Eigen::VectorXd compatible_neumann_data;
    Eigen::VectorXd fixed_trace_rhs;
    Eigen::VectorXd trace_interior;
    Eigen::VectorXd trace_exterior;
    Eigen::VectorXd trace_exterior_virtual;
    Eigen::VectorXd normal_trace_interior;
    Eigen::VectorXd normal_trace_exterior;

    // physical_trace_residual = b - A f is the negative directly recovered
    // total exterior trace.  augmented_residual is the residual of the
    // (N+1)-by-(N+1) bordered system.
    Eigen::VectorXd physical_trace_residual;
    Eigen::VectorXd augmented_residual;
    std::vector<double> augmented_relative_residuals;

    double compatibility_mean_removed = 0.0;
    double bordered_multiplier = 0.0;
    double weighted_trace_mean = 0.0;
    double constant_mode_residual_inf = 0.0;
    double trace_residual_closure_inf = 0.0;
    double independent_exterior_closure_inf = 0.0;
    double rhs_trace_route_mismatch_inf = 0.0;
    double superposition_bulk_inf = 0.0;
    double physical_relative_residual = 0.0;
    double exterior_trace_relative = 0.0;
    double augmented_relative_residual = 0.0;
    int iterations = 0;
    bool augmented_converged = false;
    bool physical_converged = false;
};

// Matrix-free direct-exterior operator matching the Python harmonic-jet
// iteration:
//
//     A f = R_ext H(f, 0),
//     b   = -R_ext H(0, g).
//
// R_ext is recovered directly from fixed exterior virtual samples.  solve()
// constructs the fixed normal-jump RHS once, then solves the weighted
// bordered system for (f, lambda).  The fixed normal jump never enters
// apply(), so the GMRES matvec remains linear.
class LaplaceNeumannExteriorTrace2D final : public IKFBIOperator {
public:
    LaplaceNeumannExteriorTrace2D(
        const CartesianGrid2D& grid,
        const Interface2D& iface,
        LaplaceNeumannExteriorTraceOptions2D options = {});

    void apply(const Eigen::VectorXd& x, Eigen::VectorXd& y) const override;
    int problem_size() const override;

    // Explicit direct-exterior entry point.  It is identical to production
    // apply() and remains available to existing diagnostics.
    void apply_direct_exterior(const Eigen::VectorXd& x,
                               Eigen::VectorXd& y) const;

    LaplaceNeumannExteriorTraceRhs2D build_rhs(
        const Eigen::VectorXd& neumann_data) const;

    LaplaceExteriorTraceField2D evaluate_cauchy(
        const Eigen::VectorXd& value_jump,
        const Eigen::VectorXd& normal_jump) const;

    LaplaceNeumannExteriorTraceSolveResult2D solve(
        const Eigen::VectorXd& neumann_data,
        int max_iter = 100,
        double tol = 1.0e-8,
        int restart = 50) const;

    LaplaceNeumannExteriorTraceSolveResult2D solve(
        const LaplaceNeumannExteriorTraceRhs2D& rhs_data,
        int max_iter = 100,
        double tol = 1.0e-8,
        int restart = 50) const;

    const GridPair2D& grid_pair() const { return grid_pair_; }
    const std::vector<int>& active_interface_points() const {
        return active_interface_points_;
    }
    const Eigen::VectorXd& active_weights() const { return active_weights_; }
    const Eigen::VectorXd& normalized_weights() const {
        return normalized_weights_;
    }
    const Eigen::VectorXd& border_column() const { return border_column_; }

private:
    struct JumpFieldResult2D {
        Eigen::VectorXd u_bulk;
        Eigen::VectorXd trace_interior;
        Eigen::VectorXd normal_trace_interior;
        Eigen::VectorXd trace_exterior_virtual;
        Eigen::VectorXd normal_trace_exterior_virtual;
    };

    Eigen::VectorXd expand_active(const Eigen::VectorXd& values) const;
    JumpFieldResult2D evaluate_jump(const Eigen::VectorXd& value_jump,
                                    const Eigen::VectorXd& normal_jump,
                                    bool recover_exterior_virtual = false) const;

    GridPair2D grid_pair_;
    LaplaceQuadraticPanelCenterSpread2D spread_;
    LaplaceFftBulkSolverZfft2D bulk_solver_;
    LaplaceQuadraticPanelCenterRestrict2D restrict_op_;

    std::vector<int> active_interface_points_;
    std::vector<int> iface_to_active_;
    Eigen::VectorXd active_weights_;
    Eigen::VectorXd normalized_weights_;
    Eigen::VectorXd border_column_;
    Eigen::VectorXd zero_bulk_rhs_;
    std::vector<Eigen::VectorXd> zero_rhs_derivs_;
};

} // namespace kfbim
