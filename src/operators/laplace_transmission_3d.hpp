#pragma once

#include <vector>

#include <Eigen/Dense>

#include "../bulk_solvers/laplace_zfft_bulk_solver_3d.hpp"
#include "../geometry/grid_pair_3d.hpp"
#include "../grid/cartesian_grid_3d.hpp"
#include "../interface/interface_3d.hpp"
#include "../potentials/laplace_potential.hpp"
#include "../transfer/laplace_restrict_3d.hpp"
#include "../transfer/laplace_spread_3d.hpp"
#include "i_kfbi_operator.hpp"

namespace kfbim {

enum class LaplaceTransmissionMode3D {
    CommonRatio,
    DifferentRatios
};

struct LaplaceTransmissionCoefficients3D {
    double beta_int = 1.0;
    double beta_ext = 1.0;
    double kappa_sq_int = 0.0;
    double kappa_sq_ext = 0.0;
};

struct LaplaceTransmissionOptions3D {
    int restrict_stencil_radius = 2;
    LaplaceCorrectionMethod3D correction_method =
        LaplaceCorrectionMethod3D::NearestExpansionCenter;
};

struct LaplaceTransmissionRhsData3D {
    // Stores q=f/beta on grid nodes before outer-boundary Dirichlet elimination.
    Eigen::VectorXd reduced_rhs_bulk;

    // Entry q stores the interface jet of q_int and q_ext at point q.
    // The current 3D P2 path requires at least the value entry [0].
    std::vector<Eigen::VectorXd> reduced_rhs_int_derivs;
    std::vector<Eigen::VectorXd> reduced_rhs_ext_derivs;
};

struct LaplaceTransmissionSolveResult3D {
    Eigen::VectorXd     u_bulk;
    Eigen::VectorXd     phi_density;
    Eigen::VectorXd     psi_density;
    std::vector<double> residuals;
    int                 iterations;
    bool                converged;
};

// ---------------------------------------------------------------------------
// LaplaceTransmission3D
//
// Solves 3D piecewise-constant coefficient interface problems
//
//   -div(beta grad u) + kappa_sq u = f
//
// with jumps [u] and [beta du/dn]. The API takes reduced right-hand sides
// q=f/beta, so each phase uses -Delta u + lambda_sq*u = q with
// lambda_sq = kappa_sq / beta.
// ---------------------------------------------------------------------------
class LaplaceTransmission3D : public IKFBIOperator {
public:
    LaplaceTransmission3D(const CartesianGrid3D&            grid,
                          const Interface3D&                iface,
                          LaplaceTransmissionMode3D         mode,
                          LaplaceTransmissionCoefficients3D coefficients,
                          LaplaceTransmissionOptions3D      options = {});

    void apply(const Eigen::VectorXd& x, Eigen::VectorXd& y) const override;
    int problem_size() const override;

    LaplaceTransmissionSolveResult3D solve(
        const Eigen::VectorXd&              u_jump,
        const Eigen::VectorXd&              beta_flux_jump,
        const LaplaceTransmissionRhsData3D& rhs_data,
        const Eigen::VectorXd&              outer_dirichlet_values = Eigen::VectorXd(),
        int                                max_iter = 200,
        double                             tol = 1e-8,
        int                                restart = 80) const;

    const GridPair3D& grid_pair() const { return grid_pair_; }
    LaplaceTransmissionMode3D mode() const { return mode_; }
    const LaplaceTransmissionCoefficients3D& coefficients() const { return coefficients_; }
    double lambda_sq_int() const { return lambda_sq_int_; }
    double lambda_sq_ext() const { return lambda_sq_ext_; }

private:
    Eigen::VectorXd apply_dirichlet_boundary_elimination(
        const Eigen::VectorXd& reduced_rhs_bulk,
        const Eigen::VectorXd& outer_dirichlet_values) const;

    void restore_dirichlet_boundary(Eigen::VectorXd&       u_bulk,
                                    const Eigen::VectorXd& outer_dirichlet_values) const;

    void apply_common_ratio(const Eigen::VectorXd& x, Eigen::VectorXd& y) const;
    void apply_different_ratios(const Eigen::VectorXd& x, Eigen::VectorXd& y) const;

    LaplaceTransmissionSolveResult3D solve_common_ratio(
        const Eigen::VectorXd&              u_jump,
        const Eigen::VectorXd&              beta_flux_jump,
        const LaplaceTransmissionRhsData3D& rhs_data,
        const Eigen::VectorXd&              outer_dirichlet_values,
        int                                max_iter,
        double                             tol,
        int                                restart) const;

    LaplaceTransmissionSolveResult3D solve_different_ratios(
        const Eigen::VectorXd&              u_jump,
        const Eigen::VectorXd&              beta_flux_jump,
        const LaplaceTransmissionRhsData3D& rhs_data,
        const Eigen::VectorXd&              outer_dirichlet_values,
        int                                max_iter,
        double                             tol,
        int                                restart) const;

    GridPair3D                            grid_pair_;
    LaplaceTransmissionMode3D             mode_;
    LaplaceTransmissionCoefficients3D     coefficients_;
    double                                lambda_sq_int_;
    double                                lambda_sq_ext_;

    LaplaceQuadraticPatchCenterSpread3D   spread_int_;
    LaplaceQuadraticPatchCenterSpread3D   spread_ext_;
    LaplaceFftBulkSolverZfft3D            bulk_solver_int_;
    LaplaceFftBulkSolverZfft3D            bulk_solver_ext_;
    LaplaceQuadraticPatchCenterRestrict3D restrict_op_;
    LaplacePotentialEval3D                potentials_int_;
    LaplacePotentialEval3D                potentials_ext_;
};

} // namespace kfbim
