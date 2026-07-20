#pragma once

#include <memory>
#include <vector>

#include <Eigen/Dense>

#include "../geometry/grid_pair_2d.hpp"
#include "../geometry/corner_patch_2d.hpp"
#include "../grid/cartesian_grid_2d.hpp"
#include "../interface/interface_2d.hpp"
#include "i_kfbi_operator.hpp"
#include "../potentials/laplace_potential.hpp"
#include "../bulk_solvers/laplace_zfft_bulk_solver_2d.hpp"
#include "../transfer/laplace_restrict_2d.hpp"
#include "../transfer/laplace_spread_2d.hpp"

namespace kfbim {

enum class LaplaceTransmissionMode2D {
    CommonRatio,
    DifferentRatios
};

struct LaplaceTransmissionCoefficients2D {
    double beta_int = 1.0;
    double beta_ext = 1.0;
    double kappa_sq_int = 0.0;
    double kappa_sq_ext = 0.0;
};

struct LaplaceTransmissionOptions2D {
    ZfftBcType bulk_bc = ZfftBcType::Dirichlet;
    int restrict_stencil_radius = 2;
    LaplaceCorrectionMethod2D correction_method =
        LaplaceCorrectionMethod2D::NearestExpansionCenter;
    CornerPatchConfig2D corner_patch;
    CornerPatchSolverOptions2D corner_patch_solver;
};

struct LaplaceTransmissionRhsData2D {
    // Stores q=f/beta on grid nodes before outer-boundary Dirichlet elimination.
    Eigen::VectorXd reduced_rhs_bulk;

    // Entry q stores the interface jet of q_int and q_ext at point q.
    // The current 2D local Cauchy path requires at least the value entry [0].
    std::vector<Eigen::VectorXd> reduced_rhs_int_derivs;
    std::vector<Eigen::VectorXd> reduced_rhs_ext_derivs;
};

struct LaplaceTransmissionSolveResult2D {
    Eigen::VectorXd     u_bulk;
    Eigen::VectorXd     u_physical;
    Eigen::VectorXd     phi_density;
    Eigen::VectorXd     psi_density;
    std::vector<CornerPatchCorrectionData2D> corner_patch_corrections;
    std::vector<double> residuals;
    int                 iterations;
    bool                converged;
};

// ---------------------------------------------------------------------------
// LaplaceTransmission2D
//
// Solves 2D piecewise-constant coefficient interface problems
//
//   -div(beta grad u) + kappa_sq u = f
//
// with jumps [u] and [beta du/dn].  The API takes reduced right-hand sides
// q=f/beta, so each phase uses -Delta u + lambda_sq*u = q with
// lambda_sq = kappa_sq / beta.
//
// CommonRatio mode preserves the one-density second-kind equation used when
// lambda_sq_int == lambda_sq_ext.  DifferentRatios mode uses the generic
// two-density KFBI operator from the theory notes.
// ---------------------------------------------------------------------------
class LaplaceTransmission2D : public IKFBIOperator {
public:
    LaplaceTransmission2D(const CartesianGrid2D&                 grid,
                          const Interface2D&                     iface,
                          LaplaceTransmissionMode2D              mode,
                          LaplaceTransmissionCoefficients2D      coefficients,
                          ZfftBcType                             bulk_bc = ZfftBcType::Dirichlet);

    LaplaceTransmission2D(const CartesianGrid2D&                 grid,
                          const Interface2D&                     iface,
                          LaplaceTransmissionMode2D              mode,
                          LaplaceTransmissionCoefficients2D      coefficients,
                          LaplaceTransmissionOptions2D           options);

    void apply(const Eigen::VectorXd& x, Eigen::VectorXd& y) const override;
    int problem_size() const override;

    LaplaceTransmissionSolveResult2D solve(
        const Eigen::VectorXd&              u_jump,
        const Eigen::VectorXd&              beta_flux_jump,
        const LaplaceTransmissionRhsData2D& rhs_data,
        const Eigen::VectorXd&              outer_dirichlet_values = Eigen::VectorXd(),
        int                                max_iter = 200,
        double                             tol = 1e-8,
        int                                restart = 80) const;

    const GridPair2D& grid_pair() const { return grid_pair_; }
    LaplaceTransmissionMode2D mode() const { return mode_; }
    const LaplaceTransmissionCoefficients2D& coefficients() const { return coefficients_; }
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

    LaplaceTransmissionSolveResult2D solve_common_ratio(
        const Eigen::VectorXd&              u_jump,
        const Eigen::VectorXd&              beta_flux_jump,
        const LaplaceTransmissionRhsData2D& rhs_data,
        const Eigen::VectorXd&              outer_dirichlet_values,
        int                                max_iter,
        double                             tol,
        int                                restart) const;

    LaplaceTransmissionSolveResult2D solve_different_ratios(
        const Eigen::VectorXd&              u_jump,
        const Eigen::VectorXd&              beta_flux_jump,
        const LaplaceTransmissionRhsData2D& rhs_data,
        const Eigen::VectorXd&              outer_dirichlet_values,
        int                                max_iter,
        double                             tol,
        int                                restart) const;

    std::shared_ptr<const Interface2D> effective_iface_;
    GridPair2D                       grid_pair_;
    LaplaceTransmissionMode2D        mode_;
    LaplaceTransmissionCoefficients2D coefficients_;
    ZfftBcType                       bulk_bc_;
    double                           lambda_sq_int_;
    double                           lambda_sq_ext_;

    LaplaceQuadraticPanelCenterSpread2D   spread_int_;
    LaplaceQuadraticPanelCenterSpread2D   spread_ext_;
    LaplaceFftBulkSolverZfft2D     bulk_solver_int_;
    LaplaceFftBulkSolverZfft2D     bulk_solver_ext_;
    LaplaceQuadraticPanelCenterRestrict2D restrict_op_;
    LaplacePotentialEval2D         potentials_int_;
    LaplacePotentialEval2D         potentials_ext_;
};

} // namespace kfbim
