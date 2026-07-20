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

enum class LaplaceBvpType3D {
    InteriorDirichlet,
    ExteriorDirichlet,
    InteriorNeumann,
    ExteriorNeumann
};

struct LaplaceBvpOptions3D {
    double eta = 0.0;
    Eigen::VectorXd outer_dirichlet_values;
    int restrict_stencil_radius = 2;
    LaplaceCorrectionMethod3D correction_method =
        LaplaceCorrectionMethod3D::NearestExpansionCenter;
};

struct LaplaceBvpSolveResult3D {
    Eigen::VectorXd     u_bulk;
    Eigen::VectorXd     density;
    std::vector<double> residuals;
    int                 iterations;
    bool                converged;
};

// ---------------------------------------------------------------------------
// LaplaceBvp3D
//
// Solves 3D interior/exterior Dirichlet and Neumann BVPs on P2 triangular
// surfaces:
//   -Delta u + eta*u = f
//
// boundary_data stores Dirichlet values for Dirichlet modes and normal
// derivative values for Neumann modes.
// ---------------------------------------------------------------------------
class LaplaceBvp3D : public IKFBIOperator {
public:
    LaplaceBvp3D(const CartesianGrid3D& grid,
                 const Interface3D&     iface,
                 LaplaceBvpType3D       type,
                 LaplaceBvpOptions3D    options = {});

    // Uses correction_iface for GMRES DOFs/local Cauchy corrections and the
    // complete crossing_geometry for domain labels and grid-edge crossings.
    LaplaceBvp3D(const CartesianGrid3D& grid,
                 const Interface3D&     correction_iface,
                 const Interface3D&     crossing_geometry,
                 LaplaceBvpType3D       type,
                 LaplaceBvpOptions3D    options = {});

    void apply(const Eigen::VectorXd& x, Eigen::VectorXd& y) const override;
    int problem_size() const override;

    LaplaceBvpSolveResult3D solve(const Eigen::VectorXd&              boundary_data,
                                  const Eigen::VectorXd&              f_bulk,
                                  const std::vector<Eigen::VectorXd>& rhs_derivs,
                                  int max_iter = 100,
                                  double tol = 1e-8,
                                  int restart = 50) const;

    // Reconstruct the true residual of the reduced boundary equation for a
    // supplied density. Useful for local GMRES/stencil diagnostics.
    Eigen::VectorXd boundary_residual(
        const Eigen::VectorXd&              density,
        const Eigen::VectorXd&              boundary_data,
        const Eigen::VectorXd&              f_bulk,
        const std::vector<Eigen::VectorXd>& rhs_derivs) const;

    const GridPair3D& grid_pair() const { return grid_pair_; }
    LaplaceBvpType3D type() const { return type_; }
    double eta() const { return eta_; }

private:
    Eigen::VectorXd apply_dirichlet_boundary_elimination(
        const Eigen::VectorXd& rhs) const;

    void restore_dirichlet_boundary(Eigen::VectorXd& u_bulk) const;

    void apply_with_rhs(const Eigen::VectorXd&              density,
                        const Eigen::VectorXd&              rhs,
                        const std::vector<Eigen::VectorXd>& rhs_derivs,
                        Eigen::VectorXd&                    y) const;

    GridPair3D                            grid_pair_;
    LaplaceQuadraticPatchCenterSpread3D   spread_;
    LaplaceFftBulkSolverZfft3D            bulk_solver_;
    LaplaceQuadraticPatchCenterRestrict3D restrict_op_;
    LaplacePotentialEval3D                potentials_;

    LaplaceBvpType3D type_;
    double           rhs_deriv_sign_;
    double           eta_;
    Eigen::VectorXd  outer_dirichlet_values_;
};

} // namespace kfbim
