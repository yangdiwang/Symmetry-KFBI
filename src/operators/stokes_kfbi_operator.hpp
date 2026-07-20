#pragma once

#include <vector>
#include <Eigen/Dense>
#include "i_kfbi_operator.hpp"
#include "../transfer/i_spread.hpp"
#include "../transfer/i_restrict.hpp"
#include "../bulk_solvers/i_bulk_solver.hpp"

namespace kfbim {

// ---------------------------------------------------------------------------
// Concrete Stokes KFBIM operators (Layer 3)
//
// Packing convention:
//   GMRES iterates on (u_jump, traction_jump) at every interface quadrature point.
//   2D layout (4 unknowns/point):
//     x[4*i+0] = ux_jump,   x[4*i+1] = uy_jump
//     x[4*i+2] = tx_jump,   x[4*i+3] = ty_jump
//   3D layout (6 unknowns/point):
//     x[6*i+0..2] = u_jump,  x[6*i+3..5] = traction_jump
//
//   y has the same layout and contains the corrected MAC velocity and pressure
//   traces (for pressure: y[*+2] or y[*+3] holds the interface pressure).
//
// base_rhs_ux/uy/p: bulk MAC grid RHS from the forcing f alone (no correction).
// rhs_derivs[i]:    body-force derivative jumps at interface point i (as in StokesJumpData).
//
// Component references are borrowed; they must outlive this object.
// ---------------------------------------------------------------------------

class StokesKFBIOperator2D : public IKFBIOperator {
public:
    StokesKFBIOperator2D(const IStokesSpread2D&      spread,
                          const IStokesBulkSolver2D&  bulk_solver,
                          const IStokesRestrict2D&    restrict_op,
                          Eigen::VectorXd             base_rhs_ux,
                          Eigen::VectorXd             base_rhs_uy,
                          Eigen::VectorXd             base_rhs_p,
                          std::vector<Eigen::MatrixX2d> rhs_derivs); // one per interface point

    void apply(const Eigen::VectorXd& x, Eigen::VectorXd& y) const override;

    // = 4 * Interface2D::num_points()
    int problem_size() const override;

private:
    const IStokesSpread2D&     spread_;
    const IStokesBulkSolver2D& bulk_solver_;
    const IStokesRestrict2D&   restrict_op_;
    Eigen::VectorXd             base_rhs_ux_;
    Eigen::VectorXd             base_rhs_uy_;
    Eigen::VectorXd             base_rhs_p_;
    std::vector<Eigen::MatrixX2d> rhs_derivs_; // num_points x (num_rhs_derivs x 2)
};

class StokesKFBIOperator3D : public IKFBIOperator {
public:
    StokesKFBIOperator3D(const IStokesSpread3D&      spread,
                          const IStokesBulkSolver3D&  bulk_solver,
                          const IStokesRestrict3D&    restrict_op,
                          Eigen::VectorXd             base_rhs_ux,
                          Eigen::VectorXd             base_rhs_uy,
                          Eigen::VectorXd             base_rhs_uz,
                          Eigen::VectorXd             base_rhs_p,
                          std::vector<Eigen::MatrixX3d> rhs_derivs);

    void apply(const Eigen::VectorXd& x, Eigen::VectorXd& y) const override;

    // = 6 * Interface3D::num_points()
    int problem_size() const override;

private:
    const IStokesSpread3D&     spread_;
    const IStokesBulkSolver3D& bulk_solver_;
    const IStokesRestrict3D&   restrict_op_;
    Eigen::VectorXd             base_rhs_ux_;
    Eigen::VectorXd             base_rhs_uy_;
    Eigen::VectorXd             base_rhs_uz_;
    Eigen::VectorXd             base_rhs_p_;
    std::vector<Eigen::MatrixX3d> rhs_derivs_;
};

} // namespace kfbim
