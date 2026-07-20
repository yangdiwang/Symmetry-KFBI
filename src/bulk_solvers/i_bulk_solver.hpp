#pragma once

#include <Eigen/Dense>
#include "../grid/i_cartesian_grid.hpp"
#include "../grid/mac_grid_2d.hpp"
#include "../grid/mac_grid_3d.hpp"

namespace kfbim {

// ---------------------------------------------------------------------------
// Abstract bulk solver interfaces
//
// The grid is baked in at construction; solve() only sees rhs and solution.
// Concrete impls are tied to specific grid types:
//   LaplaceBulkSolverFFT2D/3D     — uniform CartesianGrid, constant coefficients
//   LaplaceBulkSolverMG2D/3D      — adaptive Cartesian grid, geometric multigrid
//   StokesBulkSolverFFT2D/3D      — uniform MACGrid, 3 Poisson FFT solves
//   StokesBulkSolverUzawa2D/3D    — uniform MACGrid, CG-Uzawa
//   StokesBulkSolverMG2D/3D       — adaptive MACGrid, geometric multigrid
//
// grid() returns the abstract ICartesianGrid* so KFBIOperator and GridPair
// above this layer stay grid-agnostic.
// ---------------------------------------------------------------------------

class ILaplaceBulkSolver2D {
public:
    virtual ~ILaplaceBulkSolver2D() = default;

    // rhs and solution are indexed by grid().num_dofs()
    virtual void solve(const Eigen::VectorXd& rhs,
                       Eigen::VectorXd&       solution) const = 0;

    virtual const ICartesianGrid2D& grid() const = 0;
};

class ILaplaceBulkSolver3D {
public:
    virtual ~ILaplaceBulkSolver3D() = default;

    virtual void solve(const Eigen::VectorXd& rhs,
                       Eigen::VectorXd&       solution) const = 0;

    virtual const ICartesianGrid3D& grid() const = 0;
};

// ---------------------------------------------------------------------------
// Stokes bulk solver — operates on MAC velocity + pressure fields.
// rhs and solution components are each indexed by the DOF count of the
// corresponding sub-grid in MACGrid (ux, uy, p in 2D; ux, uy, uz, p in 3D).
// ---------------------------------------------------------------------------

class IStokesBulkSolver2D {
public:
    virtual ~IStokesBulkSolver2D() = default;

    virtual void solve(const Eigen::VectorXd& rhs_ux,
                       const Eigen::VectorXd& rhs_uy,
                       const Eigen::VectorXd& rhs_p,
                       Eigen::VectorXd&       sol_ux,
                       Eigen::VectorXd&       sol_uy,
                       Eigen::VectorXd&       sol_p) = 0;

    virtual const MACGrid2D& grid() const = 0;
};

class IStokesBulkSolver3D {
public:
    virtual ~IStokesBulkSolver3D() = default;

    virtual void solve(const Eigen::VectorXd& rhs_ux,
                       const Eigen::VectorXd& rhs_uy,
                       const Eigen::VectorXd& rhs_uz,
                       const Eigen::VectorXd& rhs_p,
                       Eigen::VectorXd&       sol_ux,
                       Eigen::VectorXd&       sol_uy,
                       Eigen::VectorXd&       sol_uz,
                       Eigen::VectorXd&       sol_p) = 0;

    virtual const MACGrid3D& grid() const = 0;
};

} // namespace kfbim
