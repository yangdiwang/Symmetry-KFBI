#pragma once

#include "i_bulk_solver.hpp"
#include "fft_engine.hpp"
#include "../grid/cartesian_grid_2d.hpp"
#include <Eigen/Dense>
#include <memory>
#include <vector>
#include <complex>

namespace kfbim {

// ---------------------------------------------------------------------------
// LaplaceFftBulkSolver2D
//
// Solves  Δu = f  on a uniform periodic Cartesian grid by spectral division.
//
// The discrete Laplacian is the standard 5-point stencil with periodic
// wrap-around. Its eigenvalues for mode (kx, ky) are:
//
//   λ(kx, ky) = −4 sin²(π kx / nx) / dx²  −  4 sin²(π ky / ny) / dy²
//
// The DC mode (kx=ky=0, λ=0) is set to zero in spectral space, which
// chooses the unique zero-mean solution.  The caller should ensure f has
// zero mean (solvability condition for the periodic Poisson problem).
//
// grid must use DofLayout2D::CellCenter with dimensions matching engine.
// ---------------------------------------------------------------------------

class LaplaceFftBulkSolver2D : public ILaplaceBulkSolver2D {
public:
    LaplaceFftBulkSolver2D(CartesianGrid2D              grid,
                            std::unique_ptr<IFFTEngine2D> engine);

    void solve(const Eigen::VectorXd& rhs,
               Eigen::VectorXd&       solution) const override;

    const ICartesianGrid2D& grid() const override { return grid_; }

private:
    CartesianGrid2D                   grid_;
    std::unique_ptr<IFFTEngine2D>     engine_;
    std::vector<double>               eigenvalues_;   // size ny*(nx/2+1), index [ky*(nx/2+1)+kx]
    mutable std::vector<std::complex<double>> buf_c_; // spectral work buffer, same size
};

} // namespace kfbim
