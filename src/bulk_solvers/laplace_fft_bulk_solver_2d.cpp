#include "laplace_fft_bulk_solver_2d.hpp"
#include <cmath>
#include <stdexcept>

namespace kfbim {

// Data layout: flat index = j*nx+i (y-outer, x-inner), matching CartesianGrid.
// Engine complex output: out[j*(nx/2+1)+i] for j=ky-index, i=kx-index.
// Eigenvalue storage uses the same (ky, kx) indexing.

LaplaceFftBulkSolver2D::LaplaceFftBulkSolver2D(
    CartesianGrid2D              grid,
    std::unique_ptr<IFFTEngine2D> engine)
    : grid_(std::move(grid)), engine_(std::move(engine))
{
    auto d  = grid_.dof_dims();
    int  nx = d[0];
    int  ny = d[1];
    if (engine_->nx() != nx || engine_->ny() != ny)
        throw std::invalid_argument(
            "LaplaceFftBulkSolver2D: engine dimensions must match grid dof_dims");

    double dx = grid_.spacing()[0];
    double dy = grid_.spacing()[1];
    int    nc = nx / 2 + 1;   // x-halved dimension in r2c output

    eigenvalues_.resize(ny * nc);
    buf_c_.resize(ny * nc);

    // Outer loop: ky-index (j), inner loop: kx-index (i)
    for (int j = 0; j < ny; ++j) {
        double ly = -4.0 * std::sin(M_PI * j / ny) * std::sin(M_PI * j / ny) / (dy * dy);
        for (int i = 0; i < nc; ++i) {
            double lx = -4.0 * std::sin(M_PI * i / nx) * std::sin(M_PI * i / nx) / (dx * dx);
            eigenvalues_[j * nc + i] = lx + ly;
        }
    }
}

void LaplaceFftBulkSolver2D::solve(const Eigen::VectorXd& rhs,
                                    Eigen::VectorXd&       solution) const
{
    auto d  = grid_.dof_dims();
    int  nx = d[0];
    int  ny = d[1];
    int  nc = nx / 2 + 1;
    double norm = 1.0 / (static_cast<double>(nx) * ny);

    engine_->forward(rhs.data(), buf_c_.data());

    // DC mode: zero (selects zero-mean solution)
    buf_c_[0] = 0.0;

    for (int j = 0; j < ny; ++j) {
        for (int i = 0; i < nc; ++i) {
            if (j == 0 && i == 0) continue;
            buf_c_[j * nc + i] /= eigenvalues_[j * nc + i];
        }
    }

    solution.resize(nx * ny);
    engine_->backward(buf_c_.data(), solution.data());
    solution *= norm;
}

} // namespace kfbim
