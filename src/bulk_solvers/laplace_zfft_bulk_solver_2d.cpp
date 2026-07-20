#include "laplace_zfft_bulk_solver_2d.hpp"
#include "zfft.h"
#include <unsupported/Eigen/FFT>
#include <cmath>
#include <complex>
#include <limits>
#include <stdexcept>
#include <vector>

namespace kfbim {

namespace {
    bool is_power_of_two(int n) {
        return n > 0 && (n & (n - 1)) == 0;
    }

    Eigen::VectorXd make_dirichlet_eigenvalues(int intervals) {
        const int interior = intervals - 1;
        Eigen::VectorXd values(interior);
        constexpr double pi = 3.141592653589793238462643383279502884;
        for (int mode = 0; mode < interior; ++mode) {
            const double sine = std::sin(
                0.5 * pi * static_cast<double>(mode + 1)
                / static_cast<double>(intervals));
            values[mode] = 4.0 * sine * sine;
        }
        return values;
    }

    void apply_orthonormal_dst1_line(
        const std::vector<double>& input,
        std::vector<double>& output,
        Eigen::FFT<double>& fft,
        std::vector<double>& odd_extension,
        std::vector<std::complex<double>>& spectrum)
    {
        const int interior = static_cast<int>(input.size());
        const int intervals = interior + 1;
        const int fft_size = 2 * intervals;
        odd_extension.assign(static_cast<std::size_t>(fft_size), 0.0);
        for (int i = 0; i < interior; ++i) {
            odd_extension[static_cast<std::size_t>(i + 1)] = input[i];
            odd_extension[static_cast<std::size_t>(fft_size - i - 1)] =
                -input[i];
        }

        spectrum.resize(static_cast<std::size_t>(fft_size));
        fft.fwd(spectrum.data(), odd_extension.data(), fft_size);

        output.resize(static_cast<std::size_t>(interior));
        const double scale = -0.5 * std::sqrt(
            2.0 / static_cast<double>(intervals));
        for (int mode = 0; mode < interior; ++mode) {
            output[static_cast<std::size_t>(mode)] =
                scale * spectrum[static_cast<std::size_t>(mode + 1)].imag();
        }
    }

    void apply_orthonormal_dst1_axis(
        std::vector<double>& values,
        int mx,
        int my,
        int axis,
        Eigen::FFT<double>& fft)
    {
        const int line_size = axis == 0 ? mx : my;
        std::vector<double> line(static_cast<std::size_t>(line_size));
        std::vector<double> transformed;
        std::vector<double> odd_extension;
        std::vector<std::complex<double>> spectrum;
        const auto index = [mx](int i, int j) {
            return static_cast<std::size_t>(j * mx + i);
        };

        if (axis == 0) {
            for (int j = 0; j < my; ++j) {
                for (int i = 0; i < mx; ++i)
                    line[static_cast<std::size_t>(i)] = values[index(i, j)];
                apply_orthonormal_dst1_line(
                    line, transformed, fft, odd_extension, spectrum);
                for (int i = 0; i < mx; ++i)
                    values[index(i, j)] = transformed[static_cast<std::size_t>(i)];
            }
            return;
        }

        for (int i = 0; i < mx; ++i) {
            for (int j = 0; j < my; ++j)
                line[static_cast<std::size_t>(j)] = values[index(i, j)];
            apply_orthonormal_dst1_line(
                line, transformed, fft, odd_extension, spectrum);
            for (int j = 0; j < my; ++j)
                values[index(i, j)] = transformed[static_cast<std::size_t>(j)];
        }
    }

    int zfft_bc(ZfftBcType bc) {
        switch (bc) {
            case ZfftBcType::Periodic:  return ZFFT_PERIODIC;
            case ZfftBcType::Dirichlet: return ZFFT_DIRICHLET;
            case ZfftBcType::Neumann:   return ZFFT_NEUMANN;
        }
        return ZFFT_PERIODIC;
    }
}

LaplaceFftBulkSolverZfft2D::LaplaceFftBulkSolverZfft2D(
    CartesianGrid2D grid, ZfftBcType bc, double eta, int order)
    : grid_(std::move(grid)), bc_(bc), eta_(eta), order_(order)
{
    if (order_ != 2 && order_ != 4)
        throw std::invalid_argument("order must be 2 or 4");

    double dx = grid_.spacing()[0], dy = grid_.spacing()[1];
    if (std::fabs(dx - dy) > 1e-12 * dx)
        throw std::invalid_argument(
            "LaplaceFftBulkSolverZfft2D requires square cells (dx == dy)");

    const auto dims = grid_.dof_dims();
    const int intervals_x =
        bc_ == ZfftBcType::Periodic ? dims[0] : dims[0] - 1;
    const int intervals_y =
        bc_ == ZfftBcType::Periodic ? dims[1] : dims[1] - 1;
    const bool zfft_safe = is_power_of_two(intervals_x)
                        && is_power_of_two(intervals_y);
    if (!zfft_safe) {
        if (bc_ != ZfftBcType::Dirichlet || order_ != 2) {
            throw std::invalid_argument(
                "non-power-of-two 2D grids require second-order Dirichlet "
                "conditions; zFFT is not invoked because it can hang");
        }
        if (intervals_x < 2 || intervals_y < 2) {
            throw std::invalid_argument(
                "Dirichlet DST fallback requires at least two intervals per axis");
        }

        use_dirichlet_dst_ = true;
        eigenvalues_x_ = make_dirichlet_eigenvalues(intervals_x);
        eigenvalues_y_ = make_dirichlet_eigenvalues(intervals_y);
    }
}

void LaplaceFftBulkSolverZfft2D::solve_dirichlet_dst(
    const Eigen::VectorXd& rhs,
    Eigen::VectorXd& solution) const
{
    const auto dims = grid_.dof_dims();
    const int nx = dims[0];
    const int ny = dims[1];
    const int mx = nx - 2;
    const int my = ny - 2;
    const double h = grid_.spacing()[0];
    const double h2 = h * h;
    const double eta_z = h2 * eta_;

    std::vector<double> coefficients(
        static_cast<std::size_t>(mx * my), 0.0);
    const auto interior_index = [mx](int i, int j) {
        return j * mx + i;
    };
    for (int j = 0; j < my; ++j) {
        for (int i = 0; i < mx; ++i) {
            coefficients[static_cast<std::size_t>(interior_index(i, j))] =
                -h2 * rhs[(j + 1) * nx + (i + 1)];
        }
    }

    // Tensor-product orthonormal DST-I. Each one-dimensional transform uses
    // an odd extension and Eigen's arbitrary-length FFT implementation.
    Eigen::FFT<double> fft;
    apply_orthonormal_dst1_axis(coefficients, mx, my, 0, fft);
    apply_orthonormal_dst1_axis(coefficients, mx, my, 1, fft);

    for (int j = 0; j < my; ++j) {
        for (int i = 0; i < mx; ++i) {
            const double denominator = eigenvalues_x_[i]
                                     + eigenvalues_y_[j]
                                     + eta_z;
            if (std::abs(denominator) <=
                64.0 * std::numeric_limits<double>::epsilon()) {
                throw std::runtime_error(
                    "Dirichlet DST fallback encountered a zero eigenvalue");
            }
            coefficients[static_cast<std::size_t>(interior_index(i, j))] /=
                denominator;
        }
    }

    // The orthonormal DST-I is symmetric, hence its inverse is the same
    // transform. Axis transforms commute.
    apply_orthonormal_dst1_axis(coefficients, mx, my, 1, fft);
    apply_orthonormal_dst1_axis(coefficients, mx, my, 0, fft);

    solution = Eigen::VectorXd::Zero(nx * ny);
    for (int j = 0; j < my; ++j) {
        for (int i = 0; i < mx; ++i) {
            solution[(j + 1) * nx + (i + 1)] =
                coefficients[static_cast<std::size_t>(interior_index(i, j))];
        }
    }
}

// ---------------------------------------------------------------------------
// solve() — dispatch based on BC type.
//
// Sign convention: zfft solves (-stencil + eta_z) u = f_z, where
//   stencil u[i,j] = u[i+1]+u[i-1]+u[i,j+1]+u[i,j-1] - 4u[i,j]  (positive)
// The project interface solves (Delta_h - eta) u = -rhs, i.e.,
//   (-stencil/h^2 - eta) u = -rhs
//   (-stencil + eta_z) u = -h^2*rhs   where  eta_z = h^2 * eta
// So:  f_z = -h^2 * rhs_phys.
//
// Grid conventions:
//   Periodic  — dof_dims = {nx, ny}  (cells).  MatrixXd size (ny+1, nx+1).
//   Dirichlet — dof_dims = {nx, ny}  (nodes including boundary, nx=n+1 for n cells).
//               MatrixXd size (ny, nx):  rows()-1 = ny-1 = n intervals.
//   Neumann   — same as Dirichlet.
// ---------------------------------------------------------------------------
void LaplaceFftBulkSolverZfft2D::solve(const Eigen::VectorXd& rhs,
                                        Eigen::VectorXd&       solution) const
{
    if (rhs.size() != grid_.num_dofs())
        throw std::invalid_argument(
            "LaplaceFftBulkSolverZfft2D RHS size does not match the grid");
    if (use_dirichlet_dst_) {
        solve_dirichlet_dst(rhs, solution);
        return;
    }

    auto d  = grid_.dof_dims();
    int  nx = d[0];
    int  ny = d[1];
    double h  = grid_.spacing()[0];   // dx == dy enforced in ctor
    double h2 = h * h;
    double eta_z = h2 * eta_;

    if (bc_ == ZfftBcType::Periodic) {
        // Grid: (nx, ny) cell-center DOFs, flat = j*nx+i.
        // MatrixXd(ny+1, nx+1): n1=ny intervals, n2=nx intervals.
        // Periodic data: mat[j][i] for j=0..ny-1, i=0..nx-1.
        MatrixXd mat(ny + 1, nx + 1);
        mat.fill(0.0);

        for (int j = 0; j < ny; j++)
            for (int i = 0; i < nx; i++)
                mat[j][i] = -h2 * rhs[j * nx + i];

        zfft::FastDiffusionSolver2d(mat, eta_z, ZFFT_PERIODIC, order_);

        solution.resize(nx * ny);
        for (int j = 0; j < ny; j++)
            for (int i = 0; i < nx; i++)
                solution[j * nx + i] = mat[j][i];

    } else {
        // Grid: (nx, ny) node DOFs (includes boundary), flat = j*nx+i.
        // nx = n+1 for n intervals in x.  ny = n+1 for n intervals in y.
        // MatrixXd(ny, nx): rows()=ny → n1=ny-1 intervals, n2=nx-1 intervals.
        // Interior: mat[j][i] for j=1..ny-2, i=1..nx-2.
        // Boundary nodes (j=0, j=ny-1, i=0, i=nx-1) stay 0 (homogeneous BC).
        MatrixXd mat(ny, nx);
        mat.fill(0.0);

        if (bc_ == ZfftBcType::Dirichlet) {
            for (int j = 1; j < ny - 1; j++)
                for (int i = 1; i < nx - 1; i++)
                    mat[j][i] = -h2 * rhs[j * nx + i];
        } else {  // Neumann — full domain including boundary
            for (int j = 0; j < ny; j++)
                for (int i = 0; i < nx; i++)
                    mat[j][i] = -h2 * rhs[j * nx + i];
        }

        zfft::FastDiffusionSolver2d(mat, eta_z, zfft_bc(bc_), order_);

        solution.resize(nx * ny);
        for (int j = 0; j < ny; j++)
            for (int i = 0; i < nx; i++)
                solution[j * nx + i] = mat[j][i];
    }
}

} // namespace kfbim
