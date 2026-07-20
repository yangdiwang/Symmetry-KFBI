#include "laplace_zfft_bulk_solver_3d.hpp"
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
            odd_extension[static_cast<std::size_t>(fft_size - i - 1)] = -input[i];
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
        int mz,
        int axis,
        Eigen::FFT<double>& fft)
    {
        const int line_size = axis == 0 ? mx : (axis == 1 ? my : mz);
        std::vector<double> line(static_cast<std::size_t>(line_size));
        std::vector<double> transformed;
        std::vector<double> odd_extension;
        std::vector<std::complex<double>> spectrum;
        const auto index = [mx, my](int i, int j, int k) {
            return static_cast<std::size_t>((k * my + j) * mx + i);
        };

        if (axis == 0) {
            for (int k = 0; k < mz; ++k) {
                for (int j = 0; j < my; ++j) {
                    for (int i = 0; i < mx; ++i)
                        line[static_cast<std::size_t>(i)] = values[index(i, j, k)];
                    apply_orthonormal_dst1_line(
                        line, transformed, fft, odd_extension, spectrum);
                    for (int i = 0; i < mx; ++i)
                        values[index(i, j, k)] = transformed[static_cast<std::size_t>(i)];
                }
            }
            return;
        }
        if (axis == 1) {
            for (int k = 0; k < mz; ++k) {
                for (int i = 0; i < mx; ++i) {
                    for (int j = 0; j < my; ++j)
                        line[static_cast<std::size_t>(j)] = values[index(i, j, k)];
                    apply_orthonormal_dst1_line(
                        line, transformed, fft, odd_extension, spectrum);
                    for (int j = 0; j < my; ++j)
                        values[index(i, j, k)] = transformed[static_cast<std::size_t>(j)];
                }
            }
            return;
        }
        for (int j = 0; j < my; ++j) {
            for (int i = 0; i < mx; ++i) {
                for (int k = 0; k < mz; ++k)
                    line[static_cast<std::size_t>(k)] = values[index(i, j, k)];
                apply_orthonormal_dst1_line(
                    line, transformed, fft, odd_extension, spectrum);
                for (int k = 0; k < mz; ++k)
                    values[index(i, j, k)] = transformed[static_cast<std::size_t>(k)];
            }
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

LaplaceFftBulkSolverZfft3D::LaplaceFftBulkSolverZfft3D(
    CartesianGrid3D grid, ZfftBcType bc, double eta, int order)
    : grid_(std::move(grid)), bc_(bc), eta_(eta), order_(order)
{
    if (order_ != 2 && order_ != 4)
        throw std::invalid_argument("order must be 2 or 4");

    double dx = grid_.spacing()[0], dy = grid_.spacing()[1], dz = grid_.spacing()[2];
    if (std::fabs(dx - dy) > 1e-12 * dx || std::fabs(dx - dz) > 1e-12 * dx)
        throw std::invalid_argument(
            "LaplaceFftBulkSolverZfft3D requires square cells (dx == dy == dz)");

    const auto dims = grid_.dof_dims();
    const int intervals_x = bc_ == ZfftBcType::Periodic ? dims[0] : dims[0] - 1;
    const int intervals_y = bc_ == ZfftBcType::Periodic ? dims[1] : dims[1] - 1;
    const int intervals_z = bc_ == ZfftBcType::Periodic ? dims[2] : dims[2] - 1;
    const bool zfft_safe = is_power_of_two(intervals_x)
                        && is_power_of_two(intervals_y)
                        && is_power_of_two(intervals_z);
    if (!zfft_safe) {
        if (bc_ != ZfftBcType::Dirichlet || order_ != 2) {
            throw std::invalid_argument(
                "non-power-of-two 3D grids require second-order Dirichlet "
                "conditions; zFFT is not invoked because it can hang");
        }
        if (intervals_x < 2 || intervals_y < 2 || intervals_z < 2)
            throw std::invalid_argument(
                "Dirichlet DST fallback requires at least two intervals per axis");

        use_dirichlet_dst_ = true;
        eigenvalues_x_ = make_dirichlet_eigenvalues(intervals_x);
        eigenvalues_y_ = make_dirichlet_eigenvalues(intervals_y);
        eigenvalues_z_ = make_dirichlet_eigenvalues(intervals_z);
    }
}

void LaplaceFftBulkSolverZfft3D::solve_dirichlet_dst(
    const Eigen::VectorXd& rhs,
    Eigen::VectorXd& solution) const
{
    const auto dims = grid_.dof_dims();
    const int nx = dims[0];
    const int ny = dims[1];
    const int nz = dims[2];
    const int mx = nx - 2;
    const int my = ny - 2;
    const int mz = nz - 2;
    const int nxy = nx * ny;
    const double h = grid_.spacing()[0];
    const double h2 = h * h;
    const double eta_z = h2 * eta_;

    std::vector<double> coefficients(
        static_cast<std::size_t>(mx * my * mz), 0.0);
    const auto interior_index = [mx, my](int i, int j, int k) {
        return (k * my + j) * mx + i;
    };
    for (int k = 0; k < mz; ++k) {
        for (int j = 0; j < my; ++j) {
            for (int i = 0; i < mx; ++i) {
                coefficients[static_cast<std::size_t>(interior_index(i, j, k))]
                    = -h2 * rhs[(k + 1) * nxy + (j + 1) * nx + (i + 1)];
            }
        }
    }

    // Tensor-product orthonormal DST-I.  Each one-dimensional sine transform
    // is evaluated through an odd extension and an arbitrary-length FFT.
    Eigen::FFT<double> fft;
    apply_orthonormal_dst1_axis(coefficients, mx, my, mz, 0, fft);
    apply_orthonormal_dst1_axis(coefficients, mx, my, mz, 1, fft);
    apply_orthonormal_dst1_axis(coefficients, mx, my, mz, 2, fft);

    for (int k = 0; k < mz; ++k) {
        for (int j = 0; j < my; ++j) {
            for (int i = 0; i < mx; ++i) {
                const double denominator = eigenvalues_x_[i]
                                         + eigenvalues_y_[j]
                                         + eigenvalues_z_[k]
                                         + eta_z;
                if (std::abs(denominator) <=
                    64.0 * std::numeric_limits<double>::epsilon()) {
                    throw std::runtime_error(
                        "Dirichlet DST fallback encountered a zero eigenvalue");
                }
                coefficients[static_cast<std::size_t>(interior_index(i, j, k))]
                    /= denominator;
            }
        }
    }

    // The orthonormal DST-I is symmetric, hence its inverse is the same
    // transform.  Axis transforms commute.
    apply_orthonormal_dst1_axis(coefficients, mx, my, mz, 2, fft);
    apply_orthonormal_dst1_axis(coefficients, mx, my, mz, 1, fft);
    apply_orthonormal_dst1_axis(coefficients, mx, my, mz, 0, fft);

    solution = Eigen::VectorXd::Zero(nx * ny * nz);
    for (int k = 0; k < mz; ++k) {
        for (int j = 0; j < my; ++j) {
            for (int i = 0; i < mx; ++i) {
                solution[(k + 1) * nxy + (j + 1) * nx + (i + 1)] =
                    coefficients[static_cast<std::size_t>(interior_index(i, j, k))];
            }
        }
    }
}

// ---------------------------------------------------------------------------
// solve() — dispatch based on BC type.
//
// Sign/scaling convention (same as 2D):
//   zfft solves  (-stencil + eta_z) u = f_z
//   Physical input satisfies (Delta_h - eta) u = rhs_phys.
//   Multiplication by -h^2 gives the zfft system, so
//   f_z = -h^2 * rhs_phys and eta_z = h^2 * eta.
//
// TensorXd layout: t[k][j][i] — k=z (slowest), j=y, i=x (fastest).
//   dim1 = nz-size, dim2 = ny-size, dim3 = nx-size.
//   FastDiffusionSolver3d uses n1 = dim[0]-1 intervals in z, etc.
//
// Flat index (matches CartesianGrid3D::index):  k*(ny_d*nx_d) + j*nx_d + i
//
// Periodic  — CellCenter DOFs: dof_dims = {nx, ny, nz}.
//             TensorXd(nz+1, ny+1, nx+1); data at t[k][j][i] for k<nz,j<ny,i<nx.
//
// Dirichlet — Node DOFs: dof_dims = {nx+1, ny+1, nz+1} for nx cells.
//             TensorXd(nz+1, ny+1, nx+1) where each +1 is the node count.
//             Interior: t[k][j][i] for k=1..nz-1, j=1..ny-1, i=1..nx-1.
//
// Neumann   — same layout as Dirichlet; all nodes filled.
// ---------------------------------------------------------------------------
void LaplaceFftBulkSolverZfft3D::solve(const Eigen::VectorXd& rhs,
                                        Eigen::VectorXd&       solution) const
{
    if (rhs.size() != grid_.num_dofs())
        throw std::invalid_argument(
            "LaplaceFftBulkSolverZfft3D RHS size does not match the grid");
    if (use_dirichlet_dst_) {
        solve_dirichlet_dst(rhs, solution);
        return;
    }

    auto d   = grid_.dof_dims();
    int  nx  = d[0], ny = d[1], nz = d[2];
    int  nxy = nx * ny;
    double h   = grid_.spacing()[0];
    double h2  = h * h;
    double eta_z = h2 * eta_;

    if (bc_ == ZfftBcType::Periodic) {
        // dof_dims = {nx_cells, ny_cells, nz_cells}
        // TensorXd(nz+1, ny+1, nx+1): n1=nz, n2=ny, n3=nx intervals
        TensorXd mat(nz + 1, ny + 1, nx + 1);
        mat.fill(0.0);

        for (int k = 0; k < nz; ++k)
            for (int j = 0; j < ny; ++j)
                for (int i = 0; i < nx; ++i)
                    mat[k][j][i] = -h2 * rhs[k * nxy + j * nx + i];

        zfft::FastDiffusionSolver3d(mat, eta_z, ZFFT_PERIODIC, order_);

        solution.resize(nx * ny * nz);
        for (int k = 0; k < nz; ++k)
            for (int j = 0; j < ny; ++j)
                for (int i = 0; i < nx; ++i)
                    solution[k * nxy + j * nx + i] = mat[k][j][i];

    } else {
        // dof_dims = {nx_nodes, ny_nodes, nz_nodes}  (includes boundary)
        // TensorXd(nz, ny, nx): n1=nz-1, n2=ny-1, n3=nx-1 intervals
        TensorXd mat(nz, ny, nx);
        mat.fill(0.0);

        if (bc_ == ZfftBcType::Dirichlet) {
            for (int k = 1; k < nz - 1; ++k)
                for (int j = 1; j < ny - 1; ++j)
                    for (int i = 1; i < nx - 1; ++i)
                        mat[k][j][i] = -h2 * rhs[k * nxy + j * nx + i];
        } else {  // Neumann — all nodes
            for (int k = 0; k < nz; ++k)
                for (int j = 0; j < ny; ++j)
                    for (int i = 0; i < nx; ++i)
                        mat[k][j][i] = -h2 * rhs[k * nxy + j * nx + i];
        }

        zfft::FastDiffusionSolver3d(mat, eta_z, zfft_bc(bc_), order_);

        solution.resize(nx * ny * nz);
        for (int k = 0; k < nz; ++k)
            for (int j = 0; j < ny; ++j)
                for (int i = 0; i < nx; ++i)
                    solution[k * nxy + j * nx + i] = mat[k][j][i];
    }
}

} // namespace kfbim
