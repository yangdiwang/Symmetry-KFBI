#ifdef KFBIM_HAS_FFTW3

#include "fftw3_engine.hpp"
#include <algorithm>
#include <cstring>

namespace kfbim {

// Data layout: flat index = j*nx+i  (y-outer, x-inner), matching CartesianGrid.
// FFTW r2c_2d(n0, n1) treats n0 as the outer dim and n1 as the inner dim, so
// we pass n0=ny, n1=nx.  Complex output has shape ny*(nx/2+1).
FFTW3Engine2D::FFTW3Engine2D(int nx, int ny)
    : nx_(nx), ny_(ny)
{
    int nc = nx / 2 + 1;
    buf_r_ = fftw_alloc_real(ny * nx);
    buf_c_ = fftw_alloc_complex(ny * nc);

    plan_fwd_ = fftw_plan_dft_r2c_2d(ny, nx, buf_r_, buf_c_, FFTW_ESTIMATE);
    plan_bwd_ = fftw_plan_dft_c2r_2d(ny, nx, buf_c_, buf_r_, FFTW_ESTIMATE);
}

FFTW3Engine2D::~FFTW3Engine2D()
{
    fftw_destroy_plan(plan_fwd_);
    fftw_destroy_plan(plan_bwd_);
    fftw_free(buf_r_);
    fftw_free(buf_c_);
}

void FFTW3Engine2D::forward(const double* in, std::complex<double>* out)
{
    int nc = nx_ / 2 + 1;
    std::copy(in, in + ny_ * nx_, buf_r_);
    fftw_execute(plan_fwd_);
    // fftw_complex and std::complex<double> are layout-compatible (C99 §7.3.1)
    std::memcpy(out, buf_c_, ny_ * nc * sizeof(fftw_complex));
}

void FFTW3Engine2D::backward(const std::complex<double>* in, double* out)
{
    int nc = nx_ / 2 + 1;
    std::memcpy(buf_c_, in, ny_ * nc * sizeof(fftw_complex));
    fftw_execute(plan_bwd_);
    std::copy(buf_r_, buf_r_ + ny_ * nx_, out);
}

} // namespace kfbim

#endif // KFBIM_HAS_FFTW3
