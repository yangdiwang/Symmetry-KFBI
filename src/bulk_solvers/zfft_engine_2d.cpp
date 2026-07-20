#include "zfft_engine_2d.hpp"
#include "zfft.h"
#include <algorithm>
#include <cstring>
#include <stdexcept>

namespace kfbim {

static bool is_power_of_two(int n) { return n > 0 && (n & (n - 1)) == 0; }

ZfftEngine2D::ZfftEngine2D(int nx, int ny)
    : nx_(nx), ny_(ny),
      sr_(std::max(nx, ny)), si_(std::max(nx, ny)),
      cr_(std::max(nx, ny)), ci_(std::max(nx, ny)),
      tr_(ny * nx), ti_(ny * nx)
{
    if (!is_power_of_two(nx) || !is_power_of_two(ny))
        throw std::invalid_argument(
            "ZfftEngine2D: nx and ny must be powers of 2 (zfft uses radix-2 FFT)");
    zfft::initialize();
}

ZfftEngine2D::~ZfftEngine2D()
{
    zfft::finalize();
}

// ---------------------------------------------------------------------------
// forward: real-to-complex 2D DFT (unnormalized)
//
// Algorithm: separable 1D FFTs.
//   Step 1: FFT each row (x-direction), store full nx complex values.
//   Step 2: FFT each column (y-direction) for kx = 0..nx/2.
//
// zfft::fft normalizes by 1/n; we multiply back to get the unnormalized DFT.
// ---------------------------------------------------------------------------
void ZfftEngine2D::forward(const double* in, std::complex<double>* out)
{
    int nc = nx_ / 2 + 1;

    // Step 1: row-wise 1D FFT (x-direction)
    for (int j = 0; j < ny_; j++) {
        std::fill(si_.begin(), si_.begin() + nx_, 0.0);
        zfft::fft(in + j * nx_, si_.data(), cr_.data(), ci_.data(), nx_);
        // un-normalize (zfft divides by nx)
        for (int i = 0; i < nx_; i++) {
            tr_[j * nx_ + i] = cr_[i] * nx_;
            ti_[j * nx_ + i] = ci_[i] * nx_;
        }
    }

    // Step 2: column-wise 1D FFT (y-direction), for kx = 0..nc-1 only
    for (int i = 0; i < nc; i++) {
        for (int j = 0; j < ny_; j++) {
            sr_[j] = tr_[j * nx_ + i];
            si_[j] = ti_[j * nx_ + i];
        }
        zfft::fft(sr_.data(), si_.data(), cr_.data(), ci_.data(), ny_);
        for (int j = 0; j < ny_; j++) {
            out[j * nc + i] = std::complex<double>(cr_[j] * ny_, ci_[j] * ny_);
        }
    }
}

// ---------------------------------------------------------------------------
// backward: complex-to-real 2D IDFT (unnormalized)
//
// Algorithm:
//   Step 1: IFFT each column (y-direction), for kx = 0..nc-1.
//   Step 2: Reconstruct missing kx = nc..nx-1 by Hermitian symmetry.
//   Step 3: IFFT each row (x-direction), take real part.
// ---------------------------------------------------------------------------
void ZfftEngine2D::backward(const std::complex<double>* in, double* out)
{
    int nc = nx_ / 2 + 1;

    // Step 1: column-wise 1D IFFT (y-direction)
    for (int i = 0; i < nc; i++) {
        for (int j = 0; j < ny_; j++) {
            sr_[j] = in[j * nc + i].real();
            si_[j] = in[j * nc + i].imag();
        }
        zfft::ifft(sr_.data(), si_.data(), cr_.data(), ci_.data(), ny_);
        for (int j = 0; j < ny_; j++) {
            tr_[j * nx_ + i] = cr_[j];
            ti_[j * nx_ + i] = ci_[j];
        }
    }

    // Step 2: Hermitian symmetry — F[kx] = conj(F[nx-kx]) for real input
    for (int i = nc; i < nx_; i++) {
        int ic = nx_ - i;  // conjugate index
        for (int j = 0; j < ny_; j++) {
            tr_[j * nx_ + i] =  tr_[j * nx_ + ic];
            ti_[j * nx_ + i] = -ti_[j * nx_ + ic];
        }
    }

    // Step 3: row-wise 1D IFFT (x-direction), take real part
    for (int j = 0; j < ny_; j++) {
        zfft::ifft(tr_.data() + j * nx_, ti_.data() + j * nx_,
                   cr_.data(), ci_.data(), nx_);
        for (int i = 0; i < nx_; i++) {
            out[j * nx_ + i] = cr_[i];
        }
    }
}

} // namespace kfbim
