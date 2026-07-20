#pragma once

#include "fft_engine.hpp"
#include <vector>
#include <complex>

namespace kfbim {

// ---------------------------------------------------------------------------
// ZfftEngine2D — implements IFFTEngine2D using Han Zhou's zfft library.
//
// Computes separable 2D FFT via zfft's 1D complex FFT routines.
// Conventions match IFFTEngine2D (and FFTW3Engine2D):
//   forward:  unnormalized real→complex DFT, output shape ny*(nx/2+1)
//   backward: unnormalized complex→real IDFT, output shape ny*nx
//
// REQUIREMENT: nx and ny must be powers of 2 (zfft uses radix-2 FFT).
// Constructor throws std::invalid_argument if this is violated.
// zfft::initialize() is called once at construction.
// ---------------------------------------------------------------------------
class ZfftEngine2D : public IFFTEngine2D {
public:
    ZfftEngine2D(int nx, int ny);
    ~ZfftEngine2D() override;

    ZfftEngine2D(const ZfftEngine2D&)            = delete;
    ZfftEngine2D& operator=(const ZfftEngine2D&) = delete;

    void forward(const double*               in,
                 std::complex<double>*       out) override;

    void backward(const std::complex<double>* in,
                  double*                     out) override;

    int nx() const override { return nx_; }
    int ny() const override { return ny_; }

private:
    int nx_, ny_;
    // scratch buffers reused across calls
    std::vector<double> sr_, si_, cr_, ci_;   // 1D work: size max(nx,ny)
    std::vector<double> tr_, ti_;             // 2D intermediate: size ny*nx
};

} // namespace kfbim
