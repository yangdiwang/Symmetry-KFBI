#pragma once

#ifdef KFBIM_HAS_FFTW3

#include "fft_engine.hpp"
#include <fftw3.h>
#include <vector>
#include <complex>

namespace kfbim {

// FFTW3 implementation of IFFTEngine2D.
// Plans are created at construction with FFTW_ESTIMATE.
// Internal buffers absorb any aliasing between the user's arrays and FFTW's.
class FFTW3Engine2D : public IFFTEngine2D {
public:
    FFTW3Engine2D(int nx, int ny);
    ~FFTW3Engine2D() override;

    FFTW3Engine2D(const FFTW3Engine2D&)            = delete;
    FFTW3Engine2D& operator=(const FFTW3Engine2D&) = delete;

    void forward(const double*               in,
                 std::complex<double>*       out) override;

    void backward(const std::complex<double>* in,
                  double*                     out) override;

    int nx() const override { return nx_; }
    int ny() const override { return ny_; }

private:
    int                       nx_, ny_;
    double*       buf_r_;   // fftw_alloc_real(nx*ny)
    fftw_complex* buf_c_;   // fftw_alloc_complex(nx*(ny/2+1))
    fftw_plan                 plan_fwd_;
    fftw_plan                 plan_bwd_;
};

} // namespace kfbim

#endif // KFBIM_HAS_FFTW3
