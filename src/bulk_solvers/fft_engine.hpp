#pragma once

#include <complex>

namespace kfbim {

// ---------------------------------------------------------------------------
// Abstract 2D periodic FFT engine.
//
// Data layout convention matches CartesianGrid: flat index = j*nx + i,
// i.e., y (j) is the outer (slow) dimension, x (i) is the inner (fast) one.
//
//   Real buffer:    nx * ny doubles,      indexed [j*nx + i]
//   Complex buffer: ny * (nx/2+1) complex, indexed [j*(nx/2+1) + i]
//
// forward()  — unnormalized real-to-complex DFT
// backward() — unnormalized complex-to-real DFT; LaplaceFftBulkSolver2D
//              handles the 1/(nx*ny) normalization
//
// Implementors own internal workspace; forward/backward may copy into it.
// ---------------------------------------------------------------------------

class IFFTEngine2D {
public:
    virtual ~IFFTEngine2D() = default;

    virtual void forward(const double*               in,
                         std::complex<double>*       out) = 0;

    virtual void backward(const std::complex<double>* in,
                          double*                     out) = 0;

    virtual int nx() const = 0;
    virtual int ny() const = 0;
};

} // namespace kfbim
