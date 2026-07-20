/*=============================================================================
*
*   Filename : zfft.h
*   Creator : Han Zhou
*   Date : 10/16/23
*
=============================================================================*/

#ifndef _ZFFT_H
#define _ZFFT_H

#include <complex>
#include "Variables.h"

#define ZFFT_DIRICHLET 0
#define ZFFT_NEUMANN   1
#define ZFFT_PERIODIC  2

#define ZFFT_TWO  2
#define ZFFT_FOUR 4

#define ZFFT_MAC_E1 1
#define ZFFT_MAC_E2 2
#define ZFFT_MAC_E3 3
#define ZFFT_MAC_C  4

typedef std::complex<double> Complex;

namespace zfft {

extern void initialize(int level = 16);
extern void finalize(void);

// 1-D complex FFT/IFFT (recursive)
extern void fft(const Complex*, Complex*, int);
extern void ifft(const Complex*, Complex*, int);

// 1-D FFT/IFFT on split real/imaginary arrays
extern void fft(const double f_r[], const double f_i[],
                double c_r[], double c_i[], int n);
extern void ifft(const double f_r[], const double f_i[],
                 double c_r[], double c_i[], int n);

extern void fft_c(const double f_r[], const double f_i[],
                  double c_r[], double c_i[], int n);
extern void ifft_c(const double f_r[], const double f_i[],
                   double c_r[], double c_i[], int n);

// 1-D DST / DCT
extern void dst(const double u[], double c[], int m);
extern void idst(const double u[], double c[], int m);
extern void dct(const double u[], double c[], int m);
extern void idct(const double u[], double c[], int m);

// 2-D transforms
extern void fft2d(MatrixXd &unknown_r, MatrixXd &unknown_i);
extern void ifft2d(MatrixXd &unknown_r, MatrixXd &unknown_i);
extern void dst2d(MatrixXd &unknown);
extern void idst2d(MatrixXd &unknown);
extern void dct2d(MatrixXd &unknown);
extern void idct2d(MatrixXd &unknown);

// 3-D transforms
extern void fft3d(TensorXd &unknown_r, TensorXd &unknown_i);
extern void ifft3d(TensorXd &unknown_r, TensorXd &unknown_i);
extern void dst3d(TensorXd &unknown);
extern void idst3d(TensorXd &unknown);
extern void dct3d(TensorXd &unknown);
extern void idct3d(TensorXd &unknown);

// ---------------------------------------------------------------------------
// Fast Poisson / Helmholtz solvers
//
// Solves (eta*I - Delta_h) u = f on a uniform Cartesian grid.
// Delta_h is the *dimensionless* stencil (no 1/h^2 factor); the caller is
// responsible for scaling the RHS: f_zfft = h^2 * f_physical when h = dx = dy.
//
// bc    : ZFFT_DIRICHLET, ZFFT_NEUMANN, or ZFFT_PERIODIC
// order : ZFFT_TWO (2nd order) or ZFFT_FOUR (4th order compact)
// ---------------------------------------------------------------------------

// Thomas-accelerated variant (mixed DST/Thomas in one direction)
extern void FastDiffusionSolver2d0(MatrixXd &unknown, double eta, int bc, int order);
extern void FastDiffusionSolver3d0(TensorXd &unknown, double eta, int bc, int order);

// Pure FFT/DST/DCT variant
extern void FastDiffusionSolver2d(MatrixXd &unknown, double eta, int bc, int order);
extern void FastDiffusionSolver3d(TensorXd &unknown, double eta, int bc, int order);

// MAC-staggered variants (for Stokes)
extern void FastDiffusionSolver2d_HDVN(MatrixXd &unknown, double eta, int ord);
extern void FastDiffusionSolver2d_c(MatrixXd&, double, int, int, int);
extern void FastDiffusionSolver2d_e1(MatrixXd&, double, int, int, int);
extern void FastDiffusionSolver2d_e2(MatrixXd&, double, int, int, int);

} // namespace zfft

#endif // _ZFFT_H
