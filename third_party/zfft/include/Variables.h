#pragma once
// Minimal Variables shim: provides MatrixXd and TensorXd used by zfft.
#include "MyMatrix.H"
#include "MyTensor.H"

typedef Matrix<double> MatrixXd;
typedef Matrix<float>  MatrixXf;
typedef Matrix<int>    MatrixXi;

typedef Tensor<double> TensorXd;
typedef Tensor<int>    TensorXi;
