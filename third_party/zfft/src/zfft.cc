/*=============================================================================
*   
*   Filename : zfft.cc
*   Creator : Han Zhou
*   Date : 10/16/23
*   Description : 
*
=============================================================================*/
   
#include <cmath>
#include <cassert>
#include <iostream>

#ifdef _OPENMP
#include <omp.h>
#endif

#include "zfft.h"
#include "Tridiag.h"
#include "MathTools.h"

//+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
namespace zfft	{	// functions start with namespace zfft
//+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

static bool zfft_initialized = false;

static int MAX_LEVEL = 16;

static double **pre_zr 	= 0;
static double **pre_zi1 = 0;
static double **pre_zi2 = 0;

//:::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
void initialize(int level)
{
	MAX_LEVEL = level;

	pre_zr  = new double*[MAX_LEVEL+1];
	pre_zi1 = new double*[MAX_LEVEL+1];
	pre_zi2 = new double*[MAX_LEVEL+1];

	for(int l = 0; l <= MAX_LEVEL; l++){
		int N = 2 << l;
		pre_zr [l] = new double[N];
		pre_zi1[l] = new double[N];
		pre_zi2[l] = new double[N];

		if (l == 0) {
			pre_zr [l][0] = 1.0;
			pre_zi1[l][0] = 0.0;
			pre_zi2[l][0] = 0.0;
		} else {
			int l_1 = l-1;
			for(int i = 0, i0 = 0; i < N; i += 2, i0++){
				pre_zr [l][i] = pre_zr [l_1][i0];
				pre_zi1[l][i] = pre_zi1[l_1][i0];
				pre_zi2[l][i] = pre_zi2[l_1][i0];
			}
			double h = (M_PI+M_PI)/N;
			double alpha, cs, sn;
			for(int i = 1; i < N; i += 2){
				alpha = i*h;
				cs = cos(alpha);
				sn = sin(alpha);
				pre_zr [l][i] =  cs;
				pre_zi1[l][i] = -sn;
				pre_zi2[l][i] =  sn;
			}
		}

	}

	zfft_initialized = true;
}

//:::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
void finalize(void)
{
	if (pre_zr != 0) {
		for(int l = 0; l <= MAX_LEVEL; l++){
			delete[] pre_zr[l];
			pre_zr[l] = 0;
		}
		delete[] pre_zr;
		pre_zr = 0;
	}
	if (pre_zi1 != 0) {
		for(int l = 0; l <= MAX_LEVEL; l++){
			delete[] pre_zi1[l];
			pre_zi1[l] = 0;
		}
		delete[] pre_zi1;
		pre_zi1 = 0;
	}
	if (pre_zi2 != 0) {
		for(int l = 0; l <= MAX_LEVEL; l++){
			delete[] pre_zi2[l];
			pre_zi2[l] = 0;
		}
		delete[] pre_zi2;
		pre_zi2 = 0;
	}

	zfft_initialized = false;
}

//:::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
void computeFastFourierTransform(const double z_r[], 
                                 const double z_i[], 
                                 const double f_r[], 
                                 const double f_i[], 
                                 double c_r[], 
                                 double c_i[], 
                                 int n) 
{
  for (int i = 0; i < n; i++) {
    c_r[i] = f_r[i];
    c_i[i] = f_i[i]; 
  }

  int j = 0;
  int i2 = n >> 1; 
  for (int i = 0; i < n - 1; i++) {
    if (i < j) {
      double tr = c_r[i];
      double ti = c_i[i]; 
      c_r[i] = c_r[j]; 
      c_i[i] = c_i[j]; 
      c_r[j] = tr;
      c_i[j] = ti; 
    }
    int k = i2;
    while (k <= j) {
      j -= k; 
      k >>= 1; 
    }
    j += k;
  }

	int m = Log2(n);

  int q = 1;
  int p = q << 1;
  int s = n >> 1; 
  for (int k = 0; k < m; k++) {
    for (int i = 0; i < n; i += p) {
      int j0 = i, j1 = i + q;
      double tmp_r = c_r[j1];
      double tmp_i = c_i[j1];
      c_r[j1] = c_r[j0] - tmp_r; 
      c_i[j1] = c_i[j0] - tmp_i; 
      c_r[j0] += tmp_r;
      c_i[j0] += tmp_i;
      j0++;  j1++; 
      for (int r = 1, t = s; r < q; r++, j0++, j1++, t += s) {
        double tmp_r = c_r[j1] * z_r[t] - c_i[j1] * z_i[t]; 
        double tmp_i = c_r[j1] * z_i[t] + c_i[j1] * z_r[t]; 
        c_r[j1] = c_r[j0] - tmp_r;
        c_i[j1] = c_i[j0] - tmp_i;
        c_r[j0] += tmp_r; 
        c_i[j0] += tmp_i; 
      }
    }
    q = p;  p = q << 1;  s = s >> 1; 
  }
}

//:::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
void fft(const double f_r[], 
				 const double f_i[],
         double c_r[], 
				 double c_i[], 
         int n) 
{
	if (zfft_initialized) {

  	int n2 = n >> 1; 
		int L = Log2(n2);

  	computeFastFourierTransform(pre_zr[L], pre_zi1[L], f_r, f_i, c_r, c_i, n);

		double deno_r = 1.0 / n;
		for(int i = 0; i < n; i++){
			c_r[i] *= deno_r;
			c_i[i] *= deno_r;
		}

	} else {

  	int n2 = n >> 1; 

  	double *z_r = new double[n2];
  	double *z_i = new double[n2];

  	double h = (M_PI + M_PI) / n; 

		double alpha;
		for(int i = 0; i < n2; i++){
			alpha = i*h;
			z_r[i] = cos(alpha);
			z_i[i] = -sin(alpha);
		}

  	computeFastFourierTransform(z_r, z_i, f_r, f_i, c_r, c_i, n);

		double deno_r = 1.0 / n;
		for(int i = 0; i < n; i++){
			c_r[i] *= deno_r;
			c_i[i] *= deno_r;
		}

  	delete[] z_r;
  	delete[] z_i;

	}
}

//:::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
void ifft(const double f_r[], 
					const double f_i[],
          double c_r[], 
					double c_i[], 
          int n) 
{
	if (zfft_initialized) {

  	int n2 = n >> 1; 
		int L = Log2(n2);

  	computeFastFourierTransform(pre_zr[L], pre_zi2[L], f_r, f_i, c_r, c_i, n);

	} else {

  	int n2 = n >> 1; 

  	double *z_r = new double[n2];
  	double *z_i = new double[n2];

  	double h = (M_PI + M_PI) / n; 
		double alpha;

		for(int i = 0; i < n2; i++){
			alpha = i*h;
			z_r[i] = cos(alpha);
			z_i[i] = sin(alpha);
		}

  	computeFastFourierTransform(z_r, z_i, f_r, f_i, c_r, c_i, n);

  	delete[] z_r;
  	delete[] z_i;
	}
}

//:::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
void fft_c(const double f_r[], 
					 const double f_i[],
         	 double c_r[], 
					 double c_i[], 
           int n) 
{
  int m = n + n;

	double *src_r = new double[m];
	double *src_i = new double[m];
	double *dst_r = new double[m];
	double *dst_i = new double[m];

  for (int i = 0, j0 = 0, j1 = 1; i < n; i++, j0 += 2, j1 += 2) {
    src_r[j1] = f_r[i]; 
    src_i[j1] = f_i[i]; 
    src_r[j0] = 0.0; 
    src_i[j0] = 0.0; 
  }

	fft(src_r, src_i, dst_r, dst_i, m);

  for (int i = 0; i < n; i++) {
		c_r[i] = 2.0 * dst_r[i];
		c_i[i] = 2.0 * dst_i[i];
  }

  delete[] src_r; 
  delete[] src_i; 
  delete[] dst_r;
  delete[] dst_i;
}

//:::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
void ifft_c(const double f_r[], 
						const double f_i[],
          	double c_r[], 
						double c_i[], 
            int n) 
{
  int m = n + n; 

	double *src_r = new double[m];
	double *src_i = new double[m];
	double *dst_r = new double[m];
	double *dst_i = new double[m];

  for (int i = 0; i < n; i++) {
    src_r[i] = f_r[i]; 
    src_i[i] = f_i[i]; 
  }
  for (int i = n; i < m; i++) {
    src_r[i] = 0.0; 
    src_i[i] = 0.0; 
  }

	ifft(src_r, src_i, dst_r, dst_i, m);

  for (int i = 0, j0 = 0, j1 = 1; i < n; i++, j0 += 2, j1 += 2) {
    c_r[i] = dst_r[j1];
    c_i[i] = dst_i[j1];
  }

  delete[] src_r; 
  delete[] src_i; 
  delete[] dst_r;
  delete[] dst_i;
}

//:::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
void dst(const double u[], double c[], int m)  // m = 2^k
{
	int n = m << 1;

  double *f_r = new double[n];
  double *f_i = new double[n];
  double *c_r = new double[n]; 
  double *c_i = new double[n];

  f_r[0] = f_i[0] = 0.0;
  for (int i = 0; i < m; i++) {
    f_r[i] = u[i]; 
    f_i[i] = 0.0; 
  }
  for (int i = m; i < n; i++) {
    f_r[i] = f_i[i] = 0.0; 
  }

  fft(f_r, f_i, c_r, c_i, n); 

  double factor = 2.0 * sqrt(n);
  for (int i = 1; i < m; i++) {
     c[i] = c_i[i] * factor; 
  }

  delete[] f_r;
  delete[] f_i;
  delete[] c_r; 
  delete[] c_i; 
}

//:::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
void idst(const double u[], double c[], int m)  // m = 2^k
{
	dst(u, c, m);
}

//:::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
void dct(const double u[], double c[], int m) // m = 2^k + 1
{
  int m1 = m - 1;
  int n = m1 << 1;

  double *f_r = new double[n];
  double *f_i = new double[n];
  double *c_r = new double[n];
  double *c_i = new double[n]; 

  int m12 = m1 >> 1; 

  f_r[0] = u[0] / m1; 
  for (int i = 1; i < m1; i++) {
    f_r[i] = u[i] / m12; 
  }
  f_r[m1] = u[m1] / m1;

  for (int i = m; i < n; i++) {
    f_r[i] = 0.0;
  }

	for(int i = 0; i < n; i++){
		f_i[i] = 0.0;
	}

  fft(f_r, f_i, c_r, c_i, n); 

  for (int i = 0; i < m; i++) {
    c[i] = c_r[i] * n; 
  }

  const double gamma = 0.5 * sqrt(2.0); 

  c[m1] *= gamma; 
  c[0] *= gamma;

  delete[] f_r; 
  delete[] f_i; 
  delete[] c_r;
  delete[] c_i; 
}

//:::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
void idct(const double u[], double c[], int m) // m = 2^k + 1
{
  int m1 = m - 1;
  int n = m1 << 1;

  double *f_r = new double[n]; 
  double *f_i = new double[n]; 
  double *c_r = new double[n];
  double *c_i = new double[n]; 

  const double gamma = 0.5 * sqrt(2.0);

  for (int i = 0; i < m; i++) {
    f_r[i] = u[i];
  }
  f_r[0] *= gamma;
  f_r[m1] *= gamma;
  for (int i = m; i < n; i++) {
    f_r[i] = 0.0;
  }

	for(int i = 0; i < n; i++){
		f_i[i] = 0.0;
	}

  fft(f_r, f_i, c_r, c_i, n);

  for (int i = 0; i < m; i++) {
    c[i] = c_r[i] * n; 
  }

  delete[] f_r;
  delete[] f_i;
  delete[] c_r; 
  delete[] c_i;
}

//+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
//:::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
void fft2d(MatrixXd &unknown_r, MatrixXd &unknown_i)
{
	int n1 = unknown_r.rows() - 1;
	int n2 = unknown_r.cols() - 1;

#pragma omp parallel for 
	for(int i = 0; i < n1; i++){
		double *f_r = new double[n2];
		double *f_i = new double[n2];
		double *g_r = new double[n2];
		double *g_i = new double[n2];
		for(int j = 0; j < n2; j++){
			f_r[j] = unknown_r[i][j];
			f_i[j] = unknown_i[i][j];
		}
		fft(f_r, f_i, g_r, g_i, n2);
		for(int j = 0; j < n2; j++){
			unknown_r[i][j] = g_r[j];
			unknown_i[i][j] = g_i[j];
		}
		delete[] f_r;
		delete[] f_i;
		delete[] g_r;
		delete[] g_i;
	}

#pragma omp parallel for 
	for(int j = 0; j < n2; j++){
		double *f_r = new double[n1];
		double *f_i = new double[n1];
		double *g_r = new double[n1];
		double *g_i = new double[n1];
		for(int i = 0; i < n1; i++){
			f_r[i] = unknown_r[i][j];
			f_i[i] = unknown_i[i][j];
		}
		fft(f_r, f_i, g_r, g_i, n1);
		for(int i = 0; i < n1; i++){
			unknown_r[i][j] = g_r[i];
			unknown_i[i][j] = g_i[i];
		}
		delete[] f_r;
		delete[] f_i;
		delete[] g_r;
		delete[] g_i;
	}

}

//:::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
void ifft2d(MatrixXd &unknown_r, MatrixXd &unknown_i)
{
	int n1 = unknown_r.rows() - 1;
	int n2 = unknown_r.cols() - 1;

#pragma omp parallel for 
	for(int i = 0; i < n1; i++){
		double *f_r = new double[n2];
		double *f_i = new double[n2];
		double *g_r = new double[n2];
		double *g_i = new double[n2];
		for(int j = 0; j < n2; j++){
			f_r[j] = unknown_r[i][j];
			f_i[j] = unknown_i[i][j];
		}
		ifft(f_r, f_i, g_r, g_i, n2);
		for(int j = 0; j < n2; j++){
			unknown_r[i][j] = g_r[j];
			unknown_i[i][j] = g_i[j];
		}
		delete[] f_r;
		delete[] f_i;
		delete[] g_r;
		delete[] g_i;
	}

#pragma omp parallel for 
	for(int j = 0; j < n2; j++){
		double *f_r = new double[n1];
		double *f_i = new double[n1];
		double *g_r = new double[n1];
		double *g_i = new double[n1];
		for(int i = 0; i < n1; i++){
			f_r[i] = unknown_r[i][j];
			f_i[i] = unknown_i[i][j];
		}
		ifft(f_r, f_i, g_r, g_i, n1);
		for(int i = 0; i < n1; i++){
			unknown_r[i][j] = g_r[i];
			unknown_i[i][j] = g_i[i];
		}
		delete[] f_r;
		delete[] f_i;
		delete[] g_r;
		delete[] g_i;
	}

}

//:::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
void dst2d(MatrixXd &unknown)
{
	int n1 = unknown.rows() - 1;
	int n2 = unknown.cols() - 1;

#pragma omp parallel for 
	for(int i = 1; i < n1; i++){
		double *f = new double[n2];
		double *g = new double[n2];
		for(int j = 1; j < n2; j++){
			f[j] = unknown[i][j];
		}
		dst(f, g, n2);
		for(int j = 1; j < n2; j++){
			unknown[i][j] = g[j];
		}
		delete[] f;
		delete[] g;
	}

#pragma omp parallel for 
	for(int j = 1; j < n2; j++){
		double *f = new double[n1];
		double *g = new double[n1];
		for(int i = 1; i < n1; i++){
			f[i] = unknown[i][j];
		}
		dst(f, g, n1);
		for(int i = 1; i < n1; i++){
			unknown[i][j] = g[i];
		}
		delete[] f;
		delete[] g;
	}

}

//:::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
void idst2d(MatrixXd &unknown)
{
	int n1 = unknown.rows() - 1;
	int n2 = unknown.cols() - 1;

#pragma omp parallel for 
	for(int i = 1; i < n1; i++){
		double *f = new double[n2];
		double *g = new double[n2];
		for(int j = 1; j < n2; j++){
			f[j] = unknown[i][j];
		}
		idst(f, g, n2);
		for(int j = 1; j < n2; j++){
			unknown[i][j] = g[j];
		}
		delete[] f;
		delete[] g;
	}

#pragma omp parallel for 
	for(int j = 1; j < n2; j++){
		double *f = new double[n1];
		double *g = new double[n1];
		for(int i = 1; i < n1; i++){
			f[i] = unknown[i][j];
		}
		idst(f, g, n1);
		for(int i = 1; i < n1; i++){
			unknown[i][j] = g[i];
		}
		delete[] f;
		delete[] g;
	}

}

//:::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
void dct2d(MatrixXd &unknown)
{
	int n1 = unknown.rows() - 1;
	int n2 = unknown.cols() - 1;

#pragma omp parallel for 
	for(int i = 0; i <= n1; i++){
		double *f = new double[n2+1];
		double *g = new double[n2+1];
		for(int j = 0; j <= n2; j++){
			f[j] = unknown[i][j];
		}
		dct(f, g, n2+1);
		for(int j = 0; j <= n2; j++){
			unknown[i][j] = g[j];
		}
		delete[] f;
		delete[] g;
	}

#pragma omp parallel for 
	for(int j = 0; j <= n2; j++){
		double *f = new double[n1+1];
		double *g = new double[n1+1];
		for(int i = 0; i <= n1; i++){
			f[i] = unknown[i][j];
		}
		dct(f, g, n1+1);
		for(int i = 0; i <= n1; i++){
			unknown[i][j] = g[i];
		}
		delete[] f;
		delete[] g;
	}

}

//:::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
void idct2d(MatrixXd &unknown)
{
	int n1 = unknown.rows() - 1;
	int n2 = unknown.cols() - 1;

#pragma omp parallel for 
	for(int i = 0; i <= n1; i++){
		double *f = new double[n2+1];
		double *g = new double[n2+1];
		for(int j = 0; j <= n2; j++){
			f[j] = unknown[i][j];
		}
		idct(f, g, n2+1);
		for(int j = 0; j <= n2; j++){
			unknown[i][j] = g[j];
		}
		delete[] f;
		delete[] g;
	}

#pragma omp parallel for 
	for(int j = 0; j <= n2; j++){
		double *f = new double[n1+1];
		double *g = new double[n1+1];
		for(int i = 0; i <= n1; i++){
			f[i] = unknown[i][j];
		}
		idct(f, g, n1+1);
		for(int i = 0; i <= n1; i++){
			unknown[i][j] = g[i];
		}
		delete[] f;
		delete[] g;
	}

}

//+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
//:::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
void fft3d(TensorXd &unknown_r, TensorXd &unknown_i)
{

	int dim[3];
	unknown_r.get_dim(dim);

	int n1 = dim[0]-1;
	int n2 = dim[1]-1;
	int n3 = dim[2]-1;

#pragma omp parallel for collapse(2)
	for(int i = 0; i < n1; i++){
		for(int j = 0; j < n2; j++){

			double *f_r = new double[n3];
			double *f_i = new double[n3];
			double *g_r = new double[n3];
			double *g_i = new double[n3];
			for(int k = 0; k < n3; k++){
				f_r[k] = unknown_r[i][j][k];
				f_i[k] = unknown_i[i][j][k];
			}
			fft(f_r, f_i, g_r, g_i, n3);
			for(int k = 0; k < n3; k++){
				unknown_r[i][j][k] = g_r[k];
				unknown_i[i][j][k] = g_i[k];
			}

			delete[] f_r;
			delete[] f_i;
			delete[] g_r;
			delete[] g_i;

		}
	}


#pragma omp parallel for collapse(2)
	for(int j = 0; j < n2; j++){
		for(int k = 0; k < n3; k++){

			double *f_r = new double[n1];
			double *f_i = new double[n1];
			double *g_r = new double[n1];
			double *g_i = new double[n1];
			for(int i = 0; i < n1; i++){
				f_r[i] = unknown_r[i][j][k];
				f_i[i] = unknown_i[i][j][k];
			}
			fft(f_r, f_i, g_r, g_i, n1);
			for(int i = 0; i < n1; i++){
				unknown_r[i][j][k] = g_r[i];
				unknown_i[i][j][k] = g_i[i];
			}
			delete[] f_r;
			delete[] f_i;
			delete[] g_r;
			delete[] g_i;

		}
	}

#pragma omp parallel for collapse(2)
	for(int k = 0; k < n3; k++){
		for(int i = 0; i < n1; i++){

			double *f_r = new double[n2];
			double *f_i = new double[n2];
			double *g_r = new double[n2];
			double *g_i = new double[n2];
			for(int j = 0; j < n2; j++){
				f_r[j] = unknown_r[i][j][k];
				f_i[j] = unknown_i[i][j][k];
			}
			fft(f_r, f_i, g_r, g_i, n2);
			for(int j = 0; j < n2; j++){
				unknown_r[i][j][k] = g_r[j];
				unknown_i[i][j][k] = g_i[j];
			}
			delete[] f_r;
			delete[] f_i;
			delete[] g_r;
			delete[] g_i;

		}
	}

}


//:::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
void ifft3d(TensorXd &unknown_r, TensorXd &unknown_i)
{

	int dim[3];
	unknown_r.get_dim(dim);

	int n1 = dim[0]-1;
	int n2 = dim[1]-1;
	int n3 = dim[2]-1;

#pragma omp parallel for collapse(2)
	for(int i = 0; i < n1; i++){
		for(int j = 0; j < n2; j++){

			double *f_r = new double[n3];
			double *f_i = new double[n3];
			double *g_r = new double[n3];
			double *g_i = new double[n3];
			for(int k = 0; k < n3; k++){
				f_r[k] = unknown_r[i][j][k];
				f_i[k] = unknown_i[i][j][k];
			}
			ifft(f_r, f_i, g_r, g_i, n3);
			for(int k = 0; k < n3; k++){
				unknown_r[i][j][k] = g_r[k];
				unknown_i[i][j][k] = g_i[k];
			}
			delete[] f_r;
			delete[] f_i;
			delete[] g_r;
			delete[] g_i;

		}
	}


#pragma omp parallel for collapse(2)
	for(int j = 0; j < n2; j++){
		for(int k = 0; k < n3; k++){

			double *f_r = new double[n1];
			double *f_i = new double[n1];
			double *g_r = new double[n1];
			double *g_i = new double[n1];
			for(int i = 0; i < n1; i++){
				f_r[i] = unknown_r[i][j][k];
				f_i[i] = unknown_i[i][j][k];
			}
			ifft(f_r, f_i, g_r, g_i, n1);
			for(int i = 0; i < n1; i++){
				unknown_r[i][j][k] = g_r[i];
				unknown_i[i][j][k] = g_i[i];
			}
			delete[] f_r;
			delete[] f_i;
			delete[] g_r;
			delete[] g_i;

		}
	}

#pragma omp parallel for collapse(2)
	for(int k = 0; k < n3; k++){
		for(int i = 0; i < n1; i++){

			double *f_r = new double[n2];
			double *f_i = new double[n2];
			double *g_r = new double[n2];
			double *g_i = new double[n2];
			for(int j = 0; j < n2; j++){
				f_r[j] = unknown_r[i][j][k];
				f_i[j] = unknown_i[i][j][k];
			}
			ifft(f_r, f_i, g_r, g_i, n2);
			for(int j = 0; j < n2; j++){
				unknown_r[i][j][k] = g_r[j];
				unknown_i[i][j][k] = g_i[j];
			}
			delete[] f_r;
			delete[] f_i;
			delete[] g_r;
			delete[] g_i;

		}
	}

}


//:::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
void dst3d(TensorXd &unknown)
{
	int dim[3];
	unknown.get_dim(dim);

	int n1 = dim[0]-1;
	int n2 = dim[1]-1;
	int n3 = dim[2]-1;

#pragma omp parallel for collapse(2)
	for(int i = 1; i < n1; i++){
		for(int j = 1; j < n2; j++){

			double *f = new double[n3];
			double *g = new double[n3];
			for(int k = 1; k < n3; k++){
				f[k] = unknown[i][j][k];
			}
			dst(f, g, n3);
			for(int k = 1; k < n3; k++){
				unknown[i][j][k] = g[k];
			}
			delete[] f;
			delete[] g;

		}
	}

#pragma omp parallel for collapse(2)
	for(int j = 1; j < n2; j++){
		for(int k = 1; k < n3; k++){

			double *f = new double[n1];
			double *g = new double[n1];
			for(int i = 1; i < n1; i++){
				f[i] = unknown[i][j][k];
			}
			dst(f, g, n1);
			for(int i = 1; i < n1; i++){
				unknown[i][j][k] = g[i];
			}
			delete[] f;
			delete[] g;

		}
	}

#pragma omp parallel for collapse(2)
	for(int k = 1; k < n3; k++){
		for(int i = 1; i < n1; i++){

			double *f = new double[n2];
			double *g = new double[n2];
			for(int j = 1; j < n2; j++){
				f[j] = unknown[i][j][k];
			}
			dst(f, g, n2);
			for(int j = 1; j < n2; j++){
				unknown[i][j][k] = g[j];
			}
			delete[] f;
			delete[] g;

		}
	}

}

//:::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
void idst3d(TensorXd &unknown)
{
	int dim[3];
	unknown.get_dim(dim);

	int n1 = dim[0]-1;
	int n2 = dim[1]-1;
	int n3 = dim[2]-1;

#pragma omp parallel for collapse(2)
	for(int i = 1; i < n1; i++){
		for(int j = 1; j < n2; j++){

			double *f = new double[n3];
			double *g = new double[n3];
			for(int k = 1; k < n3; k++){
				f[k] = unknown[i][j][k];
			}
			idst(f, g, n3);
			for(int k = 1; k < n3; k++){
				unknown[i][j][k] = g[k];
			}
			delete[] f;
			delete[] g;

		}
	}


#pragma omp parallel for collapse(2)
	for(int j = 1; j < n2; j++){
		for(int k = 1; k < n3; k++){

			double *f = new double[n1];
			double *g = new double[n1];
			for(int i = 1; i < n1; i++){
				f[i] = unknown[i][j][k];
			}
			idst(f, g, n1);
			for(int i = 1; i < n1; i++){
				unknown[i][j][k] = g[i];
			}
			delete[] f;
			delete[] g;

		}
	}

#pragma omp parallel for collapse(2)
	for(int k = 1; k < n3; k++){
		for(int i = 1; i < n1; i++){

			double *f = new double[n2];
			double *g = new double[n2];
			for(int j = 1; j < n2; j++){
				f[j] = unknown[i][j][k];
			}
			idst(f, g, n2);
			for(int j = 1; j < n2; j++){
				unknown[i][j][k] = g[j];
			}
			delete[] f;
			delete[] g;

		}
	}

}

//:::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
void dct3d(TensorXd &unknown)
{
	int dim[3];
	unknown.get_dim(dim);

	int n1 = dim[0]-1;
	int n2 = dim[1]-1;
	int n3 = dim[2]-1;

#pragma omp parallel for collapse(2)
	for(int i = 0; i <= n1; i++){
		for(int j = 0; j <= n2; j++){

			double *f = new double[n3+1];
			double *g = new double[n3+1];
			for(int k = 0; k <= n3; k++){
				f[k] = unknown[i][j][k];
			}
			dct(f, g, n3+1);
			for(int k = 0; k <= n3; k++){
				unknown[i][j][k] = g[k];
			}
			delete[] f;
			delete[] g;

		}
	}


#pragma omp parallel for collapse(2)
	for(int j = 0; j <= n2; j++){
		for(int k = 0; k <= n3; k++){

			double *f = new double[n1+1];
			double *g = new double[n1+1];
			for(int i = 0; i <= n1; i++){
				f[i] = unknown[i][j][k];
			}
			dct(f, g, n1+1);
			for(int i = 0; i <= n1; i++){
				unknown[i][j][k] = g[i];
			}
			delete[] f;
			delete[] g;

		}
	}

#pragma omp parallel for collapse(2)
	for(int k = 0; k <= n3; k++){
		for(int i = 0; i <= n1; i++){

			double *f = new double[n2+1];
			double *g = new double[n2+1];
			for(int j = 0; j <= n2; j++){
				f[j] = unknown[i][j][k];
			}
			dct(f, g, n2+1);
			for(int j = 0; j <= n2; j++){
				unknown[i][j][k] = g[j];
			}
			delete[] f;
			delete[] g;

		}
	}

}

//:::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
void idct3d(TensorXd &unknown)
{
	int dim[3];
	unknown.get_dim(dim);

	int n1 = dim[0]-1;
	int n2 = dim[1]-1;
	int n3 = dim[2]-1;

#pragma omp parallel for collapse(2)
	for(int i = 0; i <= n1; i++){
		for(int j = 0; j <= n2; j++){

			double *f = new double[n3+1];
			double *g = new double[n3+1];
			for(int k = 0; k <= n3; k++){
				f[k] = unknown[i][j][k];
			}
			idct(f, g, n3+1);
			for(int k = 0; k <= n3; k++){
				unknown[i][j][k] = g[k];
			}
			delete[] f;
			delete[] g;

		}
	}


#pragma omp parallel for collapse(2)
	for(int j = 0; j <= n2; j++){
		for(int k = 0; k <= n3; k++){

			double *f = new double[n1+1];
			double *g = new double[n1+1];
			for(int i = 0; i <= n1; i++){
				f[i] = unknown[i][j][k];
			}
			idct(f, g, n1+1);
			for(int i = 0; i <= n1; i++){
				unknown[i][j][k] = g[i];
			}
			delete[] f;
			delete[] g;

		}
	}

#pragma omp parallel for collapse(2)
	for(int k = 0; k <= n3; k++){
		for(int i = 0; i <= n1; i++){

			double *f = new double[n2+1];
			double *g = new double[n2+1];
			for(int j = 0; j <= n2; j++){
				f[j] = unknown[i][j][k];
			}
			idct(f, g, n2+1);
			for(int j = 0; j <= n2; j++){
				unknown[i][j][k] = g[j];
			}
			delete[] f;
			delete[] g;

		}
	}

}

//+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
//:::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
void FastDiffusionSolver2d0(MatrixXd &unknown, 
                            double eta, 
													  int bc, 
                            int order)
{
	int n1 = unknown.rows() - 1;
	int n2 = unknown.cols() - 1;

	double C0(0.0), C1(0.0), C2(0.0);

	if (order == ZFFT_TWO) {
		C0 = 4.0 + eta;
		C1 = - 1.0;
		C2 = 0.0;
	} else if (order == ZFFT_FOUR) {
		C0 = 10.0 / 3.0 + 2.0 / 3.0 * eta;
		C1 = - (2.0 / 3.0 - eta / 12.0);
		C2 = - 1.0 / 6.0;
	}

  if (ZFFT_DIRICHLET == bc) {

		double h1(M_PI/n1), h2(M_PI/n2);

    int len1(n1+1), len2(n2+1);
    double *cs_mem = new double[len1+len2];

    double *csi = cs_mem;
    double *csj = csi+len1;

	  for(int i = 0; i <= n1; i++){
	  	csi[i] = cos(i * h1);
	  }
	  for(int j = 0; j <= n2; j++){
	  	csj[j] = cos(j * h2);
	  }

#pragma omp parallel for 
	  for(int i = 1; i < n1; i++){
	  	double *f = new double[n2];
	  	double *g = new double[n2];
	  	for(int j = 1; j < n2; j++){
	  		f[j] = unknown[i][j];
	  	}
	  	dst(f, g, n2);
	  	for(int j = 1; j < n2; j++){
	  		unknown[i][j] = g[j];
	  	}
	  	delete[] f;
	  	delete[] g;
	  }

#pragma omp parallel for 
    for(int j = 1; j < n2; j++){
      double *f = new double[n1-1];
      double *g = new double[n1-1];
      double cs0 = csj[j];
      double diag0 = C0 + 2.0*C1*cs0;
      double diag1 = C1 + 2.0*C2*cs0;
      for(int i = 1, i1 = 0; i < n1; i++, i1++){
        f[i1] = unknown[i][j];
      }
      solveByThomasAlgorithm(diag0, diag1, diag1, f, n1-1, g);
      for(int i = 1, i1 = 0; i < n1; i++, i1++){
        unknown[i][j] = g[i1];
      }
      delete[] f;
      delete[] g;
    }

#pragma omp parallel for 
	  for(int i = 1; i < n1; i++){
	  	double *f = new double[n2];
	  	double *g = new double[n2];
	  	for(int j = 1; j < n2; j++){
	  		f[j] = unknown[i][j];
	  	}
	  	idst(f, g, n2);
	  	for(int j = 1; j < n2; j++){
	  		unknown[i][j] = g[j];
	  	}
	  	delete[] f;
	  	delete[] g;
	  }

    delete[] cs_mem;

  } else if (ZFFT_NEUMANN == bc) {
    std::cout << "not implemented." << std::endl;

  } else if (ZFFT_PERIODIC == bc) {

    MatrixXd &unknown_r = unknown;

	  MatrixXd unknown_i(n1+1, n2+1);
    unknown_i.fill(0.0);

		double h1(2.0*M_PI/n1), h2(2.0*M_PI/n2);

	  double *csi = new double[n1+1];
	  double *csj = new double[n2+1];

	  for(int i = 0; i <= n1; i++){
	  	csi[i] = cos(i * h1);
	  }
	  for(int j = 0; j <= n2; j++){
	  	csj[j] = cos(j * h2);
	  }

#pragma omp parallel for 
	  for(int i = 1; i < n1; i++){
		  double *f_r = new double[n2];
		  double *f_i = new double[n2];
		  double *g_r = new double[n2];
		  double *g_i = new double[n2];
		  for(int j = 0; j < n2; j++){
		  	f_r[j] = unknown_r[i][j];
		  	f_i[j] = unknown_i[i][j];
		  }
		  fft(f_r, f_i, g_r, g_i, n2);
		  for(int j = 0; j < n2; j++){
		  	unknown_r[i][j] = g_r[j];
		  	unknown_i[i][j] = g_i[j];
		  }
		  delete[] f_r;
		  delete[] f_i;
		  delete[] g_r;
		  delete[] g_i;
	  }

#pragma omp parallel for 
    for(int j = 1; j < n2; j++){
		  double *f_r = new double[n1];
		  double *f_i = new double[n1];
		  double *g_r = new double[n1];
		  double *g_i = new double[n1];
      double cs0 = 2.0*csj[j];
      double diag0 = C0 + C1*cs0;
      double diag1 = C1 + C2*cs0;
      for(int i = 0; i < n1; i++){
        f_r[i] = unknown_r[i][j];
        f_i[i] = unknown_i[i][j];
      }
      solveByThomasAlgorithm(diag0, diag1, diag1, diag1, diag1, f_r, n1, g_r);
      solveByThomasAlgorithm(diag0, diag1, diag1, diag1, diag1, f_i, n1, g_i);
      for(int i = 0; i < n1; i++){
        unknown_r[i][j] = g_r[i];
        unknown_i[i][j] = g_i[i];
      }
		  delete[] f_r;
		  delete[] f_i;
		  delete[] g_r;
		  delete[] g_i;
    }

#pragma omp parallel for 
	  for(int i = 1; i < n1; i++){
		  double *f_r = new double[n2];
		  double *f_i = new double[n2];
		  double *g_r = new double[n2];
		  double *g_i = new double[n2];
		  for(int j = 0; j < n2; j++){
		  	f_r[j] = unknown_r[i][j];
		  	f_i[j] = unknown_i[i][j];
		  }
		  ifft(f_r, f_i, g_r, g_i, n2);
		  for(int j = 0; j < n2; j++){
		  	unknown_r[i][j] = g_r[j];
		  	unknown_i[i][j] = g_i[j];
		  }
		  delete[] f_r;
		  delete[] f_i;
		  delete[] g_r;
		  delete[] g_i;
	  }

	  delete[] csi;
	  delete[] csj;
  } 


}

//:::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
void FastDiffusionSolver3d0(TensorXd &unknown, 
                            double eta, 
													  int bc, 
                            int order)
{
	int dim[3];
	unknown.get_dim(dim);

	int n1 = dim[0] - 1;
	int n2 = dim[1] - 1;
	int n3 = dim[2] - 1;

	double C0(0.0), C1(0.0), C2(0.0), C3(0.0);

	if (order == ZFFT_TWO) {
		C0 = 6.0 + eta;
		C1 = - 1.0;
		C2 = 0.0;
		C3 = 0.0;
	} else if (order == ZFFT_FOUR) {
		C0 = 25.0 / 6.0 + 0.5 * eta;
		C1 = - (5.0 - eta) / 12.0;
		C2 = - 1.0 / 8.0;
		C3 = - 1.0 / 48.0;
	}

  if (ZFFT_DIRICHLET == bc) {

		double h1(M_PI/n1), h2(M_PI/n2), h3(M_PI/n3);

    double *cs_mem = new double[n1+1 + n2+1 + n3+1];
	  double *csi = cs_mem;
	  double *csj = csi+n1+1;
	  double *csk = csj+n2+1;

	  for(int i = 0; i <= n1; i++){
	  	csi[i] = cos(i * h1);
	  }
	  for(int j = 0; j <= n2; j++){
	  	csj[j] = cos(j * h2);
	  }
	  for(int k = 0; k <= n3; k++){
	  	csk[k] = cos(k * h3);
	  }

#pragma omp parallel for collapse(2)
	  for(int i = 1; i < n1; i++){
	  	for(int j = 1; j < n2; j++){
        double *work_mem = new double[n3+n3];
	  		double *f = work_mem;
	  		double *g = f+n3;
	  		for(int k = 1; k < n3; k++){
	  			f[k] = unknown[i][j][k];
	  		}
	  		dst(f, g, n3);
	  		for(int k = 1; k < n3; k++){
	  			unknown[i][j][k] = g[k];
	  		}
	  		delete[] work_mem;
	  	}
	  }

#pragma omp parallel for collapse(2)
	  for(int k = 1; k < n3; k++){
	  	for(int i = 1; i < n1; i++){
        double *work_mem = new double[n2+n2];
	  		double *f = work_mem;
	  		double *g = f+n2;
	  		for(int j = 1; j < n2; j++){
	  			f[j] = unknown[i][j][k];
	  		}
	  		dst(f, g, n2);
	  		for(int j = 1; j < n2; j++){
	  			unknown[i][j][k] = g[j];
	  		}
	  		delete[] work_mem;
	  	}
	  }

#pragma omp parallel for collapse(2)
	  for(int j = 1; j < n2; j++){
	  	for(int k = 1; k < n3; k++){
        double *work_mem = new double[n1-1 + n1-1];
	  		double *f = work_mem;
	  		double *g = f+n1-1;
        double temp0 = 2.0*(csj[j]+csk[k]);
        double temp1 = 4.0*csj[j]*csk[k];
        double diag0 = C0 + C1*temp0 + C2*temp1;
        double diag1 = C1 + C2*temp0 + C3*temp1;
        for(int i = 1, i1 = 0; i < n1; i++, i1++){
          f[i1] = unknown[i][j][k];
        }
        solveByThomasAlgorithm(diag0, diag1, diag1, f, n1-1, g);
        for(int i = 1, i1 = 0; i < n1; i++, i1++){
          unknown[i][j][k] = g[i1];
        }
        delete[] work_mem;
      }
    }

#pragma omp parallel for collapse(2)
	  for(int i = 1; i < n1; i++){
	  	for(int j = 1; j < n2; j++){
        double *work_mem = new double[n3+n3];
	  		double *f = work_mem;
	  		double *g = f+n3;
	  		for(int k = 1; k < n3; k++){
	  			f[k] = unknown[i][j][k];
	  		}
	  		idst(f, g, n3);
	  		for(int k = 1; k < n3; k++){
	  			unknown[i][j][k] = g[k];
	  		}
	  		delete[] work_mem;
	  	}
	  }

#pragma omp parallel for collapse(2)
	  for(int k = 1; k < n3; k++){
	  	for(int i = 1; i < n1; i++){
        double *work_mem = new double[n2+n2];
	  		double *f = work_mem;
	  		double *g = f+n2;
	  		for(int j = 1; j < n2; j++){
	  			f[j] = unknown[i][j][k];
	  		}
	  		idst(f, g, n2);
	  		for(int j = 1; j < n2; j++){
	  			unknown[i][j][k] = g[j];
	  		}
	  		delete[] work_mem;
	  	}
	  }

	  delete[] cs_mem;

  } else if (ZFFT_NEUMANN == bc) {
    std::cout << "not implemented." << std::endl;
  } else if (ZFFT_PERIODIC == bc) {
    std::cout << "not implemented." << std::endl;
  } 

}

//:::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
void FastDiffusionSolver2d(MatrixXd &unknown, double eta, int bc, int order)
{
	int n1 = unknown.rows() - 1;
	int n2 = unknown.cols() - 1;

	MatrixXd unknown_i;

	if (bc == ZFFT_DIRICHLET) {
		dst2d(unknown);
	} else if (bc == ZFFT_NEUMANN) {
		dct2d(unknown);
	} else if (bc == ZFFT_PERIODIC) {
		unknown_i.reallocate(n1+1, n2+1);
		unknown_i.fill(0.0);
		fft2d(unknown, unknown_i);
	}

	double h1(0.0), h2(0.0);
	if (bc == ZFFT_DIRICHLET || bc == ZFFT_NEUMANN) {
		h1 = M_PI/n1;
		h2 = M_PI/n2;
	} else if (bc == ZFFT_PERIODIC) {
		h1 = 2.0*M_PI/n1;
		h2 = 2.0*M_PI/n2;
	}

	double *csi = new double[n1+1];
	double *csj = new double[n2+1];

	for(int i = 0; i <= n1; i++){
		csi[i] = cos(i * h1);
	}
	for(int j = 0; j <= n2; j++){
		csj[j] = cos(j * h2);
	}
	
	double C0(0.0), C1(0.0), C2(0.0);
	if (order == ZFFT_TWO) {
		C0 = 4.0 + eta;
		C1 = 2.0;
		C2 = 0.0;
	} else if (order == ZFFT_FOUR) {
		C0 = 10.0 / 3.0 + 2.0 / 3.0 * eta;
		C1 = 2.0 * (2.0 / 3.0 - eta / 12.0);
		C2 = 4.0 * 1.0 / 6.0;
	}

	if (bc == ZFFT_DIRICHLET) {

#pragma omp parallel for collapse(2)
		for(int i = 1; i < n1; i++){
			for(int j = 1; j < n2; j++){

				double cs1 = csi[i];
				double cs2 = csj[j];

				double s1 = cs1 + cs2;
				double s2 = cs1 * cs2;

				double lambda = C0 - (C1 * s1 + C2 * s2);
				unknown[i][j] /= lambda;
				
			}
		}

	} else if (bc == ZFFT_NEUMANN) {

#pragma omp parallel for collapse(2)
		for(int i = 0; i <= n1; i++){
			for(int j = 0; j <= n2; j++){

				double cs1 = csi[i];
				double cs2 = csj[j];

				double s1 = cs1 + cs2;
				double s2 = cs1 * cs2;

				double lambda = C0 - (C1 * s1 + C2 * s2);
				unknown[i][j] /= lambda;
				
			}
		}

		if (fabs(eta) < 1.0e-12) {
			unknown[0][0] = 0.0;
		} 

	} else if (bc == ZFFT_PERIODIC) {

#pragma omp parallel for collapse(2)
		for(int i = 0; i < n1; i++){
			for(int j = 0; j < n2; j++){

				double cs1 = csi[i];
				double cs2 = csj[j];

				double s1 = cs1 + cs2;
				double s2 = cs1 * cs2;

				double lambda = C0 - (C1 * s1 + C2 * s2);

				unknown[i][j] /= lambda;
				unknown_i[i][j] /= lambda;
				
			}
		}

		if (fabs(eta) < 1.0e-12) {
			unknown[0][0] = unknown_i[0][0] = 0.0;
		}

	} 

	delete[] csi;
	delete[] csj;

	if (bc == ZFFT_DIRICHLET) {

		idst2d(unknown);

	} else if (bc == ZFFT_NEUMANN) {

		idct2d(unknown);

	} else if (bc == ZFFT_PERIODIC) {

		ifft2d(unknown, unknown_i);

		for(int i = 0; i <= n1; i++){
			unknown[i][n2] = unknown[i][0];
			unknown[n1][i] = unknown[0][i];
		}
	}

}

//:::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
void FastDiffusionSolver3d(TensorXd &unknown, double eta, int bc, int order)
{
	int dim[3];
	unknown.get_dim(dim);

	int n1 = dim[0]-1;
	int n2 = dim[1]-1;
	int n3 = dim[2]-1;

	TensorXd unknown_i;

	if (bc == ZFFT_DIRICHLET) {
		dst3d(unknown);
	} else if (bc == ZFFT_NEUMANN) {
		dct3d(unknown);
	} else if (bc == ZFFT_PERIODIC) {
		unknown_i.reallocate(dim[0], dim[1], dim[2]);
		unknown_i.fill(0.0);
		fft3d(unknown, unknown_i);
	}

	double h1(0.0), h2(0.0), h3(0.0);

	if (bc == ZFFT_DIRICHLET || bc == ZFFT_NEUMANN) {
		h1 = M_PI/n1;
		h2 = M_PI/n2;
		h3 = M_PI/n3;
	} else if (bc == ZFFT_PERIODIC) {
		h1 = 2.0*M_PI/n1;
		h2 = 2.0*M_PI/n2;
		h3 = 2.0*M_PI/n3;
	}

	double *csi = new double[n1+1];
	double *csj = new double[n2+1];
	double *csk = new double[n3+1];

	for(int i = 0; i <= n1; i++){
		csi[i] = cos(i*h1);
	}
	for(int j = 0; j <= n2; j++){
		csj[j] = cos(j*h2);
	}
	for(int k = 0; k <= n3; k++){
		csk[k] = cos(k*h3);
	}
	
	double C0(0.0), C1(0.0), C2(0.0), C3(0.0);

	if (order == ZFFT_TWO) {
		C0 = 6.0 + eta;
		C1 = 2.0;
		C2 = 0.0;
		C3 = 0.0;
	} else if (order == ZFFT_FOUR) {
		C0 = 25.0 / 6.0 + 0.5 * eta;
		C1 = (5.0 - eta) / 6.0;
		C2 = 0.5;
		C3 = 1.0 / 6.0;
	}

	if (bc == ZFFT_DIRICHLET) {

#pragma omp parallel for collapse(3)
		for(int i = 1; i < n1; i++){
			for(int j = 1; j < n2; j++){
				for(int k = 1; k < n3; k++){

					double cs1 = csi[i];
					double cs2 = csj[j];
					double cs3 = csk[k];

					double s1 = cs1 + cs2 + cs3;
					double s2 = cs1 * cs2 + cs2 * cs3 + cs3 * cs1;
					double s3 = cs1 * cs2 * cs3;

					double lambda = C0 - (C1 * s1 + C2 * s2 + C3 * s3);
					unknown[i][j][k] /= lambda;
				}
			}
		}

	} else if (bc == ZFFT_NEUMANN) {

#pragma omp parallel for collapse(3)
		for(int i = 0; i <= n1; i++){
			for(int j = 0; j <= n2; j++){
				for(int k = 0; k <= n3; k++){

					double cs1 = csi[i];
					double cs2 = csj[j];
					double cs3 = csk[k];

					double s1 = cs1 + cs2 + cs3;
					double s2 = cs1 * cs2 + cs2 * cs3 + cs3 * cs1;
					double s3 = cs1 * cs2 * cs3;

					double lambda = C0 - (C1 * s1 + C2 * s2 + C3 * s3);
					unknown[i][j][k] /= lambda;
				}
			}
		}

		if (fabs(eta) < 1.0e-12) {
			unknown[0][0][0] = 0.0;
		} 

	} else if (bc == ZFFT_PERIODIC) {

#pragma omp parallel for collapse(3)
		for(int i = 0; i < n1; i++){
			for(int j = 0; j < n2; j++){
				for(int k = 0; k < n3; k++){

					double cs1 = csi[i];
					double cs2 = csj[j];
					double cs3 = csk[k];

					double s1 = cs1 + cs2 + cs3;
					double s2 = cs1 * cs2 + cs2 * cs3 + cs3 * cs1;
					double s3 = cs1 * cs2 * cs3;

					double lambda = C0 - (C1 * s1 + C2 * s2 + C3 * s3);

					unknown	 [i][j][k] /= lambda;
					unknown_i[i][j][k] /= lambda;
				}
			}
		}

		if (fabs(eta) < 1.0e-12) {
			unknown[0][0][0] = unknown_i[0][0][0] = 0.0;
		}

	} 

	delete[] csi;
	delete[] csj;
	delete[] csk;

	if (bc == ZFFT_DIRICHLET) {

		idst3d(unknown);

	} else if (bc == ZFFT_NEUMANN) {

		idct3d(unknown);

	} else if (bc == ZFFT_PERIODIC) {

		ifft3d(unknown, unknown_i);

		for(int i = 0; i <= n1; i++){
			for(int j = 0; j <= n2; j++){
				unknown[i][j][n3] = unknown[i][j][0];
				unknown[i][n2][j] = unknown[i][0][j];
				unknown[n1][i][j] = unknown[0][i][j];
			}
		}
	}

}


//+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
}	// functions start with namespace zfft
//+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
