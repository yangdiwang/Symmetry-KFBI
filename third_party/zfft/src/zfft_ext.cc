/*=============================================================================
*   
*   Filename : zfft_ext.cc
*   Creator : Han Zhou
*   Date : 10/16/23
*   Description : 
*
=============================================================================*/
   
#include <cmath>
#include <cassert>
#include <iostream>
 
#include "zfft.h"
#include "Tridiag.h"

#ifdef _OPENMP
#include <omp.h>
#endif

#ifdef USE_FFTW3
#include <fftw3.h>
#endif

#ifdef USE_EIGEN3
#include <eigen3/Eigen/Dense>
#endif


//+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
namespace zfft	{	// functions start with namespace zfft
//+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

//:::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
void fft(const Complex *u, Complex *f, int n)
{
  if (1 == n) {
    f[0] = u[0]; 
    return;
  } else if (2 == n) {
    f[0] = 0.5 * (u[0] + u[1]); 
    f[1] = 0.5 * (u[0] - u[1]);
    return; 
  }

  int m = n >> 1; 

  Complex *u_even = new Complex[m]; 
  Complex *u_odd 	= new Complex[m];
  Complex *f_even = new Complex[m]; 
  Complex *f_odd 	= new Complex[m];

  for (int i = 0, k = 0; i < m; i++, k += 2) {
    u_even[i] = u[k]; 
  }
  for (int i = 0, k = 1; i < m; i++, k += 2) {
    u_odd[i] = u[k]; 
  }

  fft(u_even, f_even, m); 
  fft(u_odd, f_odd, m); 

  double h = (M_PI + M_PI) / n;
  Complex omega(cos(h), -sin(h));

	Complex z, tmp;
  for (int i = 0, k = m; i < m; i++, k++) {
		Complex z = Complex(cos(i*h), -sin(i*h));
		tmp = z * f_odd[i];
    f[i] = 0.5 * (f_even[i] + tmp);
    f[k] = 0.5 * (f_even[i] - tmp); 
  }

  delete[] f_even;
  delete[] f_odd; 
  delete[] u_even;
  delete[] u_odd; 
}

//:::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
void ifft(const Complex *u, Complex *f, int n)
{
  if (1 == n) {
    f[0] = u[0];
    return; 
  } else if (2 == n) {
    f[0] = u[0] + u[1]; 
    f[1] = u[0] - u[1];
    return; 
  }

  int m = n >> 1; 

  Complex *u_even = new Complex[m]; 
  Complex *u_odd 	= new Complex[m];
  Complex *f_even = new Complex[m];
  Complex *f_odd 	= new Complex[m];

  for (int i = 0, k = 0; i < m; i++, k += 2) {
    u_even[i] = u[k]; 
  }
  for (int i = 0, k = 1; i < m; i++, k += 2) {
    u_odd[i] = u[k]; 
  }

  ifft(u_even, f_even, m); 
  ifft(u_odd, f_odd, m); 

  double h = (M_PI + M_PI) / n;
  Complex omega(cos(h), sin(h)); 

	Complex z, tmp;
  for (int i = 0, k = m; i < m; i++, k++) {
		Complex z = Complex(cos(i*h), sin(i*h));
		tmp = z * f_odd[i];
    f[i] = f_even[i] + tmp;
    f[k] = f_even[i] - tmp; 
  }

  delete[] f_even; 
  delete[] f_odd; 
  delete[] u_even; 
  delete[] u_odd;
}


#ifdef USE_FFTW3

//:::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
// node-centered
//:::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
void FastDiffusionSolver2d_fftw3(MatrixXd &unknown, double eta, int bc, int order)
{

	int n1 = unknown.rows()-1;
	int n2 = unknown.cols()-1;

	int N;

	if (bc == ZFFT_DIRICHLET) {
		N = (n1-1)*(n2-1);
	} else if (bc == ZFFT_NEUMANN) {
		N = (n1+1)*(n2+1);
	} else if (bc == ZFFT_PERIODIC) {
		N = n1*n2;
	}

	double 			 *f_r = 0;
	fftw_complex *f_c = 0;

	if (bc == ZFFT_DIRICHLET || bc == ZFFT_NEUMANN) {
		f_r = new double[N];
	} else if (bc == ZFFT_PERIODIC) {
		f_c = new fftw_complex[N];
	}


	if (bc == ZFFT_DIRICHLET) {

#pragma omp parallel for collapse(2)
		for(int i = 1; i < n1; i++){
			for(int j = 1; j < n2; j++){
				int l = (i-1)*(n2-1) + j-1;
				f_r[l] = unknown[i][j];
			}
		}

	} else if (bc == ZFFT_NEUMANN) {

#pragma omp parallel for collapse(2)
		for(int i = 0; i <= n1; i++){
			for(int j = 0; j <= n2; j++){
				int l = i*(n2+1) + j;
				f_r[l] = unknown[i][j];
			}
		}

	} else if (bc == ZFFT_PERIODIC) {

#pragma omp parallel for collapse(2)
		for(int i = 0; i < n1; i++){
			for(int j = 0; j < n2; j++){
				int l = i * n2 + j;
				f_c[l][0] = unknown[i][j];
				f_c[l][1] = 0.0;
			}
		}

	}

	int status = fftw_init_threads();
	if (status == 0) {
		std::cout << "failed to init threads." << std::endl;
		exit(1);
	}
	fftw_plan_with_nthreads(omp_get_max_threads());

	fftw_plan p;

	if (bc == ZFFT_DIRICHLET) {
		p = fftw_plan_r2r_2d(n1-1, n2-1, f_r, f_r, FFTW_RODFT00, 
												 FFTW_RODFT00, FFTW_ESTIMATE);

	} else if (bc == ZFFT_NEUMANN) {
		p = fftw_plan_r2r_2d(n1+1, n2+1, f_r, f_r, FFTW_REDFT00, 
												 FFTW_REDFT00, FFTW_ESTIMATE);

	} else if (bc == ZFFT_PERIODIC) {
		p = fftw_plan_dft_2d(n1, n2, f_c, f_c, FFTW_FORWARD, FFTW_ESTIMATE);
	} 

	fftw_execute(p);
	fftw_destroy_plan(p);

	double C0, C1, C2;

	if (order == ZFFT_TWO) {
		C0 = 4.0 + eta;
		C1 = 2.0;
		C2 = 0.0;
	} else if (order == ZFFT_FOUR) {
		C0 = 10.0 / 3.0 + 2.0 / 3.0 * eta;
		C1 = 2.0 * (2.0 / 3.0 - eta / 12.0);
		C2 = 4.0 * 1.0 / 6.0;
	}

	double h1, h2;

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
	

	if (bc == ZFFT_DIRICHLET) {

#pragma omp parallel for collapse(2)
		for(int i = 1; i < n1; i++){
			for(int j = 1; j < n2; j++){
				int l = (i-1)*(n2-1) + j-1;

				double cs1 = csi[i];
				double cs2 = csj[j];

				double s1 = cs1 + cs2;
				double s2 = cs1 * cs2;

				double lambda = C0 - (C1 * s1 + C2 * s2);
				f_r[l] /= lambda;
				
			}
		}

	} else if (bc == ZFFT_NEUMANN) {

#pragma omp parallel for collapse(2)
		for(int i = 0; i <= n1; i++){
			for(int j = 0; j <= n2; j++){

				int l = i*(n2+1) + j;

				double cs1 = csi[i];
				double cs2 = csj[j];

				double s1 = cs1 + cs2;
				double s2 = cs1 * cs2;

				double lambda = C0 - (C1 * s1 + C2 * s2);
				f_r[l] /= lambda;
				
			}
		}

		if (fabs(eta) < 1.0e-12) {
			f_r[0] = 0.0;
		} 

	} else if (bc == ZFFT_PERIODIC) {

#pragma omp parallel for collapse(2)
		for(int i = 0; i < n1; i++){
			for(int j = 0; j < n2; j++){

				int l = i * n2 + j;

				double cs1 = csi[i];
				double cs2 = csj[j];

				double s1 = cs1 + cs2;
				double s2 = cs1 * cs2;

				double lambda = C0 - (C1 * s1 + C2 * s2);

				f_c[l][0] /= lambda;
				f_c[l][1] /= lambda;
				
			}
		}

		if (fabs(eta) < 1.0e-12) {
			f_c[0][0] = f_c[0][1] = 0.0;
		}

	} 

	delete[] csi;
	delete[] csj;

	if (bc == ZFFT_DIRICHLET) {
		p = fftw_plan_r2r_2d(n1-1, n2-1, f_r, f_r, FFTW_RODFT00, 
											 	 FFTW_RODFT00, FFTW_ESTIMATE);
	} else if (bc == ZFFT_NEUMANN) {
		p = fftw_plan_r2r_2d(n1+1, n2+1, f_r, f_r, FFTW_REDFT00, 
											 	 FFTW_REDFT00, FFTW_ESTIMATE);
	} else if (bc == ZFFT_PERIODIC) {
		p = fftw_plan_dft_2d(n1, n2, f_c, f_c, FFTW_BACKWARD, FFTW_ESTIMATE);
	} 

	fftw_execute(p);
	fftw_destroy_plan(p);

	if (bc == ZFFT_DIRICHLET) {

		double r_deno = 0.25 / (n1*n2);

#pragma omp parallel for collapse(2)
		for(int i = 1; i < n1; i++){
			for(int j = 1; j < n2; j++){
				int l = (i-1)*(n2-1) + j-1;
				unknown[i][j] = f_r[l] * r_deno;
			}
		}

	} else if (bc == ZFFT_NEUMANN) {

		double r_deno = 0.25 / (n1*n2);

#pragma omp parallel for collapse(2)
		for(int i = 0; i <= n1; i++){
			for(int j = 0; j <= n2; j++){
				int l = i*(n2+1) + j;
				unknown[i][j] = f_r[l] * r_deno;
				
			}
		}


	} else if (bc == ZFFT_PERIODIC) {

#pragma omp parallel for collapse(2)
		for(int i = 0; i < n1; i++){
			for(int j = 0; j < n2; j++){
				int l = i * n2 + j;
				unknown[i][j] = f_c[l][0] / N;
				
			}
		}

		for(int i = 0; i <= n1; i++){
			unknown[i][n2] = unknown[i][0];
			unknown[n1][i] = unknown[0][i];
		}

	} 


	if (f_r != 0) {
		delete[] f_r;
		f_r = 0;
	}
	if (f_c != 0) {
		delete[] f_c;
		f_c = 0;
	}

	fftw_cleanup_threads();
}

//:::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
void FastDiffusionSolver2d_HDVN(MatrixXd &unknown, double eta, int order)
{
	// Left and Right: Dirichlet
	// Top and Bottom: Neumann

	int n1 = unknown.rows()-1;
	int n2 = unknown.cols()-1;

	int N = (n1-1)*(n2+1);

	double 			 *f_r = 0;
	fftw_complex *f_c = 0;

	f_r = new double[N];

#pragma omp parallel for collapse(2)
	for(int i = 1; i < n1; i++){
		for(int j = 0; j <= n2; j++){
			int l = (i-1)*(n2+1) + j;
			f_r[l] = unknown[i][j];
		}
	}

	int status = fftw_init_threads();
	if (status == 0) {
		std::cout << "failed to init threads." << std::endl;
		exit(1);
	}
	fftw_plan_with_nthreads(omp_get_max_threads());

	fftw_plan p;

	p = fftw_plan_r2r_2d(n1-1, n2+1, f_r, f_r, FFTW_RODFT00, 
											 FFTW_REDFT00, FFTW_ESTIMATE);

	fftw_execute(p);
	fftw_destroy_plan(p);

	double C0, C1, C2;

	if (order == ZFFT_TWO) {
		C0 = 4.0 + eta;
		C1 = 2.0;
		C2 = 0.0;
	} else if (order == ZFFT_FOUR) {
		C0 = 10.0 / 3.0 + 2.0 / 3.0 * eta;
		C1 = 2.0 * (2.0 / 3.0 - eta / 12.0);
		C2 = 4.0 * 1.0 / 6.0;
	}

	double h1, h2;

	h1 = M_PI/n1;
	h2 = M_PI/n2;

	double *csi = new double[n1+1];
	double *csj = new double[n2+1];

	for(int i = 0; i <= n1; i++){
		csi[i] = cos(i * h1);
	}
	for(int j = 0; j <= n2; j++){
		csj[j] = cos(j * h2);
	}
	

#pragma omp parallel for collapse(2)
	for(int i = 1; i < n1; i++){
		for(int j = 0; j <= n2; j++){
			int l = (i-1)*(n2+1) + j;

			double cs1 = csi[i];
			double cs2 = csj[j];

			double s1 = cs1 + cs2;
			double s2 = cs1 * cs2;

			double lambda = C0 - (C1 * s1 + C2 * s2);
			f_r[l] /= lambda;
		}
	}

	delete[] csi;
	delete[] csj;

	p = fftw_plan_r2r_2d(n1-1, n2+1, f_r, f_r, FFTW_RODFT00, 
										 	 FFTW_REDFT00, FFTW_ESTIMATE);

	fftw_execute(p);
	fftw_destroy_plan(p);

	double r_deno = 0.25 / (n1*n2);

#pragma omp parallel for collapse(2)
	for(int i = 1; i < n1; i++){
		for(int j = 0; j <= n2; j++){
			int l = (i-1)*(n2+1) + j;
			unknown[i][j] = f_r[l] * r_deno;
		}
	}

	if (f_r != 0) {
		delete[] f_r;
		f_r = 0;
	}
	if (f_c != 0) {
		delete[] f_c;
		f_c = 0;
	}

	fftw_cleanup_threads();
}

//:::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
void FastDiffusionSolver2d_c(MatrixXd &unknown, double eta, int bc,
														 int n1, int n2)
{
	int m1 = n1 << 1;
	int m2 = n2 << 1;

	assert(n1 == unknown.rows() || n1+2 == unknown.rows());
	assert(n2 == unknown.cols() || n2+2 == unknown.cols());

	int ofs = 0;
	if (n1+2 == unknown.rows()) {
		ofs = 1;
	}

	int N;

	if (bc == ZFFT_DIRICHLET) {
		N = (m1-1)*(m2-1);
	} else if (bc == ZFFT_NEUMANN) {
		N = (m1+1)*(m2+1);
	} else if (bc == ZFFT_PERIODIC) {
		N = m1*m2;
	}

	double 			 *f_r = 0;
	fftw_complex *f_c = 0;

	if (bc == ZFFT_DIRICHLET || bc == ZFFT_NEUMANN) {
		f_r = new double[N];
#pragma omp parallel for
		for(int i = 0; i < N; i++){
			f_r[i] = 0.0;
		}
	} else if (bc == ZFFT_PERIODIC) {
		f_c = new fftw_complex[N];
#pragma omp parallel for
		for(int i = 0; i < N; i++){
			f_c[i][0] = f_c[i][0] = 0.0;
		}
	}

	if (bc == ZFFT_DIRICHLET) {

#pragma omp parallel for collapse(2)
	for(int i = 0; i < n1; i++){
		for(int j = 0; j < n2; j++){
				int i1 = (i+i)+1;
				int j1 = (j+j)+1;
				int l = (i1-1)*(m2-1) + j1-1;
				f_r[l] = unknown[i+ofs][j+ofs];
			}
		}

	} else if (bc == ZFFT_NEUMANN) {

#pragma omp parallel for collapse(2)
	for(int i = 0; i < n1; i++){
		for(int j = 0; j < n2; j++){
				int i1 = (i+i)+1;
				int j1 = (j+j)+1;
				int l = i1*(m2+1) + j1;
				f_r[l] = unknown[i+ofs][j+ofs];
			}
		}

	} else if (bc == ZFFT_PERIODIC) {

#pragma omp parallel for collapse(2)
	for(int i = 0; i < n1; i++){
		for(int j = 0; j < n2; j++){
				int i1 = (i+i)+1;
				int j1 = (j+j)+1;
				int l = i1*m2 + j1;
				f_c[l][0] = unknown[i+ofs][j+ofs];
				f_c[l][1] = 0.0;
			}
		}

	}

	int status = fftw_init_threads();
	if (status == 0) {
		std::cout << "failed to init threads." << std::endl;
		exit(1);
	}

	fftw_plan_with_nthreads(omp_get_max_threads());

	fftw_plan p;

	if (bc == ZFFT_DIRICHLET) {
		p = fftw_plan_r2r_2d(m1-1, m2-1, f_r, f_r, FFTW_RODFT00, 
												 FFTW_RODFT00, FFTW_ESTIMATE);

	} else if (bc == ZFFT_NEUMANN) {
		p = fftw_plan_r2r_2d(m1+1, m2+1, f_r, f_r, FFTW_REDFT00, 
												 FFTW_REDFT00, FFTW_ESTIMATE);

	} else if (bc == ZFFT_PERIODIC) {
		p = fftw_plan_dft_2d(m1, m2, f_c, f_c, FFTW_FORWARD, FFTW_ESTIMATE);
	} 

	fftw_execute(p);
	fftw_destroy_plan(p);

	double C0, C1, C2;

	C0 = 4.0 + eta;
	C1 = 2.0;
	C2 = 0.0;

	double h1, h2;

	if (bc == ZFFT_DIRICHLET || bc == ZFFT_NEUMANN) {
		h1 = M_PI/n1;
		h2 = M_PI/n2;
	} else if (bc == ZFFT_PERIODIC) {
		h1 = 2.0*M_PI/n1;
		h2 = 2.0*M_PI/n2;
	}

	double *csi = new double[m1+1];
	double *csj = new double[m2+1];

	for(int i = 0; i <= m1; i++){
		csi[i] = cos(i*h1);
	}
	for(int j = 0; j <= m2; j++){
		csj[j] = cos(j*h2);
	}
	
	if (bc == ZFFT_DIRICHLET) {

#pragma omp parallel for collapse(2)
		for(int i = 1; i < m1; i++){
			for(int j = 1; j < m2; j++){
				int l = (i-1)*(m2-1) + j-1;

				double cs1 = csi[i];
				double cs2 = csj[j];

				double s1 = cs1 + cs2;
				double s2 = cs1 * cs2;

				double lambda = C0 - (C1 * s1 + C2 * s2);
				f_r[l] /= lambda;
				
			}
		}

	} else if (bc == ZFFT_NEUMANN) {

#pragma omp parallel for collapse(2)
		for(int i = 0; i <= m1; i++){
			for(int j = 0; j <= m2; j++){

				int l = i*(m2+1) + j;

				double cs1 = csi[i];
				double cs2 = csj[j];

				double s1 = cs1 + cs2;
				double s2 = cs1 * cs2;

				double lambda = C0 - (C1 * s1 + C2 * s2);

				if (fabs(lambda) < DBL_EPSILON) {
					f_r[l] = 0.0;
				} else {
					f_r[l] /= lambda;
				}

			}
		}

		if (fabs(eta) < 1.0e-12) {
			f_r[0] = 0.0;
		} 

	} else if (bc == ZFFT_PERIODIC) {

#pragma omp parallel for collapse(2)
		for(int i = 0; i < m1; i++){
			for(int j = 0; j < m2; j++){

				int l = i * m2 + j;

				double cs1 = csi[i];
				double cs2 = csj[j];

				double s1 = cs1 + cs2;
				double s2 = cs1 * cs2;

				double lambda = C0 - (C1 * s1 + C2 * s2);

				if (fabs(lambda) < DBL_EPSILON) {
					f_c[l][0] = 0.0;
					f_c[l][1] = 0.0;
				} else {
					f_c[l][0] /= lambda;
					f_c[l][1] /= lambda;
				}
				
			}
		}

		if (fabs(eta) < 1.0e-12) {
			f_c[0][0] = f_c[0][1] = 0.0;
		}

	} 

	delete[] csi;
	delete[] csj;


	if (bc == ZFFT_DIRICHLET) {
		p = fftw_plan_r2r_2d(m1-1, m2-1, f_r, f_r, FFTW_RODFT00, 
											 	 FFTW_RODFT00, FFTW_ESTIMATE);
	} else if (bc == ZFFT_NEUMANN) {
		p = fftw_plan_r2r_2d(m1+1, m2+1, f_r, f_r, FFTW_REDFT00, 
											 	 FFTW_REDFT00, FFTW_ESTIMATE);
	} else if (bc == ZFFT_PERIODIC) {
		p = fftw_plan_dft_2d(m1, m2, f_c, f_c, FFTW_BACKWARD, FFTW_ESTIMATE);
	} 

	fftw_execute(p);
	fftw_destroy_plan(p);


	if (bc == ZFFT_DIRICHLET) {

		double r_deno = 0.25 / (m1*m2);

#pragma omp parallel for collapse(2)
		for(int i = 0; i < n1; i++){
			for(int j = 0; j < n2; j++){
				int i1 = (i+i)+1;
				int j1 = (j+j)+1;
				int l = (i1-1)*(m2-1) + j1-1;
				unknown[i+ofs][j+ofs] = f_r[l] * r_deno;
			}
		}

	} else if (bc == ZFFT_NEUMANN) {

		double r_deno = 0.25 / (m1*m2);

#pragma omp parallel for collapse(2)
		for(int i = 0; i < n1; i++){
			for(int j = 0; j < n2; j++){
				int i1 = (i+i)+1;
				int j1 = (j+j)+1;
				int l = i1*(m2+1) + j1;
				unknown[i+ofs][j+ofs] = f_r[l] * r_deno;
			}
		}

	} else if (bc == ZFFT_PERIODIC) {

		double r_deno = 1.0 / (m1*m2);

#pragma omp parallel for collapse(2)
		for(int i = 0; i < n1; i++){
			for(int j = 0; j < n2; j++){
				int i1 = (i+i)+1;
				int j1 = (j+j)+1;
				int l = i1*m2 + j1;
				unknown[i+ofs][j+ofs] = f_c[l][0] * r_deno;
			}
		}

	} 

	if (f_r != 0) {
		delete[] f_r;
		f_r = 0;
	}
	if (f_c != 0) {
		delete[] f_c;
		f_c = 0;
	}

	fftw_cleanup_threads();
}

//:::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
void FastDiffusionSolver2d_e1(MatrixXd &unknown, double eta, int bc,
														  int n1, int n2)
{
	//int m1 = n1 << 1;
	int m2 = n2 << 1;

	assert(n1+1 == unknown.rows());
	assert(n2 == unknown.cols() || n2+2 == unknown.cols());

	int ofs = 0;
	if (n2+2 == unknown.cols())	{
		ofs = 1;
	}

	int N;

	if (bc == ZFFT_DIRICHLET) {
		N = (n1-1)*(m2-1);
	} else if (bc == ZFFT_NEUMANN) {
		N = (n1+1)*(m2+1);
	} else if (bc == ZFFT_PERIODIC) {
		N = n1*m2;
	}

	double 			 *f_r = 0;
	fftw_complex *f_c = 0;

	if (bc == ZFFT_DIRICHLET || bc == ZFFT_NEUMANN) {
		f_r = new double[N];
#pragma omp parallel for
		for(int i = 0; i < N; i++){
			f_r[i] = 0.0;
		}
	} else if (bc == ZFFT_PERIODIC) {
		f_c = new fftw_complex[N];
#pragma omp parallel for
		for(int i = 0; i < N; i++){
			f_c[i][0] = f_c[i][0] = 0.0;
		}
	}

	if (bc == ZFFT_DIRICHLET) {

#pragma omp parallel for collapse(2)
	for(int i = 1; i < n1; i++){
		for(int j = 0; j < n2; j++){
				int j1 = (j+j)+1;
				int l = (i-1)*(m2-1) + j1-1;
				f_r[l] = unknown[i][j+ofs];
			}
		}

	} else if (bc == ZFFT_NEUMANN) {

#pragma omp parallel for collapse(2)
	for(int i = 0; i <= n1; i++){
		for(int j = 0; j < n2; j++){
				int j1 = (j+j)+1;
				int l = i*(m2+1) + j1;
				f_r[l] = unknown[i][j+ofs];
			}
		}

	} else if (bc == ZFFT_PERIODIC) {

#pragma omp parallel for collapse(2)
	for(int i = 0; i < n1; i++){
		for(int j = 0; j < n2; j++){
				int j1 = (j+j)+1;
				int l = i*m2 + j1;
				f_c[l][0] = unknown[i][j+ofs];
				f_c[l][1] = 0.0;
			}
		}

	}

	int status = fftw_init_threads();
	if (status == 0) {
		std::cout << "failed to init threads." << std::endl;
		exit(1);
	}

	fftw_plan_with_nthreads(omp_get_max_threads());

	fftw_plan p;

	if (bc == ZFFT_DIRICHLET) {
		p = fftw_plan_r2r_2d(n1-1, m2-1, f_r, f_r, FFTW_RODFT00, 
												 FFTW_RODFT00, FFTW_ESTIMATE);

	} else if (bc == ZFFT_NEUMANN) {
		p = fftw_plan_r2r_2d(n1+1, m2+1, f_r, f_r, FFTW_REDFT00, 
												 FFTW_REDFT00, FFTW_ESTIMATE);

	} else if (bc == ZFFT_PERIODIC) {
		p = fftw_plan_dft_2d(n1, m2, f_c, f_c, FFTW_FORWARD, FFTW_ESTIMATE);
	} 

	fftw_execute(p);
	fftw_destroy_plan(p);

	double C0, C1, C2;

	C0 = 4.0 + eta;
	C1 = 2.0;
	C2 = 0.0;

	double h1, h2;

	if (bc == ZFFT_DIRICHLET || bc == ZFFT_NEUMANN) {
		h1 = M_PI/n1;
		h2 = M_PI/n2;
	} else if (bc == ZFFT_PERIODIC) {
		h1 = 2.0*M_PI/n1;
		h2 = 2.0*M_PI/n2;
	}

	double *csi = new double[n1+1];
	double *csj = new double[m2+1];

	for(int i = 0; i <= n1; i++){
		csi[i] = cos(i * h1);
	}
	for(int j = 0; j <= m2; j++){
		csj[j] = cos(j * h2);
	}
	
	if (bc == ZFFT_DIRICHLET) {

#pragma omp parallel for collapse(2)
		for(int i = 1; i < n1; i++){
			for(int j = 1; j < m2; j++){
				int l = (i-1)*(m2-1) + j-1;

				double cs1 = csi[i];
				double cs2 = csj[j];

				double s1 = cs1 + cs2;
				double s2 = cs1 * cs2;

				double lambda = C0 - (C1 * s1 + C2 * s2);
				f_r[l] /= lambda;
				
			}
		}

	} else if (bc == ZFFT_NEUMANN) {

#pragma omp parallel for collapse(2)
		for(int i = 0; i <= n1; i++){
			for(int j = 0; j <= m2; j++){

				int l = i*(m2+1) + j;

				double cs1 = csi[i];
				double cs2 = csj[j];

				double s1 = cs1 + cs2;
				double s2 = cs1 * cs2;

				double lambda = C0 - (C1 * s1 + C2 * s2);

				if (fabs(lambda) < DBL_EPSILON) {
					f_r[l] = 0.0;
				} else {
					f_r[l] /= lambda;
				}
				
			}
		}

		if (fabs(eta) < 1.0e-12) {
			f_r[0] = 0.0;
		} 

	} else if (bc == ZFFT_PERIODIC) {

#pragma omp parallel for collapse(2)
		for(int i = 0; i < n1; i++){
			for(int j = 0; j < m2; j++){

				int l = i * m2 + j;

				double cs1 = csi[i];
				double cs2 = csj[j];

				double s1 = cs1 + cs2;
				double s2 = cs1 * cs2;

				double lambda = C0 - (C1 * s1 + C2 * s2);

				if (fabs(lambda) < DBL_EPSILON) {
					f_c[l][0] = 0.0;
					f_c[l][1] = 0.0;
				} else {
					f_c[l][0] /= lambda;
					f_c[l][1] /= lambda;
				}
				
			}
		}

		if (fabs(eta) < 1.0e-12) {
			f_c[0][0] = f_c[0][1] = 0.0;
		}

	} 

	delete[] csi;
	delete[] csj;

	if (bc == ZFFT_DIRICHLET) {
		p = fftw_plan_r2r_2d(n1-1, m2-1, f_r, f_r, FFTW_RODFT00, 
											 	 FFTW_RODFT00, FFTW_ESTIMATE);
	} else if (bc == ZFFT_NEUMANN) {
		p = fftw_plan_r2r_2d(n1+1, m2+1, f_r, f_r, FFTW_REDFT00, 
											 	 FFTW_REDFT00, FFTW_ESTIMATE);
	} else if (bc == ZFFT_PERIODIC) {
		p = fftw_plan_dft_2d(n1, m2, f_c, f_c, FFTW_BACKWARD, FFTW_ESTIMATE);
	} 

	fftw_execute(p);
	fftw_destroy_plan(p);

	if (bc == ZFFT_DIRICHLET) {

		double r_deno = 0.25 / (n1*m2);

#pragma omp parallel for collapse(2)
		for(int i = 1; i < n1; i++){
			for(int j = 0; j < n2; j++){
				int j1 = (j+j)+1;
				int l = (i-1)*(m2-1) + j1-1;
				unknown[i][j+ofs] = f_r[l] * r_deno;
			}
		}

	} else if (bc == ZFFT_NEUMANN) {

		double r_deno = 0.25 / (n1*m2);

#pragma omp parallel for collapse(2)
		for(int i = 0; i <= n1; i++){
			for(int j = 0; j < n2; j++){
				int j1 = (j+j)+1;
				int l = i*(m2+1) + j1;
				unknown[i][j+ofs] = f_r[l] * r_deno;
				
			}
		}

	} else if (bc == ZFFT_PERIODIC) {

		double r_deno = 1.0 / (n1*m2);

#pragma omp parallel for collapse(2)
		for(int i = 0; i < n1; i++){
			for(int j = 0; j < n2; j++){
				int j1 = (j+j)+1;
				int l = i*m2 + j1;
				unknown[i][j+ofs] = f_c[l][0] * r_deno;
				
			}
		}

	} 

	if (f_r != 0) {
		delete[] f_r;
		f_r = 0;
	}
	if (f_c != 0) {
		delete[] f_c;
		f_c = 0;
	}

	fftw_cleanup_threads();
}

//:::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
void FastDiffusionSolver2d_e2(MatrixXd &unknown, double eta, int bc, 
															int n1, int n2)
{
	int m1 = n1 << 1;
	//int m2 = n2 << 1;

	assert(n1 == unknown.rows() || n1+2 == unknown.rows());
	assert(n2+1 == unknown.cols());

	int ofs = 0;
	if (n1+2 == unknown.rows()) {
		ofs = 1;
	}

	int N;

	if (bc == ZFFT_DIRICHLET) {
		N = (m1-1)*(n2-1);
	} else if (bc == ZFFT_NEUMANN) {
		N = (m1+1)*(n2+1);
	} else if (bc == ZFFT_PERIODIC) {
		N = m1*n2;
	}

	double 			 *f_r = 0;
	fftw_complex *f_c = 0;

	if (bc == ZFFT_DIRICHLET || bc == ZFFT_NEUMANN) {
		f_r = new double[N];
#pragma omp parallel for
		for(int i = 0; i < N; i++){
			f_r[i] = 0.0;
		}
	} else if (bc == ZFFT_PERIODIC) {
		f_c = new fftw_complex[N];
#pragma omp parallel for
		for(int i = 0; i < N; i++){
			f_c[i][0] = f_c[i][0] = 0.0;
		}
	}

	if (bc == ZFFT_DIRICHLET) {

#pragma omp parallel for collapse(2)
	for(int i = 0; i < n1; i++){
		for(int j = 1; j < n2; j++){
				int i1 = (i+i)+1;
				int l = (i1-1)*(n2-1) + j-1;
				f_r[l] = unknown[i+ofs][j];
			}
		}

	} else if (bc == ZFFT_NEUMANN) {

#pragma omp parallel for collapse(2)
	for(int i = 0; i < n1; i++){
		for(int j = 0; j <= n2; j++){
				int i1 = (i+i)+1;
				int l = i1*(n2+1) + j;
				f_r[l] = unknown[i+ofs][j];
			}
		}

	} else if (bc == ZFFT_PERIODIC) {

#pragma omp parallel for collapse(2)
	for(int i = 0; i < n1; i++){
		for(int j = 0; j < n2; j++){
				int i1 = (i+i)+1;
				int l = i1*n2 + j;
				f_c[l][0] = unknown[i+ofs][j];
				f_c[l][1] = 0.0;
			}
		}

	}

	int status = fftw_init_threads();
	if (status == 0) {
		std::cout << "failed to init threads." << std::endl;
		exit(1);
	}

	fftw_plan_with_nthreads(omp_get_max_threads());

	fftw_plan p;

	if (bc == ZFFT_DIRICHLET) {
		p = fftw_plan_r2r_2d(m1-1, n2-1, f_r, f_r, FFTW_RODFT00, 
												 FFTW_RODFT00, FFTW_ESTIMATE);

	} else if (bc == ZFFT_NEUMANN) {
		p = fftw_plan_r2r_2d(m1+1, n2+1, f_r, f_r, FFTW_REDFT00, 
												 FFTW_REDFT00, FFTW_ESTIMATE);

	} else if (bc == ZFFT_PERIODIC) {
		p = fftw_plan_dft_2d(m1, n2, f_c, f_c, FFTW_FORWARD, FFTW_ESTIMATE);
	} 

	fftw_execute(p);
	fftw_destroy_plan(p);

	double C0, C1, C2;

	C0 = 4.0 + eta;
	C1 = 2.0;
	C2 = 0.0;

	double h1, h2;

	if (bc == ZFFT_DIRICHLET || bc == ZFFT_NEUMANN) {
		h1 = M_PI/n1;
		h2 = M_PI/n2;
	} else if (bc == ZFFT_PERIODIC) {
		h1 = 2.0*M_PI/n1;
		h2 = 2.0*M_PI/n2;
	}

	double *csi = new double[m1+1];
	double *csj = new double[n2+1];

	for(int i = 0; i <= m1; i++){
		csi[i] = cos(i * h1);
	}
	for(int j = 0; j <= n2; j++){
		csj[j] = cos(j * h2);
	}
	

	if (bc == ZFFT_DIRICHLET) {

#pragma omp parallel for collapse(2)
		for(int i = 1; i < m1; i++){
			for(int j = 1; j < n2; j++){
				int l = (i-1)*(n2-1) + j-1;

				double cs1 = csi[i];
				double cs2 = csj[j];

				double s1 = cs1 + cs2;
				double s2 = cs1 * cs2;

				double lambda = C0 - (C1 * s1 + C2 * s2);
				f_r[l] /= lambda;
				
			}
		}

	} else if (bc == ZFFT_NEUMANN) {

#pragma omp parallel for collapse(2)
		for(int i = 0; i <= m1; i++){
			for(int j = 0; j <= n2; j++){

				int l = i*(n2+1) + j;

				double cs1 = csi[i];
				double cs2 = csj[j];

				double s1 = cs1 + cs2;
				double s2 = cs1 * cs2;

				double lambda = C0 - (C1 * s1 + C2 * s2);

				if (fabs(lambda) < DBL_EPSILON) {
					f_r[l] = 0.0;
				} else {
					f_r[l] /= lambda;
				}
				
			}
		}

		if (fabs(eta) < 1.0e-12) {
			f_r[0] = 0.0;
		} 

	} else if (bc == ZFFT_PERIODIC) {

#pragma omp parallel for collapse(2)
		for(int i = 0; i < m1; i++){
			for(int j = 0; j < n2; j++){

				int l = i * n2 + j;

				double cs1 = csi[i];
				double cs2 = csj[j];

				double s1 = cs1 + cs2;
				double s2 = cs1 * cs2;

				double lambda = C0 - (C1 * s1 + C2 * s2);

				if (fabs(lambda) < DBL_EPSILON) {
					f_c[l][0] = 0.0;
					f_c[l][1] = 0.0;
				} else {
					f_c[l][0] /= lambda;
					f_c[l][1] /= lambda;
				}
				
			}
		}

		if (fabs(eta) < 1.0e-12) {
			f_c[0][0] = f_c[0][1] = 0.0;
		}

	} 

	delete[] csi;
	delete[] csj;

	if (bc == ZFFT_DIRICHLET) {
		p = fftw_plan_r2r_2d(m1-1, n2-1, f_r, f_r, FFTW_RODFT00, 
											 	 FFTW_RODFT00, FFTW_ESTIMATE);
	} else if (bc == ZFFT_NEUMANN) {
		p = fftw_plan_r2r_2d(m1+1, n2+1, f_r, f_r, FFTW_REDFT00, 
											 	 FFTW_REDFT00, FFTW_ESTIMATE);
	} else if (bc == ZFFT_PERIODIC) {
		p = fftw_plan_dft_2d(m1, n2, f_c, f_c, FFTW_BACKWARD, FFTW_ESTIMATE);
	} 

	fftw_execute(p);
	fftw_destroy_plan(p);

	if (bc == ZFFT_DIRICHLET) {

		double r_deno = 0.25 / (m1*n2);

#pragma omp parallel for collapse(2)
		for(int i = 0; i < n1; i++){
			for(int j = 1; j < n2; j++){
				int i1 = (i+i)+1;
				int l = (i1-1)*(n2-1) + j-1;
				unknown[i+ofs][j] = f_r[l] * r_deno;
			}
		}

	} else if (bc == ZFFT_NEUMANN) {

		double r_deno = 0.25 / (m1*n2);

#pragma omp parallel for collapse(2)
		for(int i = 0; i < n1; i++){
			for(int j = 0; j <= n2; j++){
				int i1 = (i+i)+1;
				int l = i1*(n2+1) + j;
				unknown[i+ofs][j] = f_r[l] * r_deno;
				
			}
		}

	} else if (bc == ZFFT_PERIODIC) {

		double r_deno = 1.0 / (m1*n2);

#pragma omp parallel for collapse(2)
		for(int i = 0; i < n1; i++){
			for(int j = 0; j < n2; j++){
				int i1 = (i+i)+1;
				int l = i1*n2 + j;
				unknown[i+ofs][j] = f_c[l][0] * r_deno;
				
			}
		}

	} 

	if (f_r != 0) {
		delete[] f_r;
		f_r = 0;
	}
	if (f_c != 0) {
		delete[] f_c;
		f_c = 0;
	}

	fftw_cleanup_threads();
}

//:::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
void FastDiffusionSolver3d_fftw3(TensorXd &unknown, double eta, int bc, int order)
{
	int dim[3];
	unknown.get_dim(dim);

	int n1 = dim[0]-1;
	int n2 = dim[1]-1;
	int n3 = dim[2]-1;

	int N;

	if (bc == ZFFT_DIRICHLET) {
		N = (n1-1)*(n2-1)*(n3-1);
	} else if (bc == ZFFT_NEUMANN) {
		N = (n1+1)*(n2+1)*(n3+1);
	} else if (bc == ZFFT_PERIODIC) {
		N = n1*n2*n3;
	}

	double 			 *f_r = 0;
	fftw_complex *f_c = 0;

	if (bc == ZFFT_DIRICHLET || bc == ZFFT_NEUMANN) {
		f_r = new double[N];
	} else if (bc == ZFFT_PERIODIC) {
		f_c = new fftw_complex[N];
	}


	if (bc == ZFFT_DIRICHLET) {

#pragma omp parallel for collapse(3)
		for(int i = 1; i < n1; i++){
			for(int j = 1; j < n2; j++){
				for(int k = 1; k < n3; k++){
					int l = ((i-1)*(n2-1) + j-1)*(n3-1) + k-1;
					f_r[l] = unknown[i][j][k];
				}
			}
		}

	} else if (bc == ZFFT_NEUMANN) {

#pragma omp parallel for collapse(3)
		for(int i = 0; i <= n1; i++){
			for(int j = 0; j <= n2; j++){
				for(int k = 0; k <= n3; k++){
					int l = (i*(n2+1) + j)*(n3+1) + k;
					f_r[l] = unknown[i][j][k];
				}
			}
		}

	} else if (bc == ZFFT_PERIODIC) {

#pragma omp parallel for collapse(3)
		for(int i = 0; i < n1; i++){
			for(int j = 0; j < n2; j++){
				for(int k = 0; k < n3; k++){
					int l = (i * n2 + j) * n3 + k;
					f_c[l][0] = unknown[i][j][k];
					f_c[l][1] = 0.0;
				}
			}
		}

	}

	int status = fftw_init_threads();
	if (status == 0) {
		std::cout << "failed to init threads." << std::endl;
		exit(1);
	}
	fftw_plan_with_nthreads(omp_get_max_threads());

	fftw_plan p;

	if (bc == ZFFT_DIRICHLET) {
		p = fftw_plan_r2r_3d(n1-1, n2-1, n3-1, f_r, f_r, FFTW_RODFT00, 
												 FFTW_RODFT00, FFTW_RODFT00, FFTW_ESTIMATE);

	} else if (bc == ZFFT_NEUMANN) {
		p = fftw_plan_r2r_3d(n1+1, n2+1, n3+1, f_r, f_r, FFTW_REDFT00, 
												 FFTW_REDFT00, FFTW_REDFT00, FFTW_ESTIMATE);

	} else if (bc == ZFFT_PERIODIC) {
		p = fftw_plan_dft_3d(n1, n2, n3, f_c, f_c, FFTW_FORWARD, FFTW_ESTIMATE);
	} 

	fftw_execute(p);
	fftw_destroy_plan(p);

	double C0, C1, C2, C3;

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

	double h1, h2, h3;

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
	

	if (bc == ZFFT_DIRICHLET) {

#pragma omp parallel for collapse(3)
		for(int i = 1; i < n1; i++){
			for(int j = 1; j < n2; j++){
				for(int k = 1; k < n3; k++){
					int l = ((i-1)*(n2-1) + j-1)*(n3-1) + k-1;

					double cs1 = csi[i];
					double cs2 = csj[j];
					double cs3 = csk[k];

					double s1 = cs1 + cs2 + cs3;
					double s2 = cs1 * cs2 + cs2 * cs3 + cs3 * cs1;
					double s3 = cs1 * cs2 * cs3;

					double lambda = C0 - (C1 * s1 + C2 * s2 + C3 * s3);
					f_r[l] /= lambda;
				}
			}
		}

	} else if (bc == ZFFT_NEUMANN) {

#pragma omp parallel for collapse(3)
		for(int i = 0; i <= n1; i++){
			for(int j = 0; j <= n2; j++){
				for(int k = 0; k <= n3; k++){
					int l = (i*(n2+1) + j)*(n3+1) + k;

					double cs1 = csi[i];
					double cs2 = csj[j];
					double cs3 = csk[k];

					double s1 = cs1 + cs2 + cs3;
					double s2 = cs1 * cs2 + cs2 * cs3 + cs3 * cs1;
					double s3 = cs1 * cs2 * cs3;

					double lambda = C0 - (C1 * s1 + C2 * s2 + C3 * s3);
					f_r[l] /= lambda;
				}
			}
		}

		if (fabs(eta) < 1.0e-12) {
			f_r[0] = 0.0;
		} 

	} else if (bc == ZFFT_PERIODIC) {

#pragma omp parallel for collapse(3)
		for(int i = 0; i < n1; i++){
			for(int j = 0; j < n2; j++){
				for(int k = 0; k < n3; k++){

					int l = (i * n2 + j) * n3 + k;

					double cs1 = csi[i];
					double cs2 = csj[j];
					double cs3 = csk[k];

					double s1 = cs1 + cs2 + cs3;
					double s2 = cs1 * cs2 + cs2 * cs3 + cs3 * cs1;
					double s3 = cs1 * cs2 * cs3;

					double lambda = C0 - (C1 * s1 + C2 * s2 + C3 * s3);

					f_c[l][0] /= lambda;
					f_c[l][1] /= lambda;
				}
			}
		}

		if (fabs(eta) < 1.0e-12) {
			f_c[0][0] = f_c[0][1] = 0.0;
		}

	} 

	delete[] csi;
	delete[] csj;
	delete[] csk;

	if (bc == ZFFT_DIRICHLET) {
		p = fftw_plan_r2r_3d(n1-1, n2-1, n3-1, f_r, f_r, FFTW_RODFT00, 
											 	 FFTW_RODFT00, FFTW_RODFT00, FFTW_ESTIMATE);
	} else if (bc == ZFFT_NEUMANN) {
		p = fftw_plan_r2r_3d(n1+1, n2+1, n3+1, f_r, f_r, FFTW_REDFT00, 
											 	 FFTW_REDFT00, FFTW_REDFT00, FFTW_ESTIMATE);
	} else if (bc == ZFFT_PERIODIC) {
		p = fftw_plan_dft_3d(n1, n2, n3, f_c, f_c, FFTW_BACKWARD, FFTW_ESTIMATE);
	} 

	fftw_execute(p);
	fftw_destroy_plan(p);

	if (bc == ZFFT_DIRICHLET) {

		double r_deno = 0.125 / (n1*n2*n3);

#pragma omp parallel for collapse(3)
		for(int i = 1; i < n1; i++){
			for(int j = 1; j < n2; j++){
				for(int k = 1; k < n3; k++){
					int l = ((i-1)*(n2-1) + j-1)*(n3-1) + k-1;
					unknown[i][j][k] = f_r[l] * r_deno;
				}
			}
		}

	} else if (bc == ZFFT_NEUMANN) {

		double r_deno = 0.125 / (n1*n2*n3);

#pragma omp parallel for collapse(3)
		for(int i = 0; i <= n1; i++){
			for(int j = 0; j <= n2; j++){
				for(int k = 0; k <= n3; k++){
					int l = (i*(n2+1) + j)*(n3+1) + k;
					unknown[i][j][k] = f_r[l] * r_deno;
				}
			}
		}


	} else if (bc == ZFFT_PERIODIC) {

#pragma omp parallel for collapse(3)
		for(int i = 0; i < n1; i++){
			for(int j = 0; j < n2; j++){
				for(int k = 0; k < n3; k++){
					int l = (i * n2 + j) * n3 + k;
					unknown[i][j][k] = f_c[l][0] / N;
				}
			}
		}

		for(int i = 0; i <= n1; i++){
			for(int j = 0; j <= n2; j++){
				unknown[i][j][n3] = unknown[i][j][0];
				unknown[i][n2][j] = unknown[i][0][j];
				unknown[n1][i][j] = unknown[0][i][j];
			}
		}

	} 


	if (f_r != 0) {
		delete[] f_r;
		f_r = 0;
	}
	if (f_c != 0) {
		delete[] f_c;
		f_c = 0;
	}

	fftw_cleanup_threads();
}

#endif

//+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
//:::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
void fft2d_mac(MatrixXd &unknown_r, MatrixXd &unknown_i, int n, int type)
{

#pragma omp parallel for
  for (int i = 0; i < n; i++) {
		double *src_r = new double[n];
		double *src_i = new double[n];
		double *dst_r = new double[n];
		double *dst_i = new double[n];

    for (int j = 0; j < n; j++) {
      src_r[j] = unknown_r[i][j]; 
      src_i[j] = unknown_i[i][j]; 
    }

		if (type == ZFFT_MAC_E1) {
			fft_c(src_r, src_i, dst_r, dst_i, n);
		} else if (type == ZFFT_MAC_E2) {
			fft(src_r, src_i, dst_r, dst_i, n);
		} else if (type == ZFFT_MAC_C) {
			fft_c(src_r, src_i, dst_r, dst_i, n);
		}

    for (int j = 0; j < n; j++) {
      unknown_r[i][j] = dst_r[j];
      unknown_i[i][j] = dst_i[j];
    }

  	delete[] src_r; 
  	delete[] src_i; 
  	delete[] dst_r;
  	delete[] dst_i;
  }

#pragma omp parallel for
  for (int j = 0; j < n; j++) {
		double *src_r = new double[n];
		double *src_i = new double[n];
		double *dst_r = new double[n];
		double *dst_i = new double[n];

    for (int i = 0; i < n; i++) {
      src_r[i] = unknown_r[i][j]; 
      src_i[i] = unknown_i[i][j]; 
    }

		if (type == ZFFT_MAC_E1) {
			fft(src_r, src_i, dst_r, dst_i, n);
		} else if (type == ZFFT_MAC_E2) {
			fft_c(src_r, src_i, dst_r, dst_i, n);
		} else if (type == ZFFT_MAC_C) {
			fft_c(src_r, src_i, dst_r, dst_i, n);
		}

    for (int i = 0; i < n; i++) {
      unknown_r[i][j] = dst_r[i];
      unknown_i[i][j] = dst_i[i];
    }

  	delete[] src_r; 
  	delete[] src_i; 
  	delete[] dst_r;
  	delete[] dst_i;
  }
}

//:::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
void ifft2d_mac(MatrixXd &unknown_r, MatrixXd &unknown_i, int n, int type)
{

#pragma omp parallel for
  for (int j = 0; j < n; j++) {
		double *src_r = new double[n];
		double *src_i = new double[n];
		double *dst_r = new double[n];
		double *dst_i = new double[n];

    for (int i = 0; i < n; i++) {
      src_r[i] = unknown_r[i][j]; 
      src_i[i] = unknown_i[i][j]; 
    }

		if (type == ZFFT_MAC_E1) {
			ifft(src_r, src_i, dst_r, dst_i, n);
		} else if (type == ZFFT_MAC_E2) {
			ifft_c(src_r, src_i, dst_r, dst_i, n);
		} else if (type == ZFFT_MAC_C) {
			ifft_c(src_r, src_i, dst_r, dst_i, n);
		}

    for (int i = 0; i < n; i++) {
      unknown_r[i][j] = dst_r[i];
      unknown_i[i][j] = dst_i[i];
    }

  	delete[] src_r; 
  	delete[] src_i; 
  	delete[] dst_r;
  	delete[] dst_i;
  }

#pragma omp parallel for
  for (int i = 0; i < n; i++) {
		double *src_r = new double[n];
		double *src_i = new double[n];
		double *dst_r = new double[n];
		double *dst_i = new double[n];

    for (int j = 0; j < n; j++) {
      src_r[j] = unknown_r[i][j]; 
      src_i[j] = unknown_i[i][j]; 
    }

		if (type == ZFFT_MAC_E1) {
			ifft_c(src_r, src_i, dst_r, dst_i, n);
		} else if (type == ZFFT_MAC_E2) {
			ifft(src_r, src_i, dst_r, dst_i, n);
		} else if (type == ZFFT_MAC_C) {
			ifft_c(src_r, src_i, dst_r, dst_i, n);
		}

    for (int j = 0; j < n; j++) {
      unknown_r[i][j] = dst_r[j];
      unknown_i[i][j] = dst_i[j];
    }

  	delete[] src_r; 
  	delete[] src_i; 
  	delete[] dst_r;
  	delete[] dst_i;
  }
}

//:::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
void fft3d_mac(TensorXd &unknown_r, TensorXd &unknown_i, 
							 int n0, int n1, int n2, int type)
{

#pragma omp parallel for collapse(2)
  for (int j = 0; j < n1; j++) {
		for(int k = 0; k < n2; k++){
			int n = n0;
			double *src_r = new double[n];
			double *src_i = new double[n];
			double *dst_r = new double[n];
			double *dst_i = new double[n];

    	for (int i = 0; i < n; i++) {
    	  src_r[i] = unknown_r[i][j][k]; 
    	  src_i[i] = unknown_i[i][j][k]; 
    	}

			if (type == ZFFT_MAC_E1) {
				fft(src_r, src_i, dst_r, dst_i, n);
			} else {
				fft_c(src_r, src_i, dst_r, dst_i, n);
			}

    	for (int i = 0; i < n; i++) {
    	  unknown_r[i][j][k] = dst_r[i];
    	  unknown_i[i][j][k] = dst_i[i];
    	}

  		delete[] src_r; 
  		delete[] src_i; 
  		delete[] dst_r;
  		delete[] dst_i;
		}
  }

#pragma omp parallel for collapse(2)
	for(int k = 0; k < n2; k++){
		for(int i = 0; i < n0; i++){
			int n = n1;
			double *src_r = new double[n];
			double *src_i = new double[n];
			double *dst_r = new double[n];
			double *dst_i = new double[n];

			for(int j = 0; j < n; j++){
    	  src_r[j] = unknown_r[i][j][k]; 
    	  src_i[j] = unknown_i[i][j][k]; 
    	}

			if (type == ZFFT_MAC_E2) {
				fft(src_r, src_i, dst_r, dst_i, n);
			} else {
				fft_c(src_r, src_i, dst_r, dst_i, n);
			}

			for(int j = 0; j < n; j++){
    	  unknown_r[i][j][k] = dst_r[j];
    	  unknown_i[i][j][k] = dst_i[j];
    	}

  		delete[] src_r; 
  		delete[] src_i; 
  		delete[] dst_r;
  		delete[] dst_i;
		}
  }

#pragma omp parallel for collapse(2)
	for(int i = 0; i < n0; i++){
		for(int j = 0; j < n1; j++){
			int n = n2;
			double *src_r = new double[n];
			double *src_i = new double[n];
			double *dst_r = new double[n];
			double *dst_i = new double[n];

			for(int k = 0; k < n; k++){
    	  src_r[k] = unknown_r[i][j][k]; 
    	  src_i[k] = unknown_i[i][j][k]; 
    	}

			if (type == ZFFT_MAC_E3) {
				fft(src_r, src_i, dst_r, dst_i, n);
			} else {
				fft_c(src_r, src_i, dst_r, dst_i, n);
			}

			for(int k = 0; k < n; k++){
    	  unknown_r[i][j][k] = dst_r[k];
    	  unknown_i[i][j][k] = dst_i[k];
    	}

  		delete[] src_r; 
  		delete[] src_i; 
  		delete[] dst_r;
  		delete[] dst_i;
		}
  }

}

//:::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
void ifft3d_mac(TensorXd &unknown_r, 
								TensorXd &unknown_i, 
							  int n0, int n1, int n2, int type)
{
#pragma omp parallel for collapse(2)
  for (int j = 0; j < n1; j++) {
		for(int k = 0; k < n2; k++){
			int n = n0;
			double *src_r = new double[n];
			double *src_i = new double[n];
			double *dst_r = new double[n];
			double *dst_i = new double[n];

    	for (int i = 0; i < n; i++) {
    	  src_r[i] = unknown_r[i][j][k]; 
    	  src_i[i] = unknown_i[i][j][k]; 
    	}

			if (type == ZFFT_MAC_E1) {
				ifft(src_r, src_i, dst_r, dst_i, n);
			} else {
				ifft_c(src_r, src_i, dst_r, dst_i, n);
			}

    	for (int i = 0; i < n; i++) {
    	  unknown_r[i][j][k] = dst_r[i];
    	  unknown_i[i][j][k] = dst_i[i];
    	}

  		delete[] src_r; 
  		delete[] src_i; 
  		delete[] dst_r;
  		delete[] dst_i;
		}
  }

#pragma omp parallel for collapse(2)
	for(int k = 0; k < n2; k++){
		for(int i = 0; i < n0; i++){
			int n = n1;
			double *src_r = new double[n];
			double *src_i = new double[n];
			double *dst_r = new double[n];
			double *dst_i = new double[n];

			for(int j = 0; j < n; j++){
    	  src_r[j] = unknown_r[i][j][k]; 
    	  src_i[j] = unknown_i[i][j][k]; 
    	}

			if (type == ZFFT_MAC_E2) {
				ifft(src_r, src_i, dst_r, dst_i, n);
			} else {
				ifft_c(src_r, src_i, dst_r, dst_i, n);
			}

			for(int j = 0; j < n; j++){
    	  unknown_r[i][j][k] = dst_r[j];
    	  unknown_i[i][j][k] = dst_i[j];
    	}

  		delete[] src_r; 
  		delete[] src_i; 
  		delete[] dst_r;
  		delete[] dst_i;
		}
  }

#pragma omp parallel for collapse(2)
	for(int i = 0; i < n0; i++){
		for(int j = 0; j < n1; j++){
			int n = n2;
			double *src_r = new double[n];
			double *src_i = new double[n];
			double *dst_r = new double[n];
			double *dst_i = new double[n];

			for(int k = 0; k < n; k++){
    	  src_r[k] = unknown_r[i][j][k]; 
    	  src_i[k] = unknown_i[i][j][k]; 
    	}

			if (type == ZFFT_MAC_E3) {
				ifft(src_r, src_i, dst_r, dst_i, n);
			} else {
				ifft_c(src_r, src_i, dst_r, dst_i, n);
			}

			for(int k = 0; k < n; k++){
    	  unknown_r[i][j][k] = dst_r[k];
    	  unknown_i[i][j][k] = dst_i[k];
    	}

  		delete[] src_r; 
  		delete[] src_i; 
  		delete[] dst_r;
  		delete[] dst_i;
		}
  }

}


//:::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
/*void solveStokesEqnsWithFFT(double dx, double dy, int n, double mu,
                            const MatrixXd &f1, 
														const MatrixXd &f2, 
														const MatrixXd &f3,
                            MatrixXd &u1, MatrixXd &u2, MatrixXd &p)
{
	MatrixXd u1_i(n, n), u2_i(n, n), p_i(n, n);

#pragma omp parallel for collapse(2)
	for(int i = 0; i < n; i++){
		for(int j = 0; j < n; j++){
			u1[i][j] = f1[i][j];
			u2[i][j] = f2[i][j];
			p[i][j] = f3[i][j];

			u1_i[i][j] = 0.0;
			u2_i[i][j] = 0.0;
			p_i[i][j] = 0.0;
		}
	}

	fft2d_mac(u1, u1_i, n, ZFFT_MAC_E1);
	fft2d_mac(u2, u2_i, n, ZFFT_MAC_E2);
	fft2d_mac(p, p_i, n, ZFFT_MAC_C);

  double dx2 = dx * dx;
  double dy2 = dy * dy;

  double h = 2.0*M_PI/n;
  double h_2 = 0.5*h;

	//double *cs = new double[n+1];
	//double *sn = new double[n+1];
	//for(int i = 0; i <= n; i++){
	//	cs[i] = cos(i*h);
	//	sn[i] = sin(i*h_2);
	//}

  //for (int r = 0; r < n; r++) {
  //  for (int s = 0; s < n; s++) {
  //    if ((r > 0) || (s > 0)) {

	//			// Fourier pseudo-spectral collocation

	//			//double k1 = r < n/2 ? r : n-r;
	//			//double k2 = s < n/2 ? s : n-s;
	//			//double c1 = (k1*k1+k2*k2)*h*h/(dx*dx);
	//			//double c2 = (k1*k1+k2*k2)*h*h/(dx*dx);
	//			//double c3 = k1*h/dx;
	//			//double c4 = k2*h/dx;

	//			// 2nd-order MAC finite difference method 

	//			//double sn1 = sin(r*h_2);
	//			//double sn2 = sin(s*h_2);
	//			//double cs1 = cos(r*h);
	//			//double cs2 = cos(s*h);

	//			//double c1 = (4.0-2.0*(cs1+cs2))/(dx*dx);
	//			//double c2 = (4.0-2.0*(cs1+cs2))/(dy*dy);
	//			//double c3 = 2.0*sn1/dx;
	//			//double c4 = 2.0*sn2/dy;

	//			double c3 = 2.0*sn[r]/dx;
	//			double c4 = 2.0*sn[s]/dy;
	//			double c1 = c3*c3+c4*c4;
	//			double c2 = c1;

	//			// 4th-order MAC finite difference method 

	//			//double c1 = (10.0-4.0*(cs1+cs2)-2.0*(cs1*cs2))/(3.0*dx*dx);
	//			//double c2 = (10.0-4.0*(cs1+cs2)-2.0*(cs1*cs2))/(3.0*dy*dy);
	//			//double c3 = sn1*(11.0+cs2)/(6.0*dx);
	//			//double c4 = sn2*(11.0+cs1)/(6.0*dy);

	//			///////////////////////////////////////////////////////////////////// 

	//			Complex g1(u1[r][s], u1_i[r][s]);
	//			Complex g2(u2[r][s], u2_i[r][s]);
	//			Complex g3(p[r][s], p_i[r][s]);

	//			Eigen::Matrix3cd mat;
	//			Eigen::Vector3cd rhs, sol;

	//			mat.real() << mu*c1, 0.0, 	0.0, 
	//										0.0, 	 mu*c2, 0.0, 
	//										0.0, 	 0.0, 	0.0;

	//			mat.imag() << 0.0, 0.0, c3, 
	//										0.0, 0.0, c4, 
	//										c3,  c4,  0.0;

	//			rhs << g1, g2, g3;

	//			sol = mat.householderQr().solve(rhs);

	//			u1[r][s] = sol[0].real();
	//			u2[r][s] = sol[1].real();
	//			p[r][s]  = sol[2].real();

	//			u1_i[r][s] = sol[0].imag();
	//			u2_i[r][s] = sol[1].imag();
	//			p_i[r][s]  = sol[2].imag();

  //    }
  //  }
  //}

	//delete[] cs;
	//delete[] sn;

	double *sn = new double[n+1];

	for(int i = 0; i <= n; i++){
		sn[i] = sin(i*h_2);
	}

#pragma omp parallel for collapse(2)
  for (int r = 0; r < n; r++) {
    for (int s = 0; s < n; s++) {
      if ((r > 0) || (s > 0)) {

				double alpha 	= sn[r]*2.0/dx;
				double beta 	= sn[s]*2.0/dy;
				double alpha2 = alpha * alpha;
				double beta2 	= beta * beta;
				double sum 		= alpha2 + beta2;
				double sum2 	= sum * sum;

				Complex mat[3][3], vec[3], sol[3];

				mat[0][0] = Complex(beta2, 0.0);
				mat[0][1] = Complex(-alpha*beta, 0.0);
				mat[0][2] = Complex(0.0, -alpha*sum);

				mat[1][1] = Complex(alpha2, 0.0);
				mat[1][2] = Complex(0.0, -beta*sum);

				mat[2][2] = Complex(sum2, 0.0);

				vec[0] = Complex(u1[r][s], u1_i[r][s]);
				vec[1] = Complex(u2[r][s], u2_i[r][s]);
				vec[2] = Complex(p[r][s], p_i[r][s]);

				for(int i = 0; i < 3; i++){
					for(int j = 0; j < i; j++){
						mat[i][j] = mat[j][i];
					}
					Complex s(0.0);
					for(int j = 0; j < 3; j++){
						s += mat[i][j] * vec[j];
					}
					sol[i] = s / sum2;
				}

				u1[r][s] = sol[0].real();
				u2[r][s] = sol[1].real();
				p[r][s]  = sol[2].real();

				u1_i[r][s] = sol[0].imag();
				u2_i[r][s] = sol[1].imag();
				p_i[r][s]  = sol[2].imag();

      }
    }
  }

	delete[] sn;

	u1[0][0] = u2[0][0] = p[0][0] = 0.0;
	u1_i[0][0] = u2_i[0][0] = p_i[0][0] = 0.0;

	ifft2d_mac(u1, u1_i, n, ZFFT_MAC_E1);
	ifft2d_mac(u2, u2_i, n, ZFFT_MAC_E2);
	ifft2d_mac(p, p_i, n, ZFFT_MAC_C);

}

//:::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
void solveStokesEqnsWithFFT3d(const TensorXd &f1, 
															const TensorXd &f2, 
															const TensorXd &f3,
															const TensorXd &g,
															double dx, double kappa, 
															int n0, int n1, int n2, 
                            	TensorXd &u1,
															TensorXd &u2,
															TensorXd &u3,
															TensorXd &p)
{
	int offset = 0;
	{
		int dim[3];
		g.get_dim(dim);
		if (dim[0] == n0+2) {
			offset = 1;
		}
	}
	//std::cout << "offset = " << offset << std::endl;

	TensorXd 
	u1_r(n0, n1, n2), 
	u2_r(n0, n1, n2), 
	u3_r(n0, n1, n2), 
	 p_r(n0, n1, n2),
	u1_i(n0, n1, n2), 
	u2_i(n0, n1, n2), 
	u3_i(n0, n1, n2), 
	 p_i(n0, n1, n2);

	u1_r.fill(0.0);
	u1_i.fill(0.0);
	u2_r.fill(0.0);
	u2_i.fill(0.0);
	u3_r.fill(0.0);
	u3_i.fill(0.0);
	 p_r.fill(0.0);
	 p_i.fill(0.0);

#pragma omp parallel for collapse(3)
	for(int i = 1; i < n0; i++){
		for(int j = 0; j < n1; j++){
			for(int k = 0; k < n2; k++){
				u1_r[i][j][k] = f1[i][j+offset][k+offset];
				u1_i[i][j][k] = 0.0;
			}
		}
	}
#pragma omp parallel for collapse(3)
	for(int i = 0; i < n0; i++){
		for(int j = 1; j < n1; j++){
			for(int k = 0; k < n2; k++){
				u2_r[i][j][k] = f2[i+offset][j][k+offset];
				u2_i[i][j][k] = 0.0;
			}
		}
	}
#pragma omp parallel for collapse(3)
	for(int i = 0; i < n0; i++){
		for(int j = 0; j < n1; j++){
			for(int k = 1; k < n2; k++){
				u3_r[i][j][k] = f3[i+offset][j+offset][k];
				u3_i[i][j][k] = 0.0;
			}
		}
	}
#pragma omp parallel for collapse(3)
	for(int i = 0; i < n0; i++){
		for(int j = 0; j < n1; j++){
			for(int k = 0; k < n2; k++){
				p_r[i][j][k] = g[i+offset][j+offset][k+offset];
				p_i[i][j][k] = 0.0;
			}
		}
	}

//#pragma omp parallel for collapse(3)
//	for(int i = 0; i < n0; i++){
//		for(int j = 0; j < n1; j++){
//			for(int k = 0; k < n2; k++){
//				u1_r[i][j][k] = f1[i][j+offset][k+offset];
//				u2_r[i][j][k] = f2[i+offset][j][k+offset];
//				u3_r[i][j][k] = f3[i+offset][j+offset][k];
//				 p_r[i][j][k] =  g[i+offset][j+offset][k+offset];
//
//				u1_i[i][j][k] = 0.0;
//				u2_i[i][j][k] = 0.0;
//				u3_i[i][j][k] = 0.0;
//				p_i[i][j][k] = 0.0;
//
//			}
//		}
//	}

	fft3d_mac(u1_r, u1_i, n0, n1, n2, ZFFT_MAC_E1);
	fft3d_mac(u2_r, u2_i, n0, n1, n2, ZFFT_MAC_E2);
	fft3d_mac(u3_r, u3_i, n0, n1, n2, ZFFT_MAC_E3);
	fft3d_mac( p_r,  p_i, n0, n1, n2, ZFFT_MAC_C);

  double dx2 = dx*dx;

  double h = 2.0*M_PI/n0;
  double h_2 = 0.5*h;

	double *cs = new double[n0+1];
	double *sn = new double[n0+1];
	for(int i = 0; i <= n0; i++){
		cs[i] = cos(i*h);
		sn[i] = sin(i*h_2);
	}

	//double scale = h/dx;
	double L = n0*dx;
	double scale = L/(2.0*M_PI);
	std::cout << "scale = " << scale << std::endl;

	bool modified = (fabs(kappa) > DBL_EPSILON);
	if (modified) {
		std::cout << "modified ." << std::endl;
	} else {
		std::cout << "not modified ." << std::endl;
	}

	int n0_2 = n0 >> 1;
	int n1_2 = n1 >> 1;
	int n2_2 = n2 >> 1;

#pragma omp parallel for collapse(3)
	for(int i = 0; i < n0; i++){
		for(int j = 0; j < n1; j++){
			for(int k = 0; k < n2; k++){
				//if (modified || i+j+k > 0) {
				if (i+j+k > 0) {

					// Fourier pseudo-spectral collocation

					//double k1 = (double) (i < n0_2 ? i : (i-n0));
					//double k2 = (double) (j < n1_2 ? j : (j-n1));
					//double k3 = (double) (k < n2_2 ? k : (k-n2));

					//double c1 = k1*scale;
					//double c2 = k2*scale;
					//double c3 = k3*scale;

					//double d1, d2, d3;
					//d1 = d2 = d3 = c1*c1 + c2*c2 + c3*c3 + kappa;

					// 2nd-order MAC finite difference method 

					double c1 = 2.0*sn[i]/dx;
					double c2 = 2.0*sn[j]/dx;
					double c3 = 2.0*sn[k]/dx;

					double d1, d2, d3;
					d1 = d2 = d3 = (c1*c1 + c2*c2 + c3*c3 + kappa);

					///////////////////////////////////////////////////////////////////// 

					Complex 
					g1(u1_r[i][j][k], u1_i[i][j][k]),
					g2(u2_r[i][j][k], u2_i[i][j][k]),
					g3(u3_r[i][j][k], u3_i[i][j][k]),
					g4( p_r[i][j][k],  p_i[i][j][k]);

					Eigen::Matrix4cd mat;
					Eigen::Vector4cd rhs, sol;

					mat.real() <<  d1, 0.0, 0.0, 0.0,
												0.0,  d2, 0.0, 0.0,
												0.0, 0.0,  d3, 0.0,
												0.0, 0.0, 0.0, 0.0;

					mat.imag() << 0.0, 0.0, 0.0,  c1, 
												0.0, 0.0, 0.0,  c2, 
												0.0, 0.0, 0.0,  c3, 
												-c1, -c2, -c3, 0.0;

					rhs << g1, g2, g3, g4;

					sol = mat.householderQr().solve(rhs);

					u1_r[i][j][k] = sol[0].real();
					u2_r[i][j][k] = sol[1].real();
					u3_r[i][j][k] = sol[2].real();
					 p_r[i][j][k] = sol[3].real();

					u1_i[i][j][k] = sol[0].imag();
					u2_i[i][j][k] = sol[1].imag();
					u3_i[i][j][k] = sol[2].imag();
					 p_i[i][j][k] = sol[3].imag();

					//if (i+j+k == 0) {
					//	std::cout << "kappa = " << kappa << std::endl;
					//	for(int m = 0; m < 4; m++){
					//		std::cout << rhs[m] << ", ";
					//	}
					//	std::cout << std::endl;
					//	for(int m = 0; m < 4; m++){
					//		std::cout << sol[m] << ", ";
					//	}
					//	std::cout << std::endl;
					//}

				} 

      }
    }
  }

	delete[] cs;
	delete[] sn;

	if (!modified) {
		u1_r[0][0][0] = u2_r[0][0][0] = u3_r[0][0][0];
		u1_i[0][0][0] = u2_i[0][0][0] = u3_i[0][0][0];
	}
	p_r[0][0][0] = p_i[0][0][0] = 0.0;


	ifft3d_mac(u1_r, u1_i, n0, n1, n2, ZFFT_MAC_E1);
	ifft3d_mac(u2_r, u2_i, n0, n1, n2, ZFFT_MAC_E2);
	ifft3d_mac(u3_r, u3_i, n0, n1, n2, ZFFT_MAC_E3);
	ifft3d_mac( p_r,  p_i, n0, n1, n2, ZFFT_MAC_C);

//#pragma omp parallel for collapse(3)
//	for(int i = 0; i < n0; i++){
//		for(int j = 0; j < n1; j++){
//			for(int k = 0; k < n1; k++){
//				u1[i][j+offset][k+offset] 				= u1_r[i][j][k];
//				u2[i+offset][j][k+offset]         = u2_r[i][j][k];
//				u3[i+offset][j+offset][k]         = u3_r[i][j][k];
//				 p[i+offset][j+offset][k+offset]  =  p_r[i][j][k];
//			}
//		}
//	}

#pragma omp parallel for collapse(3)
	for(int i = 1; i < n0; i++){
		for(int j = 0; j < n1; j++){
			for(int k = 0; k < n2; k++){
				u1[i][j+offset][k+offset] = u1_r[i][j][k];
			}
		}
	}
#pragma omp parallel for collapse(3)
	for(int i = 0; i < n0; i++){
		for(int j = 1; j < n1; j++){
			for(int k = 0; k < n2; k++){
				u2[i+offset][j][k+offset] = u2_r[i][j][k];
			}
		}
	}
#pragma omp parallel for collapse(3)
	for(int i = 0; i < n0; i++){
		for(int j = 0; j < n1; j++){
			for(int k = 1; k < n2; k++){
				u3[i+offset][j+offset][k] = u3_r[i][j][k];
			}
		}
	}
#pragma omp parallel for collapse(3)
	for(int i = 0; i < n0; i++){
		for(int j = 0; j < n1; j++){
			for(int k = 0; k < n2; k++){
				p[i+offset][j+offset][k+offset] = p_r[i][j][k];
			}
		}
	}

}*/


//+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
}	// functions start with namespace zfft
//+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
