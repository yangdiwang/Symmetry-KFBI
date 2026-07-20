/*=============================================================================
*   
*   Filename : MyMatrix.H
*   Creator : Han Zhou
*   Date : 06/15/21
*   Description : 
*
=============================================================================*/
   
#ifndef _MYMATRIX_H
#define _MYMATRIX_H

#include <cfloat>
#include <cassert>
#include <iostream>

template<typename T>
class Matrix
{
	private :

		int _row = 0;
		int _col = 0;

		T **_data = 0;

  private :

    void allocateMemory(int I, int J)
    {
      size_t n1(I), n2(J);
      size_t N1(n1), N2(n1*n2);

      //size_t len = N1*sizeof(T*) + N2*sizeof(T);
      //_data = (T**)malloc(len);
      //T* ptr = (T*)(_data+N1);

      _data = new T*[N1];
      T* ptr = new T[N2];

      for(size_t i = 0; i < n1; i++){
        _data[i] = ptr;
        ptr += n2;
      }
    }

    void freeMemory(void)
    {
      //free(_data);

      delete[] _data[0];
      delete[] _data;
      _data = 0;
    }

	public :

		virtual ~Matrix(void) {clear();}

		Matrix(void) 
    : _row(0), _col(0), _data(0) 
    {}

		Matrix(int n) 
		: _row(n), _col(n) 
		{
			allocate(n, n);
		} 

		Matrix(int m, int n) 
		: _row(m), _col(n) 
		{
			allocate(m, n);
		} 

		Matrix(int m, int n, const T &a)
		: _row(m), _col(n)
		{
			allocate(m, n);
			fill(a);
		}

		Matrix(const Matrix &m)
		{
			if (this != &m) {
				allocate(m._row, m._col);
#pragma omp parallel for
        for(size_t i = 0; i < elem_num(); i++){
          _data[0][i] = m._data[0][i];
        }
			}
		}

	public :

		int rows(void) const {return _row;}
		int cols(void) const {return _col;}

    size_t elem_num(void) const 
    {
      size_t N1(_row), N2(_col);
      return (N1*N2);
    }

    T** getDataPointer(void) {return _data;};
    T const** getDataPointer(void) const {return _data;};

    T* getData0Pointer(void) {return _data[0];};
    T const* getData0Pointer(void) const {return _data[0];};

		void fill(const T &a)
    {
#pragma omp parallel for
      for(size_t i = 0; i < elem_num(); i++){
        _data[0][i] = a;
      }
    }

		void clear(void)
    {
	    if(0 != _data) {
        freeMemory();
	    	_row = _col = 0;
	    }
    }

		void allocate(int m, int n)
    {
	    _row = m;
	    _col = n;
      allocateMemory(m, n);
    }

		void reallocate(int m, int n)
    {
	    if (_data != 0) {
	    	if (_row == m && _col == n) {
	    		return;
	    	} else {
          freeMemory();
	    	}
	    }
      allocate(m, n);
    }

		T max(void) const 
    {
	    T max_u = _data[0][0];
#pragma omp parallel for reduction(max: max_u)
      for(size_t i = 1; i < elem_num(); i++){
	      max_u = max_u > _data[0][i] ? max_u : _data[0][i];
	    }
      return max_u;
    }

		T min(void) const 
    {
	    T min_u = _data[0][0];
#pragma omp parallel for reduction(min: min_u)
      for(size_t i = 1; i < elem_num(); i++){
	      min_u = min_u < _data[0][i] ? min_u : _data[0][i];
	    }
      return min_u;
    }

    T sum(void) const
    {
      T sum = 0;
      T comp = 0;

      for(size_t i = 0; i < elem_num(); i++){
        T x = _data[0][i];
        T t = sum + x;
        if (fabs(sum) >= fabs(x)) {
          comp += (sum - t) + x;
        } else {
          comp += (x - t) + sum;
        }
        sum = t;
      }
      return sum + comp;
    }

    T mean(void) const
    {
      return (1.0/elem_num()) * sum();
    }

		T max_norm(void) const
    {
	    T max_u = fabs(_data[0][0]);
#pragma omp parallel for reduction(max: max_u)
      for(size_t i = 1; i < elem_num(); i++){
	    	T u = fabs(_data[0][i]);
	    	max_u = max_u > u ? max_u : u;
      }
	    return max_u;
    }

		T l1_norm(void) const
    {
	    T sum = 0.0;
#pragma omp parallel for
      for(size_t i = 0; i < elem_num(); i++){
	    		sum += fabs(_data[0][i]);
      }
	    return sum;
    }

		T l2_norm(void) const
    {
	    T sum = 0.0;
#pragma omp parallel for
      for(size_t i = 0; i < elem_num(); i++){
			  sum += _data[0][i] * _data[0][i];
      }
	    return sqrt(sum);
    }

		T l1_norm_scaled(void) const
    {
    	T sum = 0.0;
      size_t N = elem_num();
#pragma omp parallel for
      for(size_t i = 0; i < N; i++){
    		sum += fabs(_data[0][i]);
    	}
      return (sum/N);
    }

		T l2_norm_scaled(void) const
    {
	    T sum = 0.0;
      size_t N = elem_num();
#pragma omp parallel for
      for(size_t i = 0; i < N; i++){
	    	sum += _data[0][i] * _data[0][i];
    	}
      return sqrt(sum/N);
    }

		//T max_error(const Matrix &u) const;
		//T l1_error_scaled(const Matrix &u) const;
		//T l2_error_scaled(const Matrix &u) const;
		//T inner_product(const Matrix &u) const;

    //@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@

		T*& operator[](int i) {return _data[i];}
		T* const &operator[](int i) const {return _data[i];}

    T const &operator() (int i, int j) const
    {
      assert(i >= 0 && i < _row && 
             j >= 0 && j < _col);
      return _data[i][j];
    }

    T &operator() (int i, int j)
    {
      assert(i >= 0 && i < _row && 
             j >= 0 && j < _col);
      return _data[i][j];
    }

		Matrix &operator = (const Matrix &m)
    {
    	if (this != &m) {
    		reallocate(m._row, m._col);
#pragma omp parallel for
        for(size_t i = 0; i < elem_num(); i++){
          _data[0][i] = m._data[0][i];
        }
    	}
    	return *this;
    }

		Matrix &operator += (const Matrix &v)
		{
			assert(this->_row == v._row &&
			       this->_col == v._col);
#pragma omp parallel for 
      for(size_t i = 0; i < elem_num(); i++){
				this->_data[0][i] += v._data[0][i];
      }
			return *this;
		}

		Matrix &operator -= (const Matrix &v)
		{
			assert(this->_row == v._row &&
			       this->_col == v._col);
#pragma omp parallel for 
      for(size_t i = 0; i < elem_num(); i++){
				this->_data[0][i] -= v._data[0][i];
			}
			return *this;
		}

		Matrix &operator *= (double s)
		{
#pragma omp parallel for 
      for(size_t i = 0; i < elem_num(); i++){
				this->_data[0][i] *= s;
			}
			return *this;
		}

		Matrix &operator /= (double s)
		{
			assert(fabs(s) >= DBL_MIN);
      double one_s = 1.0/s;
#pragma omp parallel for 
      for(size_t i = 0; i < elem_num(); i++){
				this->_data[0][i] *= one_s;
			}
			return *this;
		}

		Matrix cwiseProduct(const Matrix &v) const
		{
			assert(_row == v._row &&
			       _col == v._col);

			Matrix v1(_row, _col);
#pragma omp parallel for 
      for(size_t i = 0; i < elem_num(); i++){
				v1._data[0][i] = _data[0][i] * v._data[0][i];
      }
			return v1;
		}

    friend double fabs(const Matrix &a)
    {
      double res = 0.0;
      for(size_t d = 0; d < a.elem_num(); d++){
        res = std::max(res, fabs(a._data[0][d]));
      }
      return res;
    }

		friend Matrix operator + (const Matrix &a, const Matrix &b)
		{
      assert(a._row == b._row &&
             a._col == b._col);

			Matrix v1(a._row, a._col);
#pragma omp parallel for 
      for(size_t i = 0; i < v1.elem_num(); i++){
				v1._data[0][i] = a._data[0][i] + b._data[0][i];
			}
			return v1;
		}

		friend Matrix operator - (const Matrix &a, const Matrix &b)
		{
      assert(a._row == b._row &&
             a._col == b._col);

			Matrix v1(a._row, a._col);
#pragma omp parallel for 
      for(size_t i = 0; i < v1.elem_num(); i++){
				v1._data[0][i] = a._data[0][i] - b._data[0][i];
			}
			return v1;
		}

		friend Matrix operator + (double s, const Matrix &a)
		{
			Matrix v1(a._row, a._col);
#pragma omp parallel for 
      for(size_t i = 0; i < v1.elem_num(); i++){
				v1._data[0][i] = s + a._data[0][i];
			}
			return v1;
		}

		friend Matrix operator * (double s, const Matrix &a)
		{
			Matrix v1(a._row, a._col);
#pragma omp parallel for 
      for(size_t i = 0; i < v1.elem_num(); i++){
				v1._data[0][i] = s * a._data[0][i];
			}
			return v1;
		}

		friend Matrix operator * (const Matrix &a, double s)
		{
			Matrix v1(a._row, a._col);
#pragma omp parallel for 
      for(size_t i = 0; i < v1.elem_num(); i++){
				v1._data[0][i] = s * a._data[0][i];
			}
			return v1;
		}

		friend std::ostream &operator << (std::ostream &os, const Matrix &a)
		{
			for(int i = 0; i < a._row; i++){
				for(int j = 0; j < a._col-1; j++){
					os << a._data[i][j] << " ";
				}
				os << a._data[i][a._col-1] << std::endl;
			}
			std::cout << std::endl;
			return os;
		}

#ifdef DATA_DEBUG
		void print(void) const;
#endif

};

#ifdef DATA_DEBUG
//:::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
template<typename T>
inline void Matrix<T>::print(void) const
{
	if (std::is_same<T, double>::value || 
			std::is_same<T, float>::value) {

  	for (int i = 0; i < _row; i++) {
  	  for (int j = 0; j < _col; j++) {
  	    T u = _data[i][j]; 
  	    if (fabs(u) < DBL_MIN) {
  	      //printf("          "); 
  	      printf("         0"); 
  	    } else {
  	      printf("%8.4f  ", u);
  	    }
  	  }
  	  printf("\n");
  	}
  	printf("\n"); 

	} else if (std::is_same<T, int>::value || 
						 std::is_same<T, unsigned int>::value || 
						 std::is_same<T, long unsigned int>::value) {

  	for (int i = 0; i < _row; i++) {
  	  for (int j = 0; j < _col; j++) {
  	    T u = _data[i][j]; 
  	    //if (u == 0) {
  	    //  printf("        "); 
  	    //} else {
  	    //  printf("%6d  ", u);
  	    //}
  	      printf("%6d  ", u);
  	  }
  	  printf("\n");
  	}
  	printf("\n"); 

	}

}
#endif

//=============================================================================
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

#endif
