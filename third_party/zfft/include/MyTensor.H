/*=============================================================================
*   
*   Filename : MyTensor.H
*   Creator : Han Zhou
*   Date : 06/15/21
*   Description : 
*
=============================================================================*/
   
#ifndef _MYTENSOR_H
#define _MYTENSOR_H

#include <cfloat>
#include <cassert>
#include <iostream>

#include "MyMatrix.H"

//+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
template<typename T>
class Tensor
{
	private :

		int _dim1 = 0;
		int _dim2 = 0;
		int _dim3 = 0;

		T ***_data = 0;

  private :

    void allocateMemory(int I, int J, int K)
    {
      size_t n1(I), n2(J), n3(K);
      size_t N1(n1), N2(n1*n2), N3(N2*n3);

      //size_t len = N1*sizeof(T**) + N2*sizeof(T*) + N3*sizeof(T);
      //_data = (T***)malloc(len);
      //T** ptr0 = (T**)(_data+N1);
      //T*  ptr1 = (T*)(ptr0+N2);

      _data = new T**[N1];
      T** ptr0 = new T*[N2];
      T*  ptr1 = new T[N3];
      
	    for(size_t i = 0; i < n1; i++){
	    	_data[i] = ptr0;
        ptr0 += n2;
        for(size_t j = 0; j < n2; j++){
          _data[i][j] = ptr1;
          ptr1 += n3;
        }
      }
    }

    void freeMemory(void)
    {
      //free(_data);

      delete[] _data[0][0];
      delete[] _data[0];
      delete[] _data;
      _data = 0;
    }

	public :

		virtual ~Tensor(void) {clear();}

		Tensor(void) 
    : _dim1(0), _dim2(0), _dim3(0), _data(0) 
    {}

		Tensor(int n) 
		: _dim1(n), _dim2(n), _dim3(n) 
		{
			allocate(n,n,n);
		}

		Tensor(int I, int J, int K) 
		: _dim1(I), _dim2(J), _dim3(K) 
		{
			allocate(I,J,K);
		}

		Tensor(int const size[3]) 
		: _dim1(size[0]), _dim2(size[1]), _dim3(size[2]) 
		{
			allocate(size[0], size[1], size[2]);
		}

		Tensor(int I, int J, int K, const T& a) 
		: _dim1(I), _dim2(J), _dim3(K) 
		{
			allocate(I,J,K);
			fill(a);
		}

		Tensor(Tensor const &m)
		{
			if (this != &m) {
				allocate(m._dim1, m._dim2, m._dim3);
#pragma omp parallel for
        for(size_t i = 0; i < m.elem_num(); i++){
		      _data[0][0][i] = m._data[0][0][i];
        }
			}
		}

	public :

		int dim1(void) const {return _dim1;}
		int dim2(void) const {return _dim2;}
		int dim3(void) const {return _dim3;}

		void get_dim(int dim[3]) const
    {
    	dim[0] = _dim1;
    	dim[1] = _dim2;
    	dim[2] = _dim3;
    }

		void get_dim(int &n1, int &n2, int &n3) const
    {
    	n1 = _dim1;
    	n2 = _dim2;
    	n3 = _dim3;
    }

    size_t elem_num(void) const 
    {
      size_t N1(_dim1), N2(_dim2), N3(_dim3);
      return N1*N2*N3;
    }

    T*** getDataPointer(void) {return _data;};
    T const*** getDataPointer(void) const {return _data;};

    T* getData00Pointer(void) {return _data[0][0];};
    T const* getData00Pointer(void) const {return _data[0][0];};

		void fill(const T &a)
    {
#pragma omp parallel for
      for(size_t i = 0; i < elem_num(); i++){
    	  _data[0][0][i] = a;
      }
    }

		void clear(void)
    {
    	if(0 != _data) {
        freeMemory();
    		_dim1 = _dim2 = _dim3 = 0;
    	}
    }

		void allocate(int I, int J, int K)
    {
	    _dim1 = I;
	    _dim2 = J;
	    _dim3 = K;
      allocateMemory(_dim1, _dim2, _dim3);
    }

		void reallocate(int I, int J, int K)
    {
    	if(0 != _data) {
    		if (_dim1==I && _dim2==J && _dim3==K) {
    			return;
    		} else {
          freeMemory();
    		}
    	}
      allocate(I, J, K);
    }

		T max(void) const
    {
    	T max_u = _data[0][0][0];
#pragma omp parallel for reduction(max: max_u)
      for(size_t i = 1; i < elem_num(); i++){
    		max_u = max_u > _data[0][0][i] ? max_u : _data[0][0][i];
      }
    	return max_u;
    }

		T min(void) const
    {
    	T min_u = _data[0][0][0];
#pragma omp parallel for reduction(min: min_u)
      for(size_t i = 1; i < elem_num(); i++){
    		min_u = min_u < _data[0][0][i] ? min_u : _data[0][0][i];
      }
    	return min_u;
    }

		T max_norm(void) const
    {
	    T max_u = fabs(_data[0][0][0]);
#pragma omp parallel for reduction(max: max_u)
      for(size_t i = 1; i < elem_num(); i++){
        T u = fabs(_data[0][0][i]);
	    	max_u = max_u > u ? max_u : u;
      }
	    return max_u;
    }

    //@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@

		T** &operator[](int i) 
    {
      return _data[i];
    }

		T** const &operator[](int i) const 
    {
      return _data[i];
    }

    T const &operator() (int i, int j, int k) const
    {
      assert(i >= 0 && i < _dim1 &&
             j >= 0 && j < _dim2 &&
             k >= 0 && k < _dim3);
      return _data[i][j][k];
    }

    T &operator() (int i, int j, int k)
    {
      assert(i >= 0 && i < _dim1 &&
             j >= 0 && j < _dim2 &&
             k >= 0 && k < _dim3);
      return _data[i][j][k];
    }

		Tensor &operator = (const Tensor &m)
    {
    	if (this != &m) {
    		reallocate(m._dim1, m._dim2, m._dim3);
#pragma omp parallel for
        for(size_t i = 0; i < m.elem_num(); i++){
          _data[0][0][i] = m._data[0][0][i];
        }
    	}
    	return *this;
    }

		Tensor &operator += (const Tensor &v)
		{
			assert(this->_dim1 == v._dim1 &&
			       this->_dim2 == v._dim2 &&
			       this->_dim3 == v._dim3);
#pragma omp parallel for
      for(size_t i = 0; i < elem_num(); i++){
		    this->_data[0][0][i] += v._data[0][0][i];
      }
			return *this;
		}

		Tensor &operator -= (const Tensor &v)
		{
			assert(this->_dim1 == v._dim1 &&
			       this->_dim2 == v._dim2 &&
			       this->_dim3 == v._dim3);
#pragma omp parallel for
      for(size_t i = 0; i < elem_num(); i++){
		    this->_data[0][0][i] -= v._data[0][0][i];
      }
			return *this;
		}
		Tensor &operator -= (double a)
		{
#pragma omp parallel for
      for(size_t i = 0; i < elem_num(); i++){
		    this->_data[0][0][i] -= a;
      }
			return *this;
		}

		Tensor &operator *= (double s)
		{
#pragma omp parallel for
      for(size_t i = 0; i < elem_num(); i++){
		    this->_data[0][0][i] *= s;
      }
			return *this;
		}

		Tensor &operator /= (double s)
		{
			assert(fabs(s) >= DBL_MIN);
#pragma omp parallel for
      for(size_t i = 0; i < elem_num(); i++){
		    this->_data[0][0][i] /= s;
      }
			return *this;
		}

		Tensor cwiseProduct(const Tensor &v) const
		{
			assert(_dim1 == v._dim1 &&
			       _dim2 == v._dim2 &&
			       _dim3 == v._dim3);

			Tensor v1(_dim1, _dim2, _dim3);
#pragma omp parallel for
      for(size_t i = 0; i < elem_num(); i++){
		    v1._data[0][0][i] = _data[0][0][i] * v._data[0][0][i];
      }
			return v1;
		}

		friend Tensor operator + (const Tensor &a, const Tensor &b)
		{
      assert(a._dim1 == b._dim1 &&
             a._dim2 == b._dim2 &&
             a._dim3 == b._dim3);

			Tensor v1(a._dim1, a._dim2, a._dim3);

#pragma omp parallel for
      for(size_t i = 0; i < a.elem_num(); i++){
			  v1._data[0][0][i] = a._data[0][0][i] + b._data[0][0][i];
      }

			return v1;
		}

		friend Tensor operator - (const Tensor &a, const Tensor &b)
		{
      assert(a._dim1 == b._dim1 &&
             a._dim2 == b._dim2 &&
             a._dim3 == b._dim3);

			Tensor v1(a._dim1, a._dim2, a._dim3);

#pragma omp parallel for
      for(size_t i = 0; i < a.elem_num(); i++){
			  v1._data[0][0][i] = a._data[0][0][i] - b._data[0][0][i];
      }
			return v1;
		}

		friend Tensor operator + (double s, const Tensor &a)
		{
			Tensor v1(a._dim1, a._dim2, a._dim3);

#pragma omp parallel for
      for(size_t i = 0; i < a.elem_num(); i++){
			  v1._data[0][0][i] = s + a._data[0][0][i];
      }
			return v1;
		}

		friend Tensor operator * (double s, const Tensor &a)
		{
			Tensor v1(a._dim1, a._dim2, a._dim3);

#pragma omp parallel for
      for(size_t i = 0; i < a.elem_num(); i++){
			  v1._data[0][0][i] = s*a._data[0][0][i];
      }
			return v1;
		}


		friend Tensor operator * (const Tensor &a, double s)
		{
			Tensor v1(a._dim1, a._dim2, a._dim3);

#pragma omp parallel for
      for(size_t i = 0; i < a.elem_num(); i++){
			  v1._data[0][0][i] = s*a._data[0][0][i];
      }
			return v1;
		}

		friend std::ostream &operator << (std::ostream &os, const Tensor &a)
		{
			for(int i = 0; i < a._dim1; i++){
				for(int j = 0; j < a._dim2; j++){
          for(int k = 0; k < a._dim3; k++){
            os << a._data[i][j][k];
            if (k == a._dim3 - 1) {
              std::cout << std::endl;
            } else {
              std::cout << " ";
            }
          }
				}
        if (i < a._dim1 - 1) {
          std::cout << std::endl;
        }
			}
			return os;
		}

};

//=============================================================================
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

#endif

