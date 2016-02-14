#pragma once

#include <cmath>

#include "base_matrix.h"

#ifdef NO_CUDA

#include <vector>
#include <functional>
#include <algorithm>
#include <boost/iterator/counting_iterator.hpp>
#include <boost/iterator/zip_iterator.hpp>
#include <boost/tuple/tuple.hpp>
#include <boost/phoenix/phoenix.hpp>
#include "phoenix_functions.h"
extern "C" {
#include <cblas.h>
}
namespace lib = std;
namespace iterlib = boost;

#else

#include <cublas_v2.h>   
#include <thrust/device_vector.h>
#include <thrust/functional.h>

#include "thrust_functions.h"

namespace lib = thrust;
namespace iterlib = thrust;

#endif

#include "strided_iterator.h"

namespace mblas {

#ifdef NO_CUDA
using namespace boost::phoenix::placeholders;
#else
using namespace thrust::placeholders;
#endif

template <class VecType>
class TMatrix : public BaseMatrix {
  public:
    typedef typename VecType::value_type value_type;
    typedef typename VecType::iterator iterator;
    typedef typename VecType::const_iterator const_iterator;
    
    TMatrix()
    : rows_(0), cols_(0)
    {}
    
    TMatrix(size_t rows, size_t cols)
    : rows_(rows), cols_(cols), data_(rows_ * cols_)
    {}
    
    TMatrix(size_t rows, size_t cols, value_type val)
    : rows_(rows), cols_(cols), data_(rows_ * cols_, val)
    {}
    
    TMatrix(TMatrix&& m)
    : rows_(m.rows_), cols_(m.cols_), data_(std::move(m.data_)) {}
        
    TMatrix(const TMatrix& m) = delete;
    
    value_type operator()(size_t i, size_t j) const {
      return data_[i * cols_ + j];
    }

    size_t Rows() const {
      return rows_;
    }
    
    size_t Cols() const {
      return cols_;
    }
    
    void Resize(size_t rows, size_t cols) {
      rows_ = rows;
      cols_ = cols;
      data_.resize(rows_ * cols_);
    }
    
    void Resize(size_t rows, size_t cols, value_type val) {
      rows_ = rows;
      cols_ = cols;
      data_.resize(rows_ * cols_, val);
    }
    
    void Reserve(size_t rows, size_t cols) {
      data_.reserve(rows * cols);
    }
    
    void Reshape(size_t rows, size_t cols) {
      rows_ = rows;
      cols_ = cols;
    }
    
    void Purge() {
      Clear();
      VecType temp;
      data_.swap(temp);
    }

    void Clear() {
      data_.clear();
      rows_ = 0;
      cols_ = 0;
    }
    
    VecType& GetVec() {
      return data_;
    }
    
    const VecType& GetVec() const {
      return data_;
    }
    
    value_type* data() {
#ifndef NO_CUDA
      return thrust::raw_pointer_cast(data_.data());
#else
      return data_.data();
#endif
    }
    
    const value_type* data() const {
#ifndef NO_CUDA
      return thrust::raw_pointer_cast(data_.data());
#else
      return data_.data();
#endif
    }
    
    iterator begin() {
      return data_.begin();
    }
    
    iterator end() {
      return data_.end();
    }
    
    const_iterator begin() const{
      return data_.begin();
    }
    
    const_iterator end() const {
      return data_.end();
    }
    
    size_t size() const {
      return data_.size();    
    }
    
  private:
    size_t rows_;
    size_t cols_;
    VecType data_;
};

#ifndef NO_CUDA
typedef thrust::device_vector<float> FVec;
typedef thrust::device_vector<unsigned int> IVec;
#else
typedef std::vector<float> FVec;
typedef std::vector<unsigned int> IVec;
#endif

typedef TMatrix<FVec> Matrix;
typedef TMatrix<IVec> IMatrix;

template <class M>
void debug1(const M& m, size_t pos = 0, size_t l = 5) {
  std::cerr << m.Rows() << " " << m.Cols() << std::endl;
  for(size_t i = 0; i < m.Rows(); ++i) {
    for(size_t j = pos; j < m.Cols() && j < pos + l; ++j) {
      std::cerr << m.GetVec()[i * m.Cols() + j] << " ";
    }
    std::cerr << std::endl;
  }
}

Matrix& Swap(Matrix& Out, Matrix& In) {
  size_t iRows = In.Rows();
  size_t iCols = In.Cols();
  size_t oRows = Out.Rows();
  size_t oCols = Out.Cols();
  
  Out.Reshape(iRows, iCols);
  In.Reshape(oRows, oCols);
  
  In.GetVec().swap(Out.GetVec());
  return Out;
}

Matrix& Transpose(Matrix& Out, const Matrix& In) {
  size_t m = In.Rows();
  size_t n = In.Cols();
  
  Out.Resize(n, m);
  
  float alpha = 1.0;
  float beta  = 0.0;
  
  thread_local cublasHandle_t handle;
  thread_local bool initialized;
  
  if(!initialized) {
    initialized = true;
    cublasCreate(&handle); //memory leak
  }
  
  cublasSgeam(handle, CUBLAS_OP_T, CUBLAS_OP_T, m, n, &alpha, In.data(), n,
              &beta, In.data(), n, Out.data(), m); 
  
  return Out;
}

Matrix& Transpose(Matrix& Out) {
  Matrix Temp;
  Transpose(Temp, Out);
  Swap(Out, Temp);
  return Out;
}

Matrix& Copy(Matrix& Out, const Matrix& In) {
  Out.Resize(In.Rows(), In.Cols());
  
  //thread_local cublasHandle_t handle;
  //thread_local bool initialized;
  //
  //if(!initialized) {
  //  initialized = true;
  //  cublasCreate(&handle); //memory leak
  //}
  //
  //cublasScopy(handle, In.size(), In.data(), 1, Out.data(), 1);

  lib::copy(In.begin(), In.end(), Out.begin());
  return Out;
}

Matrix& AppendRow(Matrix& Out, const Matrix& In, size_t i) {
  size_t oldSize = Out.GetVec().size();
  size_t addSize = In.Cols();
  Out.Resize(Out.Rows() + 1, In.Cols());
  Out.GetVec().resize(oldSize + addSize);
  size_t start = In.Cols() * i;
  size_t end   = In.Cols() * (i + 1);
  lib::copy(In.begin() + start, In.begin() + end, Out.begin() + oldSize);
  return Out;
}

Matrix& AppendRows(Matrix& Out, const Matrix& In) {
  size_t oldSize = Out.GetVec().size();
  size_t addSize = In.GetVec().size();
  Out.Resize(Out.Rows() + In.Rows(), In.Cols());
  Out.GetVec().resize(oldSize + addSize);
  lib::copy(In.begin(), In.end(), Out.begin() + oldSize);
  return Out;
}

Matrix& PrependRows(Matrix& Out, const Matrix& In) {
  Out.Resize(Out.Rows() + In.Rows(), In.Cols());
  Out.GetVec().insert(Out.begin(), In.begin(), In.end());
  return Out;
}

Matrix& PasteRow(Matrix& Out,
                 const Matrix& In,
                 const size_t r = 0, const size_t c = 0) {
  size_t start = r * Out.Cols() + c;
  lib::copy(In.begin(), In.end(), Out.begin() + start);
  return Out;
}

Matrix& CopyRow(Matrix& Out,
                const Matrix& In,
                const size_t r = 0, const size_t c = 0) {
  size_t length = In.Cols() - c;
  Out.Resize(1, length);
  size_t start = r * In.Cols() + c;
  size_t end   = start + length;
  lib::copy(In.begin() + start, In.begin() + end, Out.begin());
  return Out;
}

typedef std::pair<size_t, size_t> RowPair;
typedef std::vector<RowPair> RowPairs;
typedef thrust::device_vector<RowPair> DeviceRowPairs;

template <typename difference_type>
struct Functor : public func::unary_function<difference_type, difference_type> {
  const RowPair* rowPairs_;
  bool first_;
  difference_type rowLength_;

  Functor(const RowPair* rowPairs, bool first, difference_type rowLength)
  : rowPairs_(rowPairs), first_(first), rowLength_(rowLength) {}

#ifndef NO_CUDA
  __host__ __device__
#endif
  difference_type operator()(const difference_type& k) const {
    difference_type i = k / rowLength_;
    difference_type j = k % rowLength_;
    if(first_)
      return rowPairs_[i].first * rowLength_ + j;
    else
      return rowPairs_[i].second * rowLength_ + j;
  }
};

Matrix& CopyRows(Matrix& Out,
                 const Matrix& In,
                 const RowPair* devPairs,
                 size_t numPairs) {
  
  typedef FVec::iterator It;
  typedef typename iterlib::iterator_difference<It>::type difference_type;
  typedef typename iterlib::counting_iterator<difference_type> CountingIterator;
  typedef typename iterlib::transform_iterator<Functor<difference_type>, CountingIterator> TransformIterator;
  typedef typename iterlib::permutation_iterator<It, TransformIterator> PermutationIterator;
  
  typedef FVec::const_iterator CIt;
  typedef typename iterlib::iterator_difference<CIt>::type Cdifference_type;
  typedef typename iterlib::counting_iterator<Cdifference_type> CCountingIterator;
  typedef typename iterlib::transform_iterator<Functor<Cdifference_type>, CCountingIterator> CTransformIterator;
  typedef typename iterlib::permutation_iterator<CIt, CTransformIterator> CPermutationIterator;
  
  size_t rowLength = In.Cols();  
  CPermutationIterator itInBegin(In.begin(),
                                 CTransformIterator(CCountingIterator(0),
                                                    Functor<Cdifference_type>(devPairs, false, rowLength)));
  CPermutationIterator itInEnd = itInBegin + numPairs * rowLength;
  
  PermutationIterator itOutBegin(Out.begin(),
                                 TransformIterator(CountingIterator(0),
                                                   Functor<difference_type>(devPairs, true, rowLength)));
  
  lib::copy(itInBegin, itInEnd, itOutBegin);
  return Out;
}

Matrix& CopyRows(Matrix& Out,
                 const Matrix& In,
                 const RowPairs& pairs) {
  thrust::device_vector<RowPair> devPairs = pairs;
  CopyRows(Out, In, thrust::raw_pointer_cast(devPairs.data()), devPairs.size());
  return Out;
}

Matrix& Assemble(Matrix& Out,
                 const Matrix& In,
                 const std::vector<size_t>& indeces) {
  RowPairs rowPairs;
  for(size_t i = 0; i < indeces.size(); i++)
    rowPairs.emplace_back(i, indeces[i]);
  Out.Resize(rowPairs.size(), In.Cols());
  CopyRows(Out, In, rowPairs);
  return Out;
}

template <class F>
Matrix& PairwiseReduce(F f, Matrix& Out) {
  typedef FVec::iterator It;
  strided_range<It> evens(Out.begin(), Out.end(), 2);
  strided_range<It> odds(Out.begin() + 1, Out.end(), 2);
  lib::transform(evens.begin(), evens.end(), odds.begin(), Out.begin(), f);
  Out.Resize(Out.Rows(), Out.Cols() / 2);
  return Out;
}

template <class F>
Matrix& Element(F f, Matrix& Out) {
  lib::transform(Out.begin(), Out.end(), Out.begin(), f);
  return Out;
}

void fill1(RowPair* ptr, size_t r, size_t factor) {
  for(size_t i = 0; i < factor; ++i) {
    for(size_t j = 0; j < r; ++j) {
      ptr->first = i * r + j;
      ptr->second = j;
      ptr++;
    }
  }
}

void fill2(RowPair* ptr, size_t r, size_t factor) {
  size_t k = 0;
  for(size_t i = 0; i < r; ++i) {
    for(size_t j = 0; j < factor; ++j) {
      ptr->first = k++;
      ptr->second = i;
      ptr++;
    }
  }
}

template <class Filler>
void gfiller(Filler filler, RowPair* ptr, size_t r, size_t factor) {
  filler(ptr, r, factor);
}

template <class Filler>
Matrix& Broadcast(Filler filler, Matrix& Out, size_t factor) {
  size_t r = Out.Rows();
  size_t c = Out.Cols();
  
  size_t newRows = r * factor;
  
  Matrix Temp;
  Temp.Resize(newRows, c);
  
  // Try to put this on the device
  RowPairs rowPairs(newRows);
  gfiller(filler, rowPairs.data(), r, factor);
  CopyRows(Temp, Out, rowPairs);
  
  Swap(Out, Temp);
  
  return Out;
}

Matrix& Broadcast1(Matrix& Out, size_t factor) {  
  Broadcast(fill1, Out, factor);
  return Out;
}

Matrix& Broadcast2(Matrix& Out, size_t factor) {  
  Broadcast(fill2, Out, factor);
  return Out;
}

template <typename difference_type>
struct Functor1 : public func::unary_function<difference_type, difference_type> {
  difference_type rows;
  difference_type cols;

  Functor1(difference_type rows, difference_type cols)
  : rows(rows), cols(cols) {}

#ifndef NO_CUDA
  __host__ __device__
#endif
  difference_type operator()(const difference_type& i) const { 
      return i % (rows * cols);
  }
};

template <typename difference_type>
struct Functor2 : public func::unary_function<difference_type, difference_type> {
  difference_type rows;
  difference_type cols;

  Functor2(difference_type rows, difference_type cols)
  : rows(rows), cols(cols) {}

#ifndef NO_CUDA
  __host__ __device__
#endif
  difference_type operator()(const difference_type& i) const { 
      return cols * (i / (rows * cols)) + i % cols;
  }
};

template <class F>
Matrix& Element(F f, Matrix& Out, const Matrix& In) {
  if(Out.Rows() == In.Rows() && Out.Cols() == In.Cols()) {
    lib::transform(Out.begin(), Out.end(), In.begin(), Out.begin(), f);
  }
  // @TODO: fix bug here!
  else if(In.Rows() != Out.Rows() && Out.Cols() == In.Cols()) {
    typedef FVec::const_iterator It;
    typedef typename iterlib::iterator_difference<It>::type difference_type;
    typedef typename iterlib::counting_iterator<difference_type> CountingIterator;
    typedef typename iterlib::transform_iterator<Functor1<difference_type>, CountingIterator> TransformIterator1;
    typedef typename iterlib::permutation_iterator<It, TransformIterator1> PermutationIterator1;
    typedef typename iterlib::transform_iterator<Functor2<difference_type>, CountingIterator> TransformIterator2;
    typedef typename iterlib::permutation_iterator<It, TransformIterator2> PermutationIterator2;
  
    PermutationIterator1 it1(Out.begin(),
                             TransformIterator1(CountingIterator(0),
                                                Functor1<difference_type>(Out.Rows(), Out.Cols())));
    PermutationIterator2 it2(In.begin(),
                             TransformIterator2(CountingIterator(0),
                                                Functor2<difference_type>(Out.Rows(), Out.Cols())));
    PermutationIterator1 end1 = it1 + (Out.end() - Out.begin()) * In.Rows();
    
    Out.Resize(Out.Rows() * In.Rows(), Out.Cols());
    lib::transform(it1, end1, it2, Out.begin(), f);
  }
  // TODO: finish this
  else if(In.Cols() == 1 && Out.Rows() == In.Rows()) {
    
    typedef FVec::const_iterator It;
    col_repeater<It> repeater(In.begin(), In.end(), Out.Cols());
    lib::transform(Out.begin(), Out.end(), repeater.begin(), Out.begin(), f);
  }
  return Out;
}

template <class F, typename T>
struct Tuple3Func {
    Tuple3Func(F f) : f_(f) {}
  
#ifndef NO_CUDA
  __host__ __device__
#endif
  T operator()(const iterlib::tuple<T, T, T>& t) const { 
    return f_(iterlib::get<0>(t), iterlib::get<1>(t), iterlib::get<2>(t));
  }

  F f_;
};

template <class F>
Matrix& Element(F f, Matrix& Out, const Matrix& In1, const Matrix& In2) {
  typedef FVec::const_iterator CIt;
  typedef iterlib::tuple<CIt, CIt, CIt> IteratorTuple;
  typedef iterlib::zip_iterator<IteratorTuple> ZipIterator;
  
  ZipIterator begin(iterlib::make_tuple(Out.begin(), In1.begin(), In2.begin()));
  ZipIterator end(iterlib::make_tuple(Out.end(), In1.end(), In2.end()));

  Tuple3Func<F, Matrix::value_type> t3f(f);
  
  lib::transform(begin, end, Out.begin(), t3f);
  return Out;
}

Matrix& Prod(Matrix& C, const Matrix& A, const Matrix& B,
             bool transA = false, bool transB = false) {
  Matrix::value_type alpha = 1.0;
  Matrix::value_type beta = 0.0;

  size_t m = A.Rows();
  size_t k = A.Cols();
  if(transA)
    std::swap(m, k);
  
  size_t l = B.Rows();
  size_t n = B.Cols();
  if(transB)
    std::swap(l, n);
  
  size_t lda = A.Cols();                                                                              
  size_t ldb = B.Cols();                                                                              
  size_t ldc = B.Cols();

  if(transB)
    ldc = B.Rows();  
  
  C.Resize(m, n);
  
#ifndef NO_CUDA
  cublasOperation_t opA = transA ? CUBLAS_OP_T : CUBLAS_OP_N;
  cublasOperation_t opB = transB ? CUBLAS_OP_T : CUBLAS_OP_N;

  thread_local cublasHandle_t handle;
  thread_local bool initialized;
  
  if(!initialized) {
    initialized = true;
    cublasCreate(&handle); 
  }
  
  cublasSgemm(handle, opB, opA,
              n, m, k, &alpha, B.data(), ldb, A.data(), lda, &beta, C.data(), ldc);
  
#else
  CBLAS_TRANSPOSE opA = transA ? CblasTrans : CblasNoTrans;
  CBLAS_TRANSPOSE opB = transB ? CblasTrans : CblasNoTrans;

  cblas_sgemm(CblasColMajor, opB, opA,
              n, m, k, alpha, B.data(), ldb, A.data(), lda, beta, C.data(), ldc);
#endif

  return C;
}

mblas::Matrix& SoftmaxRows(mblas::Matrix& In,
                           mblas::Matrix& Ones,
                           mblas::Matrix& Sums) {
  using namespace mblas;
  
  Element(Exp(_1), In);
  Ones.Resize(In.Cols(), 1, 1.0);
  Prod(Sums, In, Ones);
  Element(_1 / _2, In, Sums); // Broadcasting col-wise
  
  return In;
}

}