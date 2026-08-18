/**
 * ==========================================================================
 * CoQuí: Correlated Quantum ínterface
 *
 * Copyright (c) 2022-2025 Simons Foundation & The CoQuí developer team
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 * 
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * ==========================================================================
 */


#pragma once

#include "numerics/sparse/detail/concepts.hpp"
#include <cassert>
#include "numerics/sparse/detail/CPU/sparse.hpp"
#if defined(ENABLE_CUDA)
#include "numerics/sparse/detail/CUDA/cusparse_interface.hpp"
//#elif defined(ENABLE_HIP)
#endif

#include "utilities/check.hpp"
#include "numerics/detail/ops_aux.hpp"
#include "nda/nda.hpp"
#include "nda/blas.hpp"

namespace math::sparse
{

/***************************************************************************/
/*                              blas                                       */
/***************************************************************************/

template<char op_A, CSRMatrix A, ::nda::MemoryVector X, ::nda:: MemoryVector Y>
requires(::nda::have_same_value_type_v<X, Y> and
         ::nda::mem::have_compatible_addr_space<A, Y, Y> and
         ::nda::is_blas_lapack_v<::nda::get_value_t<X>>)
void csrmv(typename A::value_type alpha, A const& a, X const &x, typename A::value_type beta, Y &&y) {  
  using sfqmc::utils::check;
  static_assert( std::is_same_v<typename A::index_type,int> or std::is_same_v<typename A::index_type,long>, "Invalid type");
  static_assert( std::is_same_v<typename A::int_type,int> or std::is_same_v<typename A::int_type,long>, "Invalid type");

  check(math::is_valid_op(op_A), "Invalid operation: {}",op_A);
  auto [m, n] = a.shape();

  if(op_A == 'N') {
    check(m == y.extent(0), "Shape mismatch");
    check(n == x.extent(0), "Shape mismatch");
  } else {
    check(m == x.extent(0), "Shape mismatch");
    check(n == y.extent(0), "Shape mismatch");
  }

  // Must be lapack compatible
  check(x.indexmap().min_stride() == 1, "Stride mismatch");
  check(y.indexmap().min_stride() == 1, "Stride mismatch");

  if constexpr (::nda::mem::have_device_compatible_addr_space<A,X,Y>) {
#if defined(ENABLE_DEVICE)
    device::csrmv(op_A,alpha,a,x,beta,y);
#else
    check(false," csr_blas on device without gpu support! Compile for GPU. ");
#endif
  } else {
    cpu::csrmv(op_A, m, n, alpha, "GxxCxx", a.values().data(), a.columns().data(), 
               a.row_begin().data(), a.row_end().data(), x.data(), beta, y.data());
  }  
}

template<char op_A, CSRMatrix A, ::nda::MemoryVector X, ::nda:: MemoryVector Y>
requires(::nda::have_same_value_type_v<X, Y> and
         ::nda::mem::have_compatible_addr_space<A, Y, Y> and
         ::nda::is_blas_lapack_v<::nda::get_value_t<X>>)
void csrmv(A const& a, X const &x, Y &&y)
{
  using T = typename A::value_type;
  csrmm<op_A>(T(1.0),a,x,T(0.0),std::forward<Y>(y));
}

template<char op_A, CSRMatrix A, ::nda::MemoryMatrix B, ::nda::MemoryMatrix C>
requires(::nda::have_same_value_type_v<B, C> and
         ::nda::mem::have_compatible_addr_space<A, B, C> and
         ::nda::is_blas_lapack_v<::nda::get_value_t<C>>)
void csrmm(typename A::value_type alpha, B const& b, A const &a, typename A::value_type beta, C &&c) {
  static_assert( std::is_same_v<typename A::value_type,::nda::get_value_t<B>>, "Type mismatch.");
  static_assert((std::decay_t<B>::is_stride_order_C() and std::decay_t<C>::is_stride_order_C()) or
                (std::decay_t<B>::is_stride_order_Fortran() and std::decay_t<C>::is_stride_order_Fortran()));

  if constexpr (op_A == 'N' or op_A == 'T') {
    auto bt = ::nda::transpose(b);
    auto ct = ::nda::transpose(c);
 
    if(op_A == 'N') 
     csrmm<'T'>(alpha,a,bt,beta,ct);
    else
     csrmm<'N'>(alpha,a,bt,beta,ct);
  } else {
    // either A or B needs to be copied and conjugated. Which one???
    auto bdag = nda::make_regular(::nda::dagger(b));
    auto ct = ::nda::transpose(c);

    csrmm<'N'>(alpha,a,bdag,beta,ct);
    c() = nda::conj(c());
  }
}

template<char op_A, CSRMatrix A, ::nda::MemoryMatrix B, ::nda::MemoryMatrix C>
requires(::nda::have_same_value_type_v<B, C> and
         ::nda::mem::have_compatible_addr_space<A, B, C> and
         ::nda::is_blas_lapack_v<::nda::get_value_t<C>>)
void csrmm(B const& b, A const &a, C &&c)
{
  using T = typename A::value_type;
  csrmm<op_A>(T(1.0),b,a,T(0.0),std::forward<C>(c));
}

// Note: cuSparse supports opB on csrmm, but MKL does not.
//       Limiting current implementation to only op_A. 
//       If needed, implement custom backend for op_B != 'N' on CPU and enable.

template<char op_A, typename A, ::nda::MemoryMatrix B, ::nda::MemoryMatrix C>
requires(::nda::have_same_value_type_v<B, C> and
         ::nda::mem::have_compatible_addr_space<A, B, C> and
         ::nda::is_blas_lapack_v<::nda::get_value_t<C>>)
void csrmm(typename A::value_type alpha, A const& a, B const &b, typename A::value_type beta, C &&c) {          
  using sfqmc::utils::check;
  static_assert( std::is_same_v<typename A::index_type,int> or std::is_same_v<typename A::index_type,long>, "Invalid type");
  static_assert( std::is_same_v<typename A::int_type,int> or std::is_same_v<typename A::int_type,long>, "Invalid type");
  static_assert((nda::blas::has_C_layout<B> and nda::blas::has_C_layout<C>) or 
                (nda::blas::has_F_layout<B> and nda::blas::has_F_layout<C>), "Layout mismatch"); 

  auto [m, k] = a.shape();
  auto n = c.extent(1);
    
  if(op_A == 'N') {
    check(b.shape() == std::array<long,2>{k,n}, "Shape mismatch");
    check(c.shape() == std::array<long,2>{m,n}, "Shape mismatch");
  } else {
    check(b.shape() == std::array<long,2>{m,n}, "Shape mismatch");
    check(c.shape() == std::array<long,2>{k,n}, "Shape mismatch");
  }
    
  // Must be lapack compatible
  check(b.indexmap().min_stride() == 1, "Stride mismatch");
  check(c.indexmap().min_stride() == 1, "Stride mismatch");
    
    
  if constexpr (::nda::mem::have_device_compatible_addr_space<A,B,C>) {
#if defined(ENABLE_DEVICE)
    device::csrmm(op_A,'N',alpha,a,b,beta,c);
#else 
    check(false," csr_blas on device without gpu support! Compile for GPU. ");
#endif
  } else {
    if constexpr (std::decay_t<B>::is_stride_order_C()) {
      cpu::csrmm(op_A, m, n, k, alpha, "GxxCxx", a.values().data(), a.columns().data(), 
                 a.row_begin().data(), a.row_end().data(), b.data(), b.strides()[0], 0, 
                 beta, c.data(), c.strides()[0], 0, 1);
    } else if (std::decay_t<B>::is_stride_order_Fortran()) {
      cpu::csrmm(op_A, m, n, k, alpha, "GxxFxx", a.values().data(), a.columns().data(), 
                 a.row_begin().data(), a.row_end().data(), b.data(), b.strides()[1], 0, 
                 beta, c.data(), c.strides()[1], 0, 1);
    }
  }
}

template<char op_A, typename A, ::nda::MemoryMatrix B, ::nda::MemoryMatrix C>
requires(::nda::have_same_value_type_v<B, C> and
         ::nda::mem::have_compatible_addr_space<A, B, C> and
         ::nda::is_blas_lapack_v<::nda::get_value_t<C>>)
void csrmm(A const& a, B const &b, C &&c)
{
  using T = typename A::value_type;
  csrmm<op_A>(T(1.0),a,b,T(0.0),std::forward<C>(c));
}

template<char op_A, CSRMatrix A, nda::MemoryArrayOfRank<3> B, nda::MemoryArrayOfRank<3> C>
requires(::nda::have_same_value_type_v<B, C> and
         ::nda::is_blas_lapack_v<::nda::get_value_t<C>> and
         ::nda::mem::have_compatible_addr_space<A, B, C>) 
void csrmm(typename A::value_type alpha, A const& a, B const &b, typename A::value_type beta, C &&c) {          
  using sfqmc::utils::check;
  static_assert( std::is_same_v<typename A::index_type,int> or std::is_same_v<typename A::index_type,long>, "Invalid type");
  static_assert( std::is_same_v<typename A::int_type,int> or std::is_same_v<typename A::int_type,long>, "Invalid type");
  static_assert((nda::blas::has_C_layout<B> and nda::blas::has_C_layout<C>) or 
                (nda::blas::has_F_layout<B> and nda::blas::has_F_layout<C>), "Layout mismatch"); 

  // Must be lapack compatible
  check(b.indexmap().min_stride() == 1, "Stride mismatch");
  check(c.indexmap().min_stride() == 1, "Stride mismatch");
    
  auto [m, k] = a.shape();
  auto batchSize = ( nda::blas::has_C_layout<C> ? c.extent(0) : c.extent(2) );
  auto n = ( nda::blas::has_C_layout<C> ? c.extent(2) : c.extent(1) );

  if constexpr (nda::blas::has_C_layout<C>) {
 
    if constexpr (op_A == 'N') {
      check(b.shape() == std::array<long,3>{batchSize,k,n}, "Shape mismatch");
      check(c.shape() == std::array<long,3>{batchSize,m,n}, "Shape mismatch");
    } else {
      check(b.shape() == std::array<long,3>{batchSize,m,n}, "Shape mismatch");
      check(c.shape() == std::array<long,3>{batchSize,k,n}, "Shape mismatch");
    }

  } else {

    if constexpr (op_A == 'N') {
      check(b.shape() == std::array<long,3>{k,n,batchSize}, "Shape mismatch");
      check(c.shape() == std::array<long,3>{m,n,batchSize}, "Shape mismatch");
    } else {
      check(b.shape() == std::array<long,3>{m,n,batchSize}, "Shape mismatch");
      check(c.shape() == std::array<long,3>{k,n,batchSize}, "Shape mismatch");
    }

  }
    
  if constexpr (::nda::mem::have_device_compatible_addr_space<A,B,C>) {
#if defined(ENABLE_DEVICE)
    device::csrmm(op_A,'N',alpha,a,b,beta,c);
#else 
    check(false," csr_blas on device without gpu support! Compile for GPU. ");
#endif
  } else {

    if constexpr (nda::blas::has_C_layout<C>) {
      // op(A) * B = C
      cpu::csrmm(op_A, m, n, k, alpha, "GxxCxx", a.values().data(), a.columns().data(), 
                 a.row_begin().data(), a.row_end().data(), b.data(), b.strides()[1], b.strides()[0], 
                 beta, c.data(), c.strides()[1], c.strides()[0], batchSize);
    } else if (std::decay_t<B>::is_stride_order_Fortran()) {
      cpu::csrmm(op_A, m, n, k, alpha, "GxxFxx", a.values().data(), a.columns().data(),
                 a.row_begin().data(), a.row_end().data(), b.data(), b.strides()[1], b.strides()[2],
                 beta, c.data(), c.strides()[1], c.strides()[2], batchSize);
    }
  }

}

template<char op_A, CSRMatrix A, nda::MemoryArrayOfRank<3> B, nda::MemoryArrayOfRank<3> C>
requires(::nda::have_same_value_type_v<B, C> and
         ::nda::is_blas_lapack_v<::nda::get_value_t<C>> and
         ::nda::mem::have_compatible_addr_space<A, B, C>)
void csrmm(A const& a, B const &b, C &&c)
{
  using T = typename A::value_type;
  csrmm<op_A>(T(1.0),a,b,T(0.0),std::forward<C>(c));
}

}

