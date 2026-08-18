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


#ifndef SPARSE_CUDA_GPU_HPP
#define SPARSE_CUDA_GPU_HPP

#include <type_traits>
#include <cassert>
#include <vector>
#include <complex>
#include <string>

#include "configuration.hpp"
#include "IO/AppAbort.hpp"
#include "utilities/check.hpp"

#include "numerics/sparse/detail/CUDA/cusparse_aux.hpp"
#include "numerics/detail/ops_aux.hpp"

namespace math::sparse::device
{

// defined in cusparse_interface.cpp
cusparseHandle_t &get_cusparse_handle_ptr();

// MAM: csr_matrix stores row pointers in host, so right now pntrb/pntre are host pointers.
//      They are copied to device on the fly here. Write alternative routine that takes
//      the (compact) arrays in device
template<CSRMatrix A, ::nda::MemoryVector X, ::nda:: MemoryVector Y>
requires(::nda::have_same_value_type_v<X, Y> and
         ::nda::mem::have_device_compatible_addr_space<A,X,Y>) 
void csrmv(char oper_A,typename A::value_type alpha, A const& a, X const &x, typename A::value_type beta, Y &&y)
{
  sfqmc::utils::check(math::is_valid_op(oper_A), "Invalid operation: {}",oper_A);
  auto [m, n] = a.shape();
 
  constexpr MEMORY_SPACE MEM = A::mem_type; 
  using value_type = std::decay_t<typename A::value_type>;
  using int_type   = std::decay_t<typename A::int_type>;
  static_assert( std::is_same_v<value_type,::nda::get_value_t<X>>, "value_type mismatch.");
  static_assert( std::is_same_v<value_type,::nda::get_value_t<Y>>, "value_type mismatch.");

  auto handle = get_cusparse_handle_ptr(); 
  auto cuX = cuDn(x);  
  auto cuY = cuDn(y);  

  memory::buffered_array<MEM,int_type,1> ofs(m+1,int_type(0));
  auto op_A = get_operation<value_type>(oper_A);
  auto cuA = cuCSR(a,ofs);
   
  // allocate an external buffer if needed
  size_t bufferSize = 0;
  CUSPARSE_CHECK( cusparseSpMV_bufferSize, handle, op_A, 
                  &alpha, cuA, cuX, &beta, cuY, cusparse_datatype<value_type>,
                  CUSPARSE_SPMV_ALG_DEFAULT, &bufferSize) 
  memory::buffered_array<MEM,char,1> buffer(bufferSize,char(0));

  // execute preprocess (optional)
//  CUSPARSE_CHECK( cusparseSpMV_preprocess, handle, op_A, 
//                  &alpha, cuA, cuX, &beta, cuY, cusparse_datatype<value_type>,
//                  CUSPARSE_SPMV_ALG_DEFAULT, (void*) buffer.data() )

  // execute SpMV
  CUSPARSE_CHECK( cusparseSpMV, handle, op_A, 
                  &alpha, cuA, cuX, &beta, cuY, cusparse_datatype<value_type>,
                  CUSPARSE_SPMV_ALG_DEFAULT, (void*) buffer.data() )

  CUSPARSE_CHECK( cusparseDestroySpMat, cuA )
  CUSPARSE_CHECK( cusparseDestroyDnVec, cuX )
  CUSPARSE_CHECK( cusparseDestroyDnVec, cuY )
}

template<CSRMatrix A, typename B, typename C> 
requires((::nda::MemoryMatrix<B> or ::nda::MemoryArrayOfRank<B,3>) and   
         (::nda::MemoryMatrix<C> or ::nda::MemoryArrayOfRank<C,3>) and   
         (::nda::get_rank<B> == ::nda::get_rank<C>) and
         ::nda::mem::have_device_compatible_addr_space<A,B,C> and 
         ::nda::have_same_value_type_v<B,C>)
void csrmm(char oper_A, char oper_B, typename A::value_type alpha, A const& a, B const &b, typename A::value_type beta, C &&c) 
{ 
  sfqmc::utils::check(math::is_valid_op(oper_A), "Invalid operation: {}",oper_A);
  auto [m, n] = a.shape();
  
  constexpr MEMORY_SPACE MEM = A::mem_type; 
  using value_type = std::decay_t<typename A::value_type>;
  using int_type   = std::decay_t<typename A::int_type>;

  if constexpr (std::is_same_v<::nda::get_value_t<B>,std::complex<value_type>>) {
    static_assert(std::decay_t<B>::is_stride_order_C() and std::decay_t<C>::is_stride_order_C(),
        "Mixed real/complex csrmm only with row-major matrices.");
    sfqmc::utils::check(b.indexmap().min_stride() == 1, "Stride mismatch");
    sfqmc::utils::check(c.indexmap().min_stride() == 1, "Stride mismatch");
    // can bypass this by constructing indexmap with array strides
    sfqmc::utils::check(b.is_contiguous(), "Layout mismatch");
    sfqmc::utils::check(c.is_contiguous(), "Layout mismatch");
    if constexpr (::nda::MemoryMatrix<B>) {
      memory::array_view< memory::get_memory_space<B>(), const value_type, 2, typename B::layout_policy_t> b_(std::array<long,2>{b.extent(0),2*b.extent(1)},reinterpret_cast<const value_type*>(b.data()));
      memory::array_view< memory::get_memory_space<C>(), value_type, 2, typename std::decay_t<C>::layout_policy_t> c_(std::array<long,2>{c.extent(0),2*c.extent(1)},reinterpret_cast<value_type*>(c.data()));
      csrmm(oper_A,oper_B,alpha,a,b_,beta,c_);
    } else {
      memory::array_view< memory::get_memory_space<B>(), const value_type, 3, typename B::layout_policy_t> b_(std::array<long,3>{b.extent(0),b.extent(1),2*b.extent(2)},reinterpret_cast<const value_type*>(b.data()));
      memory::array_view< memory::get_memory_space<C>(), value_type, 3, typename std::decay_t<C>::layout_policy_t> c_(std::array<long,3>{c.extent(0),c.extent(1),2*c.extent(2)},reinterpret_cast<value_type*>(c.data()));
      csrmm(oper_A,oper_B,alpha,a,b_,beta,c_);
    }
    return;
  } else {
    static_assert( std::is_same_v<value_type,::nda::get_value_t<B>>, "value_type mismatch.");
    static_assert( std::is_same_v<value_type,::nda::get_value_t<C>>, "value_type mismatch.");
  }
  
  auto handle = get_cusparse_handle_ptr();
  // not enabled yet. Take as argument if needed and implement custom backend in cpu.
  auto op_A = get_operation<value_type>(oper_A);
  auto op_B = get_operation<value_type>(oper_B);
  auto cuB = cuDn(b);
  auto cuC = cuDn(c);

  int batchCountB=1, batchCountC=1;
  int64_t batchStride=0;
  CUSPARSE_CHECK( cusparseDnMatGetStridedBatch, cuB, &batchCountB, &batchStride ); 
  CUSPARSE_CHECK( cusparseDnMatGetStridedBatch, cuC, &batchCountC, &batchStride ); 
  sfqmc::utils::check(batchCountB == batchCountC, "Batch count mismatch.");
  
  memory::buffered_array<MEM,int_type,1> ofs(m+1,int_type(0));
  auto cuA = cuCSR(a,ofs,batchCountB);

  // allocate an external buffer if needed
  size_t bufferSize = 0;
  CUSPARSE_CHECK( cusparseSpMM_bufferSize, handle, op_A, op_B, 
                  &alpha, cuA, cuB, &beta, cuC, cusparse_datatype<value_type>,
                  CUSPARSE_SPMM_CSR_ALG2, &bufferSize)
  memory::buffered_array<MEM,char,1> buffer(bufferSize,char{0});
  
  // execute preprocess (optional)
  CUSPARSE_CHECK( cusparseSpMM_preprocess, handle, op_A, op_B, 
                  &alpha, cuA, cuB, &beta, cuC, cusparse_datatype<value_type>,
                  CUSPARSE_SPMM_CSR_ALG2, (void*) buffer.data() )
  
  // execute SpMM
  CUSPARSE_CHECK( cusparseSpMM, handle, op_A, op_B, 
                  &alpha, cuA, cuB, &beta, cuC, cusparse_datatype<value_type>,
                  CUSPARSE_SPMM_CSR_ALG2, (void*) buffer.data() )
  
  CUSPARSE_CHECK( cusparseDestroySpMat, cuA )
  CUSPARSE_CHECK( cusparseDestroyDnMat, cuB )
  CUSPARSE_CHECK( cusparseDestroyDnMat, cuC )
}

} // namespace math::sparse::device


#endif
