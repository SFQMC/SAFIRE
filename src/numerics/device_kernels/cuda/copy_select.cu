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


//////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////

#include "stdio.h"
#include <complex>
#include <algorithm>

#include "configuration.hpp"
#include "utilities/check.hpp"
#include "utilities/type_traits.hpp"
#include "numerics/device_kernels/cuda/cuda_settings.h"
#include "numerics/device_kernels/cuda/cuda_aux.hpp"
#include "nda/nda.hpp"
#include <cuda/std/mdspan>
#include "cub/device/device_for.cuh"
#include "thrust/for_each.h"
#include "thrust/iterator/counting_iterator.h"

namespace kernels::device::detail
{

template<typename V1, typename V3, typename V4, typename T>
void copy_select_impl(bool expand, int dim, V1 const& m, T alpha, V3 const& A, T scl, V4&& B)
{
  auto m_d = to_cuda_std_mdspan(m);
  auto A_d = to_cuda_std_mdspan(A);
  auto B_d = to_cuda_std_mdspan(B);
  auto scl_d = cuda_std_value_cast(scl);
  auto alpha_d = cuda_std_value_cast(alpha);
  int M = (expand?A.extent(0):B.extent(0));
  int N = (expand?A.extent(1):B.extent(1));
  if( dim == 0 ) {
    if(expand) {
      auto f = [=] __device__(long n) {
        long i = n/N;
        long j = n - i*N;
        B_d(m_d(i),j) = scl_d*B_d(m_d(i),j) + alpha_d * A_d(i,j);
      };
      cub::DeviceFor::Bulk(N*M,f);
    } else {
      auto f = [=] __device__(long n) {
        long i = n/N;
        long j = n - i*N;
        B_d(i,j) = scl_d*B_d(i,j) + alpha_d * A_d(m_d(i),j);
      };
      cub::DeviceFor::Bulk(N*M,f);
    }
  } else if( dim == 1 ) {
    if(expand) {
/*
      auto f = [=] __device__(long n, long i, long j) {
        B_d(i,m_d(j)) = scl_d*B_d(i,m_d(j)) + A_d(i,j);
      };
      cuda::std::array<long,2> shape = {M,N};
      cuda::std::dextents<long,2> extents(shape);
      MAM: ForEachInExtents seems broken to me, calls elements more than once
      cub::DeviceFor::ForEachInExtents(extents, f);
*/
      auto f = [=] __device__(long n) {
        long i = n/N;
        long j = n - i*N;
        B_d(i,m_d(j)) = scl_d*B_d(i,m_d(j)) + alpha_d * A_d(i,j);
      };
      cub::DeviceFor::Bulk(N*M,f);
    } else {
      auto f = [=] __device__(long n) {
        long i = n/N;
        long j = n - i*N;
        B_d(i,j) = scl_d*B_d(i,j) + alpha_d * A_d(i,m_d(j));
      };
      cub::DeviceFor::Bulk(N*M,f);
    }
  }
}

template<typename V1, typename V2, typename V3, typename V4, typename T>
void copy_select_impl(bool expand, int dim, V1 const& m, V2 const& s, T alpha, V3 const& A, T scl, V4&& B)
{ 
  auto m_d = to_cuda_std_mdspan(m);
  auto s_d = to_cuda_std_mdspan(s);
  auto A_d = to_cuda_std_mdspan(A);
  auto B_d = to_cuda_std_mdspan(B);
  auto scl_d = cuda_std_value_cast(scl);
  auto alpha_d = cuda_std_value_cast(alpha);
  int M = (expand?A.extent(0):B.extent(0));
  int N = (expand?A.extent(1):B.extent(1));
  // MAM: Is it clear which is the "fast" index in ForEachInExtents?
  if( dim == 0 ) {
    if(expand) {
      auto f = [=] __device__(long n) {
        long i = n/N;
        long j = n - i*N;
        B_d(m_d(i),j) = scl_d*B_d(m_d(i),j) + alpha_d * s_d(i) * A_d(i,j);
      };
      cub::DeviceFor::Bulk(N*M,f);
    } else { 
      auto f = [=] __device__(long n) {
        long i = n/N;
        long j = n - i*N;
        B_d(i,j) = scl_d*B_d(i,j) + alpha_d * s_d(i) * A_d(m_d(i),j);
      };
      cub::DeviceFor::Bulk(N*M,f);
    }
  } else if( dim == 1 ) {
    if(expand) {
      auto f = [=] __device__(long n) {
        long i = n/N;
        long j = n - i*N;
        B_d(i,m_d(j)) = scl_d*B_d(i,m_d(j)) + alpha_d * s_d(j) * A_d(i,j);
      };
      cub::DeviceFor::Bulk(N*M,f);
    } else {
      auto f = [=] __device__(long n) {
        long i = n/N;
        long j = n - i*N;
        B_d(i,j) = scl_d*B_d(i,j) + alpha_d * s_d(j) * A_d(i,m_d(j));
      };
      cub::DeviceFor::Bulk(N*M,f);
    }
  }
}

template<typename V1, typename V3, typename V4, typename T>
void copy_select_impl(bool expand, V1 const& m, T alpha, V3 const& A, T scl, V4&& B)
{
  long N = m.extent(0);
  auto m_d = to_cuda_std_mdspan(m);
  auto A_d = to_cuda_std_mdspan(A);
  auto B_d = to_cuda_std_mdspan(B);
  auto scl_d = cuda_std_value_cast(scl); 
  auto alpha_d = cuda_std_value_cast(alpha);
  if(expand) {
    auto f = [=] __device__(long i) {  
      B_d(m_d(i)) = scl_d*B_d(m_d(i)) + alpha_d * A_d(i);
    };
    cub::DeviceFor::Bulk(N,f);
  } else {
    auto f = [=] __device__(long i) {   
      B_d(i) = scl_d*B_d(i) + alpha_d * A_d(m_d(i));
    };
    cub::DeviceFor::Bulk(N,f);
  }
}

template<typename V1, typename V2, typename V3, typename V4, typename T>
void copy_select_impl(bool expand, V1 const& m, V2 const& s, T alpha, V3 const& A, T scl, V4&& B)
{
  long N = m.extent(0);
  auto m_d = to_cuda_std_mdspan(m);
  auto s_d = to_cuda_std_mdspan(s);
  auto A_d = to_cuda_std_mdspan(A);
  auto B_d = to_cuda_std_mdspan(B);
  auto scl_d = cuda_std_value_cast(scl);
  auto alpha_d = cuda_std_value_cast(alpha);
  if(expand) {
    auto f = [=] __device__(long i) {
      B_d(m_d(i)) = scl_d*B_d(m_d(i)) + alpha_d * s_d(i) * A_d(i);
    };
    cub::DeviceFor::Bulk(N,f);
  } else {
    auto f = [=] __device__(long i) {
      B_d(i) = scl_d*B_d(i) + alpha_d * s_d(i) * A_d(m_d(i));
    };
    cub::DeviceFor::Bulk(N,f);
  }
}

//MAM: can I convert array_views to some type of common base?

using memory::device_array_view;
using std::complex;

template<int Rank>
using basic_layout_t = typename nda::basic_layout<0, nda::C_stride_order<Rank>, nda::layout_prop_e::none>;

#define _inst_(T,V,I) \
template void copy_select_impl(bool,V<const I,1,basic_layout_t<1>> const&, \
                          T,V<const T,1,basic_layout_t<1>> const&, \
                          T,V<T,1,basic_layout_t<1>> &);  \
template void copy_select_impl(bool,V<const I,1,basic_layout_t<1>> const&, \
                          V<const T,1,basic_layout_t<1>> const&, \
                          T,V<const T,1,basic_layout_t<1>> const&, \
                          T,V<T,1,basic_layout_t<1>> &);  \
template void copy_select_impl(bool,int,V<const I,1,basic_layout_t<1>> const&, \
                          T,V<const T,2,basic_layout_t<2>> const&, \
                          T,V<T,2,basic_layout_t<2>> &);  \
template void copy_select_impl(bool,int,V<const I,1,basic_layout_t<1>> const&, \
                          V<const T,1,basic_layout_t<1>> const&, \
                          T,V<const T,2,basic_layout_t<2>> const&, \
                          T,V<T,2,basic_layout_t<2>> &);  \
template void copy_select_impl(bool,V<const I,1,basic_layout_t<1>> const&, \
                          complex<T>,V<const complex<T>,1,basic_layout_t<1>> const&, \
                          complex<T>,V<complex<T>,1,basic_layout_t<1>> &); \
template void copy_select_impl(bool,V<const I,1,basic_layout_t<1>> const&, \
                          V<const complex<T>,1,basic_layout_t<1>> const&, \
                          complex<T>,V<const complex<T>,1,basic_layout_t<1>> const&, \
                          complex<T>,V<complex<T>,1,basic_layout_t<1>> &); \
template void copy_select_impl(bool,int,V<const I,1,basic_layout_t<1>> const&, \
                          complex<T>,V<const complex<T>,2,basic_layout_t<2>> const&, \
                          complex<T>,V<complex<T>,2,basic_layout_t<2>> &); \
template void copy_select_impl(bool,int,V<const I,1,basic_layout_t<1>> const&, \
                          V<const complex<T>,1,basic_layout_t<1>> const&, \
                          complex<T>,V<const complex<T>,2,basic_layout_t<2>> const&, \
                          complex<T>,V<complex<T>,2,basic_layout_t<2>> &); 

_inst_(double,device_array_view,int)
_inst_(double,device_array_view,long)

} // namespace kernels::device::detail

