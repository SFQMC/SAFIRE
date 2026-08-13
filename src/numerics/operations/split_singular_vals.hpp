////////////////////////////////////////////////////////////////////////////////
// This file is distributed under the Apache License, Version 2.0 License.
// See LICENSE file in top directory for details.
//
// Copyright (c) 2021-2025 The Simons Foundation, Inc.
//
// You may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//      http://www.apache.org/licenses/LICENSE-2.0
//
////////////////////////////////////////////////////////////////////////////////

#pragma once

#include "algorithm"
#include <complex>
#include "configuration.hpp"
#include "nda/nda.hpp"
#include "split_singular_vals_impl.hpp"

#if defined(ENABLE_DEVICE)
#include "numerics/device_kernels/device_api.hpp"
#include "numerics/device_kernels/to_view.hpp"
#endif

namespace math
{

#if defined(ENABLE_DEVICE)
using kernels::device::to_view;
#endif

template<nda::MemoryArrayOfRank<2> A_t, nda::MemoryArrayOfRank<2> B_t,
                     nda::MemoryArrayOfRank<2> C_t, nda::MemoryArrayOfRank<1> O_t,
                     nda::MemoryArrayOfRank<1> T_t>
requires( nda::mem::have_compatible_addr_space<A_t,B_t,C_t,O_t,T_t> and
          nda::have_same_value_type_v<A_t, B_t, C_t, O_t, T_t> and
          std::decay_t<A_t>::is_stride_order_C() and std::decay_t<B_t>::is_stride_order_C() and
          std::decay_t<C_t>::is_stride_order_C() 
        )
void splitDmatrix(A_t const& A, B_t&& B, C_t&& C, O_t&& logdet, T_t const& scl0)
{
  sfqmc::utils::check(A.shape() == B.shape(), "Size mismatch");
  sfqmc::utils::check(A.shape() == C.shape(), "Size mismatch");
  sfqmc::utils::check(A.extent(0) == logdet.extent(0), "Size mismatch");
  sfqmc::utils::check(A.extent(0) == scl0.extent(0), "Size mismatch");
#if defined(ENABLE_DEVICE)
    if constexpr (nda::mem::have_device_compatible_addr_space<A_t,B_t,C_t,O_t,T_t>){
        kernels::device::splitDmatrix(to_view(A),to_view(B),to_view(C),to_view(logdet),to_view(scl0));
    } 
    else
#endif
    {
        auto F = detail::splitDmatrix_impl<A_t,B_t,C_t,O_t,T_t>{A,B,C,logdet,scl0};
        std::ranges::for_each(nda::range(A.extent(0)),F);
    }
}

template<nda::MemoryArrayOfRank<1> A_t, nda::MemoryArrayOfRank<1> B_t,
                     nda::MemoryArrayOfRank<1> C_t, nda::MemoryArrayOfRank<1> O_t,
                     typename T_t>
requires( nda::mem::have_compatible_addr_space<A_t,B_t,C_t,O_t> and
          nda::have_same_value_type_v<A_t, B_t, C_t, O_t> and
          std::decay_t<A_t>::is_stride_order_C() and std::decay_t<B_t>::is_stride_order_C() and
          std::decay_t<C_t>::is_stride_order_C() 
        )
void splitDmatrix(A_t const& A, B_t&& B, C_t&& C, O_t&& logdet, T_t const& scl0)
{
  sfqmc::utils::check(A.shape() == B.shape(), "Size mismatch");
  sfqmc::utils::check(A.shape() == C.shape(), "Size mismatch");
  //sfqmc::utils::check(A.extent(0) == logdet.extent(0), "Size mismatch");
  //sfqmc::utils::check(A.extent(0) == scl0.extent(0), "Size mismatch");
  auto A2D = nda::reshape(A,std::array<long,2>{1,A.extent(0)});
  auto B2D = nda::reshape(B,std::array<long,2>{1,B.extent(0)});
  auto C2D = nda::reshape(C,std::array<long,2>{1,C.extent(0)});
#if defined(ENABLE_DEVICE)
    if constexpr (nda::mem::have_device_compatible_addr_space<A_t,B_t,C_t,O_t,T_t>){
        kernels::device::splitDmatrix(to_view(A2D),to_view(B2D),to_view(C2D),to_view(logdet),to_view(scl0));
    } 
    else
#endif
    {
        auto F = detail::splitDmatrix_impl<decltype(A2D),decltype(B2D),decltype(C2D),O_t,T_t>{A2D,B2D,C2D,logdet,scl0};
        std::ranges::for_each(nda::range(1),F);
    }
}

}
