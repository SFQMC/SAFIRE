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
#include "determinants_impl.hpp"

#if defined(ENABLE_DEVICE)
#include "numerics/device_kernels/device_api.hpp"
#include "numerics/device_kernels/to_view.hpp"
#endif

namespace math
{

#if defined(ENABLE_DEVICE)
using kernels::device::to_view;
#endif

template<nda::MemoryArrayOfRank<3> A, nda::MemoryMatrix IPIV, nda::MemoryVector V>
requires(std::decay_t<A>::is_stride_order_C() and std::decay_t<IPIV>::is_stride_order_C() and
         nda::mem::have_compatible_addr_space<A,IPIV,V>)
void log_determinant_from_getrf(A const& a, IPIV const& ipiv, V && log_det) {
  static_assert(nda::is_complex_v<nda::get_value_t<V>>, 
                "log_determinant_from_getrf expects complex numbers.");
  sfqmc::utils::check(a.extent(0) == ipiv.extent(0), "Size mismatch");
  sfqmc::utils::check(a.extent(0) == log_det.extent(0), "Size mismatch");
  sfqmc::utils::check(a.extent(1) == ipiv.extent(1), "Size mismatch");
  sfqmc::utils::check(a.extent(1) == a.extent(2), "Size mismatch");
#if defined(ENABLE_DEVICE)
  if constexpr (nda::mem::have_device_compatible_addr_space<A,IPIV,V>) {
    kernels::device::log_determinant_from_getrf(to_view(a),to_view(ipiv),to_view(log_det));
  } else 
#endif
  {
    auto F = detail::log_determinant_from_getrf_impl<A,IPIV,V>{a,ipiv,log_det};
    std::ranges::for_each(nda::range(a.extent(0)),F); 
  }
}

template<nda::MemoryArrayOfRank<2> A, nda::MemoryVector IPIV>
requires(std::decay_t<A>::is_stride_order_C() and std::decay_t<IPIV>::is_stride_order_C() and
         nda::mem::have_compatible_addr_space<A,IPIV>)
void log_determinant_from_getrf(A const& aM, IPIV const& ipiv_v, nda::get_value_t<A>& val) {
  using T = nda::get_value_t<A>;
  static_assert(nda::is_complex_v<T>, "log_determinant_from_getrf expects complex numbers.");
  sfqmc::utils::check(aM.extent(0) == ipiv_v.extent(0), "Size mismatch");
  sfqmc::utils::check(aM.extent(0) == aM.extent(1), "Size mismatch");
  sfqmc::utils::check(aM.is_contiguous(), "Expects contiguous array. Fix if needed!");
  constexpr MEMORY_SPACE MEM = memory::get_memory_space<A>();
  auto a = nda::reshape(aM,std::array<long,3>{1,aM.extent(0),aM.extent(1)});
  memory::array_view<MEM,const nda::get_value_t<IPIV>,2> ipiv(std::array<long,2>{1,ipiv_v.size()},ipiv_v.data());
  memory::array_view<MEM,T,1> log_det(std::array<long,1>{1},&val);
#if defined(ENABLE_DEVICE)
  if constexpr (nda::mem::have_device_compatible_addr_space<A,IPIV>) {
    kernels::device::log_determinant_from_getrf(to_view(a),to_view(ipiv),to_view(log_det));
  } else 
#endif
  {
    auto F = detail::log_determinant_from_getrf_impl<decltype(a),decltype(ipiv),decltype(log_det)>{a,ipiv,log_det};
    std::ranges::for_each(nda::range(1),F); 
  }
}

template<nda::MemoryArrayOfRank<3> A, nda::MemoryMatrix S, nda::MemoryVector V>
requires(std::decay_t<A>::is_stride_order_C() and std::decay_t<S>::is_stride_order_C() and
         nda::mem::have_compatible_addr_space<A,S,V>)
void log_determinant_from_geqrf(A const& a, S && scl, V && log_det) {
  using T = nda::get_value_t<V>;
  static_assert(nda::is_complex_v<T>, "log_determinant_from_geqrf expects complex numbers.");
  sfqmc::utils::check(a.extent(0) == log_det.extent(0), "Size mismatch");
  sfqmc::utils::check(a.extent(0) == scl.extent(0), "Size mismatch");
  sfqmc::utils::check(scl.extent(1) >= std::min(a.extent(1),a.extent(2)), "Size mismatch");
#if defined(ENABLE_DEVICE)
  if constexpr (nda::mem::have_device_compatible_addr_space<A,S,V>) {
    kernels::device::log_determinant_from_geqrf(to_view(a),to_view(scl),to_view(log_det));
  } else 
#endif
  {
    auto F = detail::log_determinant_from_geqrf_impl<A,S,V>{a,scl,log_det};
    std::ranges::for_each(nda::range(a.extent(0)),F); 
  }
}


}
