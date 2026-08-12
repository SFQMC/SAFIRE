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

#include<complex>
#include "config.h"
#include "config.0.h"
#include "IO/AppAbort.hpp"
#include "nda/nda.hpp"
#include "utilities/allocators.hpp"

using RealType = double;
using SPRealType = float;
using ComplexType = std::complex<RealType>;
using SPComplexType = std::complex<RealType>;

enum MEMORY_SPACE { HOST_MEMORY, DEVICE_MEMORY, UNIFIED_MEMORY };

inline static constexpr nda::mem::AddressSpace to_nda_address_space(MEMORY_SPACE m)
{
  if(m == HOST_MEMORY)
    return nda::mem::Host; 
  else if(m == DEVICE_MEMORY)
    return nda::mem::Device;  
  else if(m == UNIFIED_MEMORY)
    return nda::mem::Unified;  
  return nda::mem::None; 
}

inline auto memory_space_to_string(MEMORY_SPACE m)
{
  if(m == HOST_MEMORY)
    return std::string("host");
  else if(m == DEVICE_MEMORY)
    return std::string("device");
  else if(m == UNIFIED_MEMORY)
    return std::string("unified");
  return std::string("unknown"); 
}

namespace memory 
{

template<typename A>
constexpr MEMORY_SPACE get_memory_space()
{
  return HOST_MEMORY;
}

template<nda::Array a_t>
constexpr MEMORY_SPACE get_memory_space()
{
  static_assert(nda::mem::on_host<a_t> or nda::mem::on_device<a_t> or nda::mem::on_unified<a_t>, "Unknown memory space");
  if constexpr (nda::mem::on_host<a_t>)
    return HOST_MEMORY;
  else if constexpr (nda::mem::on_device<a_t>)
    return DEVICE_MEMORY;
  else if constexpr (nda::mem::on_unified<a_t>)
    return UNIFIED_MEMORY;
  return HOST_MEMORY; 
}

template<MEMORY_SPACE MEM, typename... Args>
constexpr void check_memory_space(nda::Array auto && a, Args... rest)
{
  constexpr MEMORY_SPACE M = get_memory_space<std::decay_t<decltype(a)>>();
  static_assert(MEM == M, "Memory space mismatch");
  if constexpr (sizeof...(Args))
    check_memory_space<MEM>(rest...); 
} 

template<typename T, int N, typename Layout = nda::C_layout>
using host_array = nda::array<T,N,Layout>;
template<typename T, int N, typename Layout = nda::C_stride_layout>
using host_array_view = nda::array_view<T,N,Layout>;

#if defined(ENABLE_DEVICE)
template<typename T, int N, typename Layout = nda::C_layout>
using device_array = nda::cuarray<T,N,Layout>;
template<typename T, int N, typename Layout = nda::C_stride_layout>
using device_array_view = nda::cuarray_view<T,N,Layout>;
#else
template<typename T, int N, typename Layout = nda::C_layout>
using device_array = nda::array<T,N,Layout>;
template<typename T, int N, typename Layout = nda::C_stride_layout>
using device_array_view = nda::array_view<T,N,Layout>;
#endif

#if defined(ENABLE_DEVICE)
template<typename T, int N, typename Layout = nda::C_layout>
using unified_array = nda::basic_array<T, N, Layout, 'A', nda::heap<nda::mem::Unified>>;
template<typename T, int N, typename Layout = nda::C_stride_layout>
using unified_array_view = nda::basic_array_view<T, N, Layout, 'A', nda::default_accessor, nda::borrowed<nda::mem::Unified>>; 
#else
template<typename T, int N, typename Layout = nda::C_layout>
using unified_array = nda::array<T,N,Layout>;
template<typename T, int N, typename Layout = nda::C_stride_layout>
using unified_array_view = nda::array_view<T,N,Layout>;
#endif

template<MEMORY_SPACE MEM, typename T, int N, typename Layout = nda::C_layout>
using array = std::conditional_t<MEM==HOST_MEMORY, host_array<T,N,Layout>,
              std::conditional_t<MEM==DEVICE_MEMORY, device_array<T,N,Layout>,
              unified_array<T,N,Layout>>>;

template<MEMORY_SPACE MEM, typename T, int N, typename Layout = nda::C_stride_layout>
using array_view = std::conditional_t<MEM==HOST_MEMORY, host_array_view<T,N,Layout>,
                   std::conditional_t<MEM==DEVICE_MEMORY, device_array_view<T,N,Layout>,
                   unified_array_view<T,N,Layout>>>;

template<MEMORY_SPACE MEM>
decltype(auto) to_memory_space(auto &&A)
{
  if constexpr (MEM==HOST_MEMORY) {
    return nda::to_host(std::forward<decltype(A)>(A));
  } else if constexpr (MEM==DEVICE_MEMORY) {
    return nda::to_device(std::forward<decltype(A)>(A));
  } else {
    return nda::to_unified(std::forward<decltype(A)>(A));
  }
}

// Buffered Arrays

namespace detail
{

  template<MEMORY_SPACE MEM>
  using static_allocator_t = memory::static_fallback<memory::dynamic_bucket<to_nda_address_space(MEM)>>;

  template<MEMORY_SPACE MEM>
  using buffered_handle_t = nda::heap_basic<static_allocator_t<MEM>>;
  //using buffered_handle_t = nda::heap_basic<nda::mem::static_fallback<nda::mem::dynamic_bucket<to_nda_address_space(MEM)>>>;

}

  template<typename T, int N, typename Layout = nda::C_layout>
  using host_buffered_array = nda::array<T,N,Layout,detail::buffered_handle_t<HOST_MEMORY>>;

#if defined(ENABLE_DEVICE)

  template<typename T, int N, typename Layout = nda::C_layout>
  using device_buffered_array = nda::array<T,N,Layout,detail::buffered_handle_t<DEVICE_MEMORY>>;

  template<typename T, int N, typename Layout = nda::C_layout>
  using unified_buffered_array = nda::array<T,N,Layout,detail::buffered_handle_t<UNIFIED_MEMORY>>;

#else

  template<typename T, int N, typename Layout = nda::C_layout>
  using device_buffered_array = host_buffered_array<T,N,Layout>;

  template<typename T, int N, typename Layout = nda::C_layout>
  using unified_buffered_array = host_buffered_array<T,N,Layout>;

#endif

  template<MEMORY_SPACE MEM, typename T, int N, typename Layout = nda::C_layout>
  using buffered_array = std::conditional_t<MEM==HOST_MEMORY,    host_buffered_array<T,N,Layout>,
                         std::conditional_t<MEM==DEVICE_MEMORY,  device_buffered_array<T,N,Layout>,
                         unified_buffered_array<T,N,Layout>>>;

// routine to return an array_view to the provided array, but converted to 
// real type (with remove_complex_t<T>) with an extra dimension. Only works
// for C or Fortran arrays with min_stride==1.  
// should this be somewhere???
  template<nda::MemoryArray A_t>
  auto to_real_view(A_t && a) {
    using A = std::decay_t<A_t>;
    // don't use nda::get_value_t, which removes qualifiers 
    using value_type = typename A::value_type;  
    using real_t = nda::remove_complex_t<value_type>;
    constexpr int rank = nda::get_rank<A>;
    constexpr MEMORY_SPACE MEM = get_memory_space<A>();

    if constexpr (nda::is_complex_v<value_type>) {
      static_assert(A::is_stride_order_C() or A::is_stride_order_Fortran(), "Stride order mismatch");
      // A zero extent zeroes out the strides of the enclosing dimensions, so an
      // empty array carries no usable stride information: min_stride() is not
      // meaningful and the doubled strides no longer respect the stride order.
      // Nothing is addressable either way, so build the view from the shape alone.
      bool const empty = (a.size() == 0);
      if(not empty and a.indexmap().min_stride() != 1) {
        std::source_location loc = std::source_location::current();
        sfqmc::APP_ABORT_with_source(loc, "Strides mismatch");
      }
      if constexpr (A::is_stride_order_C()) {
        using idx_map_t = nda::idx_map<rank+1, 0, nda::C_stride_order<rank+1>, nda::layout_prop_e::none>;
        std::array<long,rank+1> shape;
        std::copy_n(a.shape().begin(),rank,shape.begin());
        shape[rank] = 2;
        std::array<long,rank+1> str;
        std::transform(a.strides().begin(),a.strides().end(),str.begin(),
            [](auto const& x) {return 2*x;} );
        str[rank] = 1;
        idx_map_t idxm = (empty ? idx_map_t(shape) : idx_map_t(shape,str));
        if constexpr (std::is_const_v<std::remove_pointer_t<decltype(a.data())>>)
          return memory::array_view<MEM,const real_t,rank+1>(idxm, reinterpret_cast<real_t const*>(a.data()));
        else
          return memory::array_view<MEM,real_t,rank+1>(idxm, reinterpret_cast<real_t*>(a.data()));
      } else {
        using idx_map_t = nda::idx_map<rank+1, 0, nda::Fortran_stride_order<rank+1>, nda::layout_prop_e::none>;
        std::array<long,rank+1> shape;
        std::copy_n(a.shape().begin(),rank,shape.begin()+1);
        shape[0] = 2;
        std::array<long,rank+1> str;
        std::transform(a.strides().begin(),a.strides().end(),str.begin()+1,
            [](auto const& x) {return 2*x;} );
        str[0] = 1;
        idx_map_t idxm = (empty ? idx_map_t(shape) : idx_map_t(shape,str));
        if constexpr (std::is_const_v<std::remove_pointer_t<decltype(a.data())>>)
          return memory::array_view<MEM,const real_t,rank+1,nda::F_stride_layout>(idxm, reinterpret_cast<real_t const*>(a.data()));
        else
          return memory::array_view<MEM,real_t,rank+1,nda::F_stride_layout>(idxm, reinterpret_cast<real_t*>(a.data()));
      }
    } else {
      return a();
    }
  }

  template<nda::MemoryArray A_t>
  requires(nda::get_rank<A_t> > 1 and (std::decay_t<A_t>::is_stride_order_C() or std::decay_t<A_t>::is_stride_order_Fortran()) )
  auto diagonal_view(A_t && a) {
    using A = std::decay_t<A_t>;
    // don't use nda::get_value_t, which removes qualifiers
    using value_type = typename A::value_type;
    constexpr int rank = nda::get_rank<A>;
    constexpr MEMORY_SPACE MEM = get_memory_space<A>();
    std::array<long,rank-1> str; 
    std::array<long,rank-1> shape; 
    if constexpr (std::decay_t<A_t>::is_stride_order_C()) {
      if(a.extent(rank-1) != a.extent(rank-2) ) {
        std::source_location loc = std::source_location::current();
        sfqmc::APP_ABORT_with_source(loc, "Shape mismatch");
      }
      std::copy_n(a.strides().begin(),rank-1,str.begin());
      std::copy_n(a.shape().begin(),rank-1,shape.begin());
      str[rank-2]++;  
      nda::idx_map<rank-1, 0, nda::C_stride_order<rank-1>, nda::layout_prop_e::none> idxm(shape,str);
      return memory::array_view<MEM,value_type,rank-1>(idxm, a.data());
    } else if constexpr (std::decay_t<A_t>::is_stride_order_Fortran()) {
      if(a.extent(0) != a.extent(1) ) {
        std::source_location loc = std::source_location::current();
        sfqmc::APP_ABORT_with_source(loc, "Shape mismatch");
      }
      if constexpr (rank>2) std::copy_n(a.strides().begin()+2,rank-2,str.begin()+1);
      str[0] = a.strides()[0]+1;  
      std::copy_n(a.shape().begin()+1,rank-1,shape.begin());
      nda::idx_map<rank-1, 0, nda::Fortran_stride_order<rank-1>, nda::layout_prop_e::none> idxm(shape,str);
      return memory::array_view<MEM,value_type,rank-1,nda::F_stride_layout>(idxm, a.data());
    }
  }

}

