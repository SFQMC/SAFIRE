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

// The vocabulary of the nvcc <-> host-compiler boundary. Included by both, so it may include
// nothing but the standard library and CCCL -- no nda, no configuration.hpp.

#include <complex>
#include <type_traits>

#include <cuda/std/complex>
#include <cuda/std/mdspan>

namespace kernels::device
{

/**
 * @brief Device-side spelling of a host value type.
 *
 * std::complex<T> has no device operators, so it becomes cuda::std::complex<T>. The two have the
 * same size and layout and differ only in declared alignment (16 vs 8), which makes the pointer
 * cast in to_view() safe on the cudaMalloc'd memory it is restricted to.
 */
template<typename T>
struct native
{
  using type = T;
};

template<typename T>
struct native<std::complex<T>>
{
  using type = ::cuda::std::complex<T>;
};

template<typename T>
struct native<T const>
{
  using type = typename native<T>::type const;
};

template<typename T>
using native_t = typename native<T>::type;

/**
 * @brief Convert a scalar to its device spelling.
 *
 * Scalars cross the boundary as their host type -- the two compilers agree on std::complex<double>,
 * and it is what nda hands the caller -- so a kernel body converts on the way in.
 */
template<typename T>
native_t<T> to_native(T x)
{
  return x;
}

template<typename T>
::cuda::std::complex<T> to_native(std::complex<T> x)
{
  return {x.real(), x.imag()};
}

/**
 * @brief Convert a pointer to its device spelling. Layout is identical; only the declared
 *        alignment of complex differs, and this is only ever applied to cudaMalloc'd memory.
 */
template<typename T>
native_t<T>* to_native_ptr(T* p)
{
  return reinterpret_cast<native_t<T>*>(p);
}

/**
 * @brief Convert a scalar back from its device spelling. T is explicit -- native_t<T> is an alias
 *        template and so a non-deduced context.
 */
template<typename T>
T from_native(native_t<T> x)
{
  if constexpr(std::is_same_v<T, native_t<T>>) {
    return x;
  } else {
    return T(x.real(), x.imag());
  }
}

/**
 * @brief The ABI of every device kernel: a layout-erased view of device memory.
 *
 * Both compilers mangle this identically, so a host translation unit can declare and call a
 * function that an nvcc translation unit defines -- which is the whole point, since cub launches a
 * __global__ and therefore cannot leave nvcc.
 *
 * layout_stride carries extents and strides at runtime, so C, Fortran, strided and permuted nda
 * layouts all collapse into a single instantiation per (T, R). Device and unified memory collapse
 * too: both are just a pointer here.
 */
template<typename T, int R>
using view =
    ::cuda::std::mdspan<native_t<T>, ::cuda::std::dextents<long, R>, ::cuda::std::layout_stride>;

/**
 * @brief A scalar argument that is either a value or an address in device memory.
 *
 * A caller already holding the scalar on the device passes its address, so nothing is copied back
 * and nothing synchronizes. A caller holding it on the host passes the value, so nothing is
 * allocated to get it across. Telling the two apart costs one branch, uniform over the grid.
 */
template<typename T>
struct scalar_arg
{
  native_t<T>        value{};
  native_t<T> const* address = nullptr;
};

/// A scalar the caller holds by value.
template<typename T>
scalar_arg<T> scalar_value(T v)
{
  return {to_native(v), nullptr};
}

/// A scalar the caller holds in device memory.
template<typename T>
scalar_arg<T> scalar_at(T const* p)
{
  return {native_t<T>{}, to_native_ptr(p)};
}

} // namespace kernels::device
