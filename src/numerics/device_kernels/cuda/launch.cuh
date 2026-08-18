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

// cub launch helpers. nvcc only -- cub instantiates and launches a __global__, so it cannot be
// compiled by the host compiler.

#include <cub/device/device_for.cuh>

#include "arch/CUDA/cuda_init.h"
#include "numerics/device_kernels/device_view.hpp"

namespace kernels::device
{

namespace detail
{

/// cub calls op(linear_index, coord_0, ..., coord_R-1); kernel bodies only want the coordinates.
///
/// f is mutable so that a functor with a non-const operator() -- what a body shared with a host
/// caller needs, since nda views propagate constness where mdspan does not -- still fits.
template<typename F>
struct drop_linear_index
{
  mutable F f;

  template<typename... I>
  __device__ void operator()(long, I... c) const
  {
    f(c...);
  }
};

} // namespace detail

/**
 * @brief Run f(i) for i in [0, n).
 */
template<typename F>
void bulk(long n, F f)
{
  if(n <= 0) {
    return;
  }
  sfqmc::cuda::cuda_check(cub::DeviceFor::Bulk(n, f));
}

/**
 * @brief Run f(i0, ..., iR-1) over every element of a, last index fastest by default.
 *
 * Layout selects the traversal order: layout_right visits the last index fastest, layout_left the
 * first. The default matches the memory order of the views these kernels take, so consecutive lanes
 * touch consecutive elements. Pass layout_left when a body accumulates into something indexed by an
 * outer coordinate -- coalescing is then worth less than spreading the atomics across lanes.
 *
 * ForEachInLayout rejects layout_stride, so the traversal mapping is built separately from the
 * view's extents; the coordinates it hands back then index the strided view. It is preferred over
 * ForEachInExtents because it is [[nodiscard]] and accepts layout_left.
 */
template<int R, typename Layout = ::cuda::std::layout_right, typename F>
void for_each_extents(::cuda::std::array<long, R> const& ext, F f)
{
  using dext = ::cuda::std::dextents<long, R>;

  long n = 1;
  for(int d = 0; d < R; ++d) {
    n *= ext[d];
  }
  if(n == 0) {
    return;
  }

  typename Layout::template mapping<dext> map{dext{ext}};
  sfqmc::cuda::cuda_check(cub::DeviceFor::ForEachInLayout(map, detail::drop_linear_index<F>{f}));
}

/**
 * @brief Run f(i0, ..., iR-1) over every element of a.
 */
template<typename Layout = ::cuda::std::layout_right, typename V, typename F>
void for_each(V const& a, F f)
{
  // V is a view<T,R>; deduce the rank from the mdspan rather than from view<>, whose value type is
  // an alias template and so a non-deduced context.
  constexpr int               R = static_cast<int>(V::rank());
  ::cuda::std::array<long, R> ext{};
  for(int d = 0; d < R; ++d) {
    ext[d] = a.extent(d);
  }
  for_each_extents<R, Layout>(ext, f);
}

} // namespace kernels::device
