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

// nda -> device kernel ABI. Host compiler only; a .cu must never include this, or nda comes back
// into nvcc.

#include <type_traits>

#include "nda/nda.hpp"
#include "numerics/device_kernels/device_view.hpp"

namespace kernels::device
{

/**
 * @brief Convert an nda array or view into the kernel ABI view.
 *
 * Reads data(), shape() and strides() directly. Those live in nda's
 * _impl_basic_array_view_common.hpp, which is textually included into both basic_array and
 * basic_array_view, so an owning array works without an A() to decay it first.
 *
 * The result is layout-erased, so the caller's layout (C, Fortran, strided, permuted) and its
 * memory space (device or unified) do not appear in the kernel's signature.
 */
template<nda::MemoryArray A>
auto to_view(A&& a)
{
  static_assert(nda::mem::on_device<A> or nda::mem::on_unified<A>,
                "to_view: only device or unified arrays.");
  static_assert(not(std::is_rvalue_reference_v<A&&> and nda::is_regular_v<std::decay_t<A>>),
                "to_view: an owning temporary would be destroyed before the kernel runs.");

  constexpr int R = nda::get_rank<A>;
  using elem_t    = std::remove_pointer_t<decltype(a.data())>;
  using dext      = ::cuda::std::dextents<long, R>;

  ::cuda::std::array<long, R> ext{}, str{};
  for(int d = 0; d < R; ++d) {
    ext[d] = a.shape()[d];
    str[d] = a.strides()[d];
  }

  return view<elem_t, R>(reinterpret_cast<native_t<elem_t>*>(a.data()),
                         ::cuda::std::layout_stride::mapping<dext>(ext, str));
}

} // namespace kernels::device
