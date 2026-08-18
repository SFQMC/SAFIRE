/*
 * This file is distributed under the Apache License, Version 2.0 License.
 * See LICENSE file in top directory for details.
 *
 * Copyright (c) 2021-2025 The Simons Foundation, Inc.
 *
 * You may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *      http://www.apache.org/licenses/LICENSE-2.0
 *
 */

#pragma once

#include <array>
#include <string_view>
#include <source_location>
#include <tuple>
#include <type_traits>
#include "IO/AppAbort.hpp"

namespace sfqmc::utils {

/**
 * Checks whether `a` has the shape given by `shape...`, and aborts otherwise
 * with the message "array <name> has unexpected shape <actual> != <expected>".
 *
 * Modeled on check: a struct template plus a deduction guide, so that the
 * trailing defaulted std::source_location follows the variadic extents and is
 * still captured at the call site.
 *
 * Accepts anything exposing its extents either through a `shape()` method (nda
 * arrays/views and sparse (csr) matrices, rank checked at compile time) or
 * through a `lengths` member
 * holding the per-dimension sizes (h5 dataset_info and friends, rank checked at
 * runtime). For the `lengths` case an optional trailing dimension of size 2 is
 * tolerated: a dataset of rank N+1 whose last extent is 2 (interleaved real/imag
 * storage of complex data) matches an N-extent shape.
 *
 * @tparam A      - nda array type or h5 dataset_info-like type
 * @tparam Longs  - integral extents
 * @param a       - object to check
 * @param name    - name of the array, used in the abort message
 * @param shape   - expected extents (one per dimension)
 */
template<class A, class... Longs>
struct check_shape
{
  check_shape(A const& a, std::string_view name, Longs... shape,
              const std::source_location& loc = std::source_location::current())
  {
    static_assert((std::is_integral_v<Longs> && ...),
                  "check_shape: extents must be integral");
    constexpr int N = sizeof...(Longs);
    std::array<long, N> expected{ static_cast<long>(shape)... };
    std::array<long, N> actual{};
    if constexpr (requires { a.shape(); }) {
      auto const sh = a.shape();
      static_assert(std::tuple_size_v<std::remove_cvref_t<decltype(sh)>> == N,
                    "check_shape: array rank must equal the number of extents");
      for(int d = 0; d < N; ++d) {
        actual[d] = static_cast<long>(sh[d]);
      }
    }
    else if constexpr (requires { a.lengths; }) {
      // Tolerate an optional trailing dimension of size 2, which encodes the
      // interleaved real/imag storage of complex data: a dataset of rank N+1
      // whose last extent is 2 is accepted as a match for an N-extent shape.
      long const rank = static_cast<long>(a.lengths.size());
      bool const has_interleaved_cplx_dim =
          (rank == N + 1) and (static_cast<long>(a.lengths[N]) == 2);
      if(rank != N and not has_interleaved_cplx_dim) {
        APP_ABORT_with_source(loc, "array {} has unexpected rank {} != {}",
                              name, a.lengths.size(), N);
      }
      for(int d = 0; d < N; ++d) {
        actual[d] = static_cast<long>(a.lengths[d]);
      }
    }
    else {
      static_assert(requires { a.shape(); } || requires { a.lengths; },
                    "check_shape: argument must expose a shape() method or a lengths member");
    }
    if(actual != expected) {
      APP_ABORT_with_source(loc, "array {} has unexpected shape {} != {}",
                            name, actual, expected);
    }
  }
};

template<class A, class... Longs>
check_shape(A const&, std::string_view, Longs...) -> check_shape<A, Longs...>;

} // namespace sfqmc::utils
