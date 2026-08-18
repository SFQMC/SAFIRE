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

#include "numerics/device_kernels/cuda/launch.cuh"
#include "numerics/device_kernels/device_api.hpp"

namespace kernels::device
{

template<typename T, typename I, bool Expand, bool HasS>
struct select_1d_fn
{
  native_t<T>      alpha;
  native_t<T>      scl;
  view<I const, 1> m;
  view<T const, 1> s;
  view<T const, 1> A;
  view<T, 1>       B;

  __device__ void operator()(long i) const
  {
    native_t<T> a = alpha;
    if constexpr(HasS) {
      a *= s(i);
    }
    if constexpr(Expand) {
      B(m(i)) = scl * B(m(i)) + a * A(i);
    } else {
      B(i) = scl * B(i) + a * A(m(i));
    }
  }
};

template<typename T, typename I, int Dim, bool Expand, bool HasS>
struct select_2d_fn
{
  native_t<T>      alpha;
  native_t<T>      scl;
  view<I const, 1> m;
  view<T const, 1> s;
  view<T const, 2> A;
  view<T, 2>       B;

  __device__ void operator()(long i, long j) const
  {
    native_t<T> a = alpha;
    if constexpr(HasS) {
      a *= (Dim == 0 ? s(i) : s(j));
    }
    if constexpr(Dim == 0) {
      if constexpr(Expand) {
        B(m(i), j) = scl * B(m(i), j) + a * A(i, j);
      } else {
        B(i, j) = scl * B(i, j) + a * A(m(i), j);
      }
    } else {
      if constexpr(Expand) {
        B(i, m(j)) = scl * B(i, m(j)) + a * A(i, j);
      } else {
        B(i, j) = scl * B(i, j) + a * A(i, m(j));
      }
    }
  }
};

template<typename T, typename I, bool HasS>
static void select_1d(bool expand, view<I const, 1> m, view<T const, 1> s, T alpha,
                      view<T const, 1> A, T scl, view<T, 1> B)
{
  auto al = to_native(alpha);
  auto sc = to_native(scl);
  if(expand) {
    bulk(m.extent(0), select_1d_fn<T, I, true, HasS>{al, sc, m, s, A, B});
  } else {
    bulk(m.extent(0), select_1d_fn<T, I, false, HasS>{al, sc, m, s, A, B});
  }
}

template<typename T, typename I, bool HasS>
static void select_2d(bool expand, int dim, view<I const, 1> m, view<T const, 1> s, T alpha,
                      view<T const, 2> A, T scl, view<T, 2> B)
{
  auto al = to_native(alpha);
  auto sc = to_native(scl);
  // the traversal runs over whichever of the two the map indexes into
  ::cuda::std::array<long, 2> ext{expand ? A.extent(0) : B.extent(0),
                                  expand ? A.extent(1) : B.extent(1)};
  if(dim == 0) {
    if(expand) {
      for_each_extents<2>(ext, select_2d_fn<T, I, 0, true, HasS>{al, sc, m, s, A, B});
    } else {
      for_each_extents<2>(ext, select_2d_fn<T, I, 0, false, HasS>{al, sc, m, s, A, B});
    }
  } else if(dim == 1) {
    if(expand) {
      for_each_extents<2>(ext, select_2d_fn<T, I, 1, true, HasS>{al, sc, m, s, A, B});
    } else {
      for_each_extents<2>(ext, select_2d_fn<T, I, 1, false, HasS>{al, sc, m, s, A, B});
    }
  }
}

template<typename T, typename I>
void copy_select(bool expand, view<I const, 1> m, T alpha, view<T const, 1> A, T scl, view<T, 1> B)
{
  select_1d<T, I, false>(expand, m, {}, alpha, A, scl, B);
}

template<typename T, typename I>
void copy_select(bool expand, view<I const, 1> m, view<T const, 1> s, T alpha, view<T const, 1> A,
                 T scl, view<T, 1> B)
{
  select_1d<T, I, true>(expand, m, s, alpha, A, scl, B);
}

template<typename T, typename I>
void copy_select(bool expand, int dim, view<I const, 1> m, T alpha, view<T const, 2> A, T scl,
                 view<T, 2> B)
{
  select_2d<T, I, false>(expand, dim, m, {}, alpha, A, scl, B);
}

template<typename T, typename I>
void copy_select(bool expand, int dim, view<I const, 1> m, view<T const, 1> s, T alpha,
                 view<T const, 2> A, T scl, view<T, 2> B)
{
  select_2d<T, I, true>(expand, dim, m, s, alpha, A, scl, B);
}

#define _inst_(T, I)                                                                               \
  template void copy_select<T, I>(bool, view<I const, 1>, T, view<T const, 1>, T, view<T, 1>);      \
  template void copy_select<T, I>(bool, view<I const, 1>, view<T const, 1>, T, view<T const, 1>, T, \
                                  view<T, 1>);                                                     \
  template void copy_select<T, I>(bool, int, view<I const, 1>, T, view<T const, 2>, T,              \
                                  view<T, 2>);                                                     \
  template void copy_select<T, I>(bool, int, view<I const, 1>, view<T const, 1>, T,                 \
                                  view<T const, 2>, T, view<T, 2>);

#define _inst_types_(I)                                                                            \
  _inst_(double, I) _inst_(std::complex<double>, I)

_inst_types_(int)
_inst_types_(long)

} // namespace kernels::device
