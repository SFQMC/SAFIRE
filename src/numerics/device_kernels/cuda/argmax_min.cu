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

#include <limits>

#include <cub/device/device_reduce.cuh>
#include <cub/iterator/arg_index_input_iterator.cuh>

#include "arch/CUDA/cuda_init.h"
#include "numerics/device_kernels/device_api.hpp"

namespace kernels::device
{

namespace
{

__host__ __device__ inline double real_part(double x)
{
  return x;
}

__host__ __device__ inline double real_part(::cuda::std::complex<double> const& z)
{
  return z.real();
}

/**
 * @brief Extremum by real part, ties resolved to the smaller index.
 *
 * Symmetric and associative, so the result does not depend on the order cub folds in. That is what
 * makes it reproducible and keeps it agreeing with the thrust::max_element it replaces, which
 * returned the first extremum.
 */
template<typename V, bool Max>
struct pick_extremum
{
  using KV = cub::KeyValuePair<long, V>;

  __device__ KV operator()(KV const& a, KV const& b) const
  {
    double ra = real_part(a.value);
    double rb = real_part(b.value);
    if(ra != rb) {
      return (Max ? (rb > ra) : (rb < ra)) ? b : a;
    }
    return (b.key < a.key) ? b : a;
  }
};

template<typename T, bool Max>
std::tuple<long, T> arg_extremum(T const* x, long N)
{
  using V  = native_t<T>;
  using KV = cub::KeyValuePair<long, V>;

  cub::ArgIndexInputIterator<V const*, long> in(reinterpret_cast<V const*>(x));
  pick_extremum<V, Max>                      op{};

  // never wins on value, and loses every tie because its key is past the end
  KV init{N, V(Max ? std::numeric_limits<double>::lowest() : std::numeric_limits<double>::max())};

  KV*    d_out     = nullptr;
  void*  d_tmp     = nullptr;
  size_t tmp_bytes = 0;
  sfqmc::cuda::cuda_check(cudaMalloc(&d_out, sizeof(KV)), "arg_extremum: allocate result");
  sfqmc::cuda::cuda_check(cub::DeviceReduce::Reduce(nullptr, tmp_bytes, in, d_out, N, op, init),
                          "arg_extremum: temp storage query");
  sfqmc::cuda::cuda_check(cudaMalloc(&d_tmp, tmp_bytes), "arg_extremum: allocate temp storage");
  sfqmc::cuda::cuda_check(cub::DeviceReduce::Reduce(d_tmp, tmp_bytes, in, d_out, N, op, init),
                          "arg_extremum: reduce");

  KV h{};
  sfqmc::cuda::cuda_check(cudaMemcpy(&h, d_out, sizeof(KV), cudaMemcpyDeviceToHost),
                          "arg_extremum: copy result back");
  sfqmc::cuda::cuda_check(cudaFree(d_tmp), "arg_extremum: free temp storage");
  sfqmc::cuda::cuda_check(cudaFree(d_out), "arg_extremum: free result");

  return {h.key, from_native<T>(h.value)};
}

} // namespace

template<typename T>
std::tuple<long, T> argmax(T const* x, long N)
{
  return arg_extremum<T, true>(x, N);
}

template<typename T>
std::tuple<long, T> argmin(T const* x, long N)
{
  return arg_extremum<T, false>(x, N);
}

#define _inst_(T)                                                                                  \
  template std::tuple<long, T> argmax(T const*, long);                                             \
  template std::tuple<long, T> argmin(T const*, long);

_inst_(double)
_inst_(std::complex<double>)

} // namespace kernels::device
