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

template<int R>
struct zero_imag_fn
{
  view<std::complex<double>, R> a;

  __device__ void operator()(auto... idx) const { a(idx...).imag(0.0); }
};

template<int R>
void zero_imag(view<std::complex<double>, R> a)
{
  for_each(a, zero_imag_fn<R>{a});
}

template void zero_imag<1>(view<std::complex<double>, 1>);
template void zero_imag<2>(view<std::complex<double>, 2>);
template void zero_imag<3>(view<std::complex<double>, 3>);

} // namespace kernels::device
