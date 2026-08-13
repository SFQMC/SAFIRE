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


#include <numbers>
#if defined(__CUDACC__)
#include <cuda/std/limits>
#include <cuda/std/complex>
#else
#include <complex>
#include <limits>
#endif
#include "arch/host_device.h"

namespace math
{

namespace detail {

template<typename V1, typename V2, typename V3>
struct log_determinant_from_getrf_impl
{
  using T = typename std::decay_t<V3>::value_type; 
  V1 a;
  V2 pivot;
  V3 res;
#if defined(__CUDACC__)
  T small = T(::cuda::std::numeric_limits<double>::min());
#else
  T small = T(std::numeric_limits<double>::min());
#endif
  static constexpr double pi = std::numbers::pi;

  __device__
  void operator()(long b)
  { 
#if defined(__CUDACC__)
    using ::cuda::std::complex;
#else
    using std::complex; 
    using std::imag;
    using std::log;
#endif
    for (long i = 0, ip = 1; i != a.extent(1); i++, ip++) {
      if(pivot(b,i) == ip)
        res(b) += log(static_cast<T>(a(b,i,i))+small);
      else
        res(b) += log(T(-1.0)*static_cast<T>(a(b,i,i))+small);
    }
    // bring imaginary part to [-pi,pi]
    if(imag(res(b)) > pi) {
      while(imag(res(b)) > pi) {
        auto v = imag(res(b));
        res(b).imag(v-2*pi);
      }
    } else if (imag(res(b)) < -pi) {
      while(imag(res(b)) < -pi) {
        auto v = imag(res(b));
        res(b).imag(v+2*pi);
      }
    }
  };
};

template<typename V1, typename V2, typename V3>
struct log_determinant_from_geqrf_impl
{
  using T = typename std::decay_t<V3>::value_type; 
  V1 a;
  V2 scl;
  V3 res;
#if defined(__CUDACC__)
  T small = T(::cuda::std::numeric_limits<double>::min());
#else
  T small = T(std::numeric_limits<double>::min());
#endif

  __device__
  void operator()(long b)
  { 
#if defined(__CUDACC__)
    using ::cuda::std::complex;
#else
    using std::complex;
    using std::log;
#endif
    long N = std::min(a.extent(1),a.extent(2));
    for (long i = 0; i != N; i++) {
      if(real(a(b,i,i)) < 0)
        scl(b,i) = T(-1.);
      else
        scl(b,i) = T(1.);;
      res(b) += log(static_cast<T>(scl(b,i) * a(b,i,i))+small);
    }
  };
};

}

}
