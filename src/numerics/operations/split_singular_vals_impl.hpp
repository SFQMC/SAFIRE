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

//FIX : add version for 1 D matrix, but ovlp is a vector of size nbatch

template<typename V1, typename V2, typename V3, typename V4, typename V5>
struct splitDmatrix_impl
{
  using T = typename std::decay_t<V4>::value_type; 
  V1 A;
  V2 B;
  V3 C;
  V4 res;
  V5 scl;

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
    for (long i = 0; i < A.extent(1); i++) {
        double ksi = log(A(b,i).real()) + scl(b).real();
          
      if(ksi > 0.0)
      {
        B(b,i) = T(1.0); // Dmin
        if(ksi >= 32*log(10.0)){
          C(b,i) = T(0.0); // Dmax^-1
        }
        else{
          C(b,i) = exp(-1.0*ksi);
        }
        res(b) += ksi;
      }
      else
      {
        if(ksi <= -32*log(10.0)){
          B(b,i) = T(0.0);
        }
        else{
          B(b,i) = exp(ksi);
        }
        C(b,i) = T(1.0);
      }

    }
  };
};

}

}
