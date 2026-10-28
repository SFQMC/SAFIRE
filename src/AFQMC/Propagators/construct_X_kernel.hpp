
#pragma once

#include <cassert>

#include "AFQMC/propagator_types.hpp"
#include "config.0.h"
#if defined(__CUDACC__)
#include <cuda/std/mdspan>
#include <cuda/std/complex>
#else
#include <complex>
#endif
#include "AFQMC/Utilities/probit.h"
#include "arch/atomics.hpp"

namespace sfqmc::afqmc::detail
{
  
template<typename V1, typename V2, typename V3, typename V4, typename V5>
struct construct_X_impl
{
  bool project_vbias;
  bool free_projection;
  double vbias_bound;
  V1 FieldTypes;
  V2 vMF;
  V3 HWs;
  V4 RNs;
  V5 X;

  // m is parallelized over as well, hence the atomic on HWs(iw) below
  __device__
  void operator()(long iw, long m)
  {
    namespace stdx =
#if defined(__CUDACC__)
    ::cuda::std;
#else
    std;
#endif
    using ComplexType = stdx::complex<RealType>;

    ComplexType im(0.0,1.0);
    auto vmf_0 = vMF(m);
    auto vmf_t = stdx::abs(vmf_0) > vbias_bound ? vmf_0 / stdx::abs(vmf_0) * vbias_bound : vmf_0;

    PropagatorTypes Fp = PropagatorTypes(FieldTypes(m));
    // X[iw,m] = rand[iw,m] + im * ( vbias[iw,m] - vMF[m]  )
    // HW[iw] = sum_m [ im * ( vMF[m] - vbias[iw,m] ) *
    //                     ( rand[iw,m] + halfim * ( vbias[iw,m] - vMF[m] ) ) ]
    //           = sum_m [ im * ( vMF[m] - vbias[iw,m] ) *
    //                     ( X[iw,m] - halfim * ( vbias[iw,m] - vMF[m] ) ) ]

    // X stores vbias at call site
    auto vb_t = stdx::abs(X(iw, m)) > vbias_bound ? X(iw, m) / stdx::abs(X(iw, m)) * vbias_bound : X(iw, m);

    if (project_vbias) {
      if (Fp == ContinuousSpinPropagator) {
        vb_t = ComplexType(0.0,stdx::imag(vb_t));
      } else if (Fp == ContinuousChargePropagator) {
        vb_t = ComplexType(stdx::real(vb_t),0.0);
      }
    }

    // Free projection samples the fields from the unshifted N(0,1), so there is nothing to
    // reweight: both branches below leave HWs at zero.
    ComplexType vdiff = free_projection ? 0.0 : im * (vb_t - vmf_t);
    if( Fp == ContinuousChargePropagator or
       Fp == ContinuousSpinPropagator) {
      X(iw,m) = probit(RNs(iw,m)) + vdiff;
      arch::atomic_add(&HWs(iw), - vdiff * (X(iw,m) - 0.5 * vdiff));
    } else if(Fp == DiscreteSpinPropagator or
           Fp == DiscreteChargePropagator) {
      auto wp = stdx::abs(stdx::exp(vdiff));
      auto wm = stdx::abs(stdx::exp(-vdiff));
      auto P = wm/(wp+wm);
      if( RNs(iw,m) < P ) {
        X(iw,m) = -1.0;
        arch::atomic_add(&HWs(iw), stdx::log(0.5/P));
      } else {
        X(iw,m) = 1.0;
        arch::atomic_add(&HWs(iw), stdx::log(0.5/(1.0-P)));
      }
    } else {
      assert(false);
    }
  }
};

} // sfqmc::afqmc::utils
