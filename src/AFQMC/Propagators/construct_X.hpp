
#pragma once

#include "AFQMC/config.h"
#if defined(__CUDACC__)
#include <cuda/std/mdspan>
#include <cuda/std/complex>
#else
#include <complex>
#include "utilities/check.hpp"
#include "nda/nda.hpp"
#endif
#include "AFQMC/Utilities/probit.h"
#include "arch/atomics.hpp"

namespace sfqmc::afqmc::detail
{
  
template<typename V1, typename V2, typename V3, typename V4, typename V5>
struct construct_X_impl
{
  bool zero;
  bool free_projection;
  double sqrtdt;
  double vbias_bound;
  V1 FieldTypes;
  V2 vMF;
  V3 MF;
  V3 HWs;
  V4 RNs;
  V5 X;

  __device__
  void operator()(long nn)
  {
    namespace stdx =
#if defined(__CUDACC__)
    ::cuda::std;
#else
    std;
#endif
    using ComplexType = stdx::complex<RealType>;

    ComplexType im(0.0,1.0);
    long iw = nn % X.extent(0);
    int m = nn / X.extent(0);
    // parallelizing over m would lead to race condition below
    auto vmf_0 = vMF(m);
    auto vmf_t = stdx::abs(vmf_0) > vbias_bound * sqrtdt ? vmf_0 / stdx::abs(vmf_0) * vbias_bound * sqrtdt : vmf_0;

    PropagatorTypes Fp = PropagatorTypes(FieldTypes(m));
    // X[iw,m] = rand[iw,m] + im * ( vbias[iw,m] - vMF[m]  )
    // HW[iw] = sum_m [ im * ( vMF[m] - vbias[iw,m] ) *
    //                     ( rand[iw,m] + halfim * ( vbias[iw,m] - vMF[m] ) ) ]
    //           = sum_m [ im * ( vMF[m] - vbias[iw,m] ) *
    //                     ( X[iw,m] - halfim * ( vbias[iw,m] - vMF[m] ) ) ]
    // MF[iw] = sum_m ( im * X[iw,m] * vMF[m] )
    auto vb_t = stdx::abs(X(iw, m)) > vbias_bound * sqrtdt ? X(iw, m) / stdx::abs(X(iw, m)) * vbias_bound * sqrtdt : X(iw, m);

    if (zero) {
      if (Fp == ContinuousSpinPropagator) {
        vb_t = ComplexType(0.0,stdx::imag(vb_t));
      } else if (Fp == ContinuousChargePropagator) {
        vb_t = ComplexType(stdx::real(vb_t),0.0);
      }
    }

    if( Fp == ContinuousChargePropagator or
       Fp == ContinuousSpinPropagator) {
      ComplexType vdiff =
          free_projection ? 0 : (im * (vb_t - vmf_t));
      X(iw,m) = probit(RNs(iw,m)) + vdiff;
      arch::atomic_add(&HWs(iw), - vdiff * (X(iw,m) - 0.5 * vdiff));
      arch::atomic_add(&MF(iw), im * X(iw,m) * vmf_0);
    } else if( Fp == DiscreteSpinPropagator or
           Fp == DiscreteChargePropagator ) {
      ComplexType vdiff =
          free_projection ? 0.0 : (im * (vb_t - vmf_t));
      auto wp = stdx::abs(stdx::exp(vdiff));
      auto wm = stdx::abs(stdx::exp(-vdiff));
      auto P = wm/(wp+wm);
      if( RNs(iw,m) < P ) {
        X(iw,m) = -1.0;
        //arch::atomic_add(&HWs(iw), stdx::log(1.0/P));
        // discrete HS transformation has factor (1/2)^NMO
        arch::atomic_add(&HWs(iw), stdx::log(0.5/P));
      } else {
        X(iw,m) = 1.0;
        //arch::atomic_add(&HWs(iw), stdx::log(1.0/(1.0-P)));
        // discrete HS transformation has factor (1/2)^NMO
        arch::atomic_add(&HWs(iw), stdx::log(0.5/(1.0-P)));
      }
      // W_MSsub = exp(-MF), careful with sign convention
      arch::atomic_add(&MF(iw), im * X(iw,m) * vmf_0);
    } else {
      assert(false);
    }
  }
};

} // sfqmc::afqmc::utils
