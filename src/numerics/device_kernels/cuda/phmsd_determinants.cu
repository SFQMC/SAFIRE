#include <complex>

#include "configuration.hpp"
#include "utilities/check.hpp"
#include "numerics/device_kernels/cuda/cuda_settings.h"
#include "numerics/device_kernels/cuda/cuda_aux.hpp"
#include "arch/atomics.hpp"
#include <nda/nda.hpp>
#include "numerics/nda_functions.hpp"
#include <cuda/std/mdspan>
#include "cub/device/device_for.cuh"
#include "numerics/operations/determinants.hpp"

namespace kernels::device::detail
{
template<typename T>
__device__ T _D3x3_( T const a11, T const a12, T const a13, 
                   T const a21, T const a22, T const a23, 
                   T const a31, T const a32, T const a33)
{
  return   a11 * (a22 * a33 - a32 * a23)
         - a21 * (a12 * a33 - a32 * a13)
         + a31 * (a12 * a23 - a22 * a13);
}

template<typename T>
__device__ void I3x3(T const a11, T const a12, T const a13,
                     T const a21, T const a22, T const a23,
                     T const a31, T const a32, T const a33, T const scl, T* M)
{
  M[0] = (a22 * a33 - a32 * a23) * scl;
  M[1] = (a13 * a32 - a12 * a33) * scl;
  M[2] = (a12 * a23 - a13 * a22) * scl;
  M[3] = (a23 * a31 - a21 * a33) * scl;
  M[4] = (a11 * a33 - a13 * a31) * scl;
  M[5] = (a13 * a21 - a11 * a23) * scl;
  M[6] = (a21 * a32 - a22 * a31) * scl;
  M[7] = (a12 * a31 - a11 * a32) * scl;
  M[8] = (a11 * a22 - a12 * a21) * scl;
}

//#ifdef NDEBUG
template<class T>
__device__ T D4x4( T const a11, T const a12, T const a13, T const a14, 
                   T const a21, T const a22, T const a23, T const a24, 
                   T const a31, T const a32, T const a33, T const a34, 
                   T const a41, T const a42, T const a43, T const a44 ) 
{
  return (a11 * (a22 * (a33 * a44 - a43 * a34) - a32 * (a23 * a44 - a43 * a24) + a42 * (a23 * a34 - a33 * a24)) -
          a21 * (a12 * (a33 * a44 - a43 * a34) - a32 * (a13 * a44 - a43 * a14) + a42 * (a13 * a34 - a33 * a14)) +
          a31 * (a12 * (a23 * a44 - a43 * a24) - a22 * (a13 * a44 - a43 * a14) + a42 * (a13 * a24 - a23 * a14)) -
          a41 * (a12 * (a23 * a34 - a33 * a24) - a22 * (a13 * a34 - a33 * a14) + a32 * (a13 * a24 - a23 * a14)));
}

template<class T>
__device__ void I4x4(T const a11, T const a12, T const a13, T const a14,
                     T const a21, T const a22, T const a23, T const a24,
                     T const a31, T const a32, T const a33, T const a34,
                     T const a41, T const a42, T const a43, T const a44, T const scl, T* M)
{
  M[0] = scl * _D3x3_(
    a22, a23, a24,
    a32, a33, a34,
    a42, a43, a44
    );
  M[4] = -scl * _D3x3_(
    a21, a23, a24,
    a31, a33, a34,
    a41, a43, a44
    );
  M[8] = scl * _D3x3_(
    a21, a22, a24,
    a31, a32, a34,
    a41, a42, a44
    );
  M[12] = -scl * _D3x3_(
    a21, a22, a23,
    a31, a32, a33,
    a41, a42, a43
    );

  M[1] = -scl * _D3x3_(
    a12, a13, a14,
    a32, a33, a34,
    a42, a43, a44
    );
  M[5] = scl * _D3x3_(
    a11, a13, a14,
    a31, a33, a34,
    a41, a43, a44
    );
  M[9] = -scl * _D3x3_(
    a11, a12, a14,
    a31, a32, a34,
    a41, a42, a44
    );
  M[13] = scl * _D3x3_(
    a11, a12, a13,
    a31, a32, a33,
    a41, a42, a43
    );

  M[2] = scl * _D3x3_(
    a12, a13, a14,
    a22, a23, a24,
    a42, a43, a44
    );
  M[6] = -scl * _D3x3_(
    a11, a13, a14,
    a21, a23, a24,
    a41, a43, a44
    );
  M[10] = scl * _D3x3_(
    a11, a12, a14,
    a21, a22, a24,
    a41, a42, a44
    );
  M[14] = -scl * _D3x3_(
    a11, a12, a13,
    a21, a22, a23,
    a41, a42, a43
    );

  M[3] = -scl * _D3x3_(
    a12, a13, a14,
    a22, a23, a24,
    a32, a33, a34
    );
  M[7] = scl * _D3x3_(
    a11, a13, a14,
    a21, a23, a24,
    a31, a33, a34
    );
  M[11] = -scl * _D3x3_(
    a11, a12, a14,
    a21, a22, a24,
    a31, a32, a34
    );
  M[15] = scl * _D3x3_(
    a11, a12, a13,
    a21, a22, a23,
    a31, a32, a33
    );
}

template<class T>
__device__  T D5x5( T const a11, T const a12, T const a13, T const a14, T const a15, 
                    T const a21, T const a22, T const a23, T const a24, T const a25, 
                    T const a31, T const a32, T const a33, T const a34, T const a35, 
                    T const a41, T const a42, T const a43, T const a44, T const a45, 
                    T const a51, T const a52, T const a53, T const a54, T const a55 ) 
{
  return (a11 *
              (a22 * (a33 * (a44 * a55 - a54 * a45) - a43 * (a34 * a55 - a54 * a35) + a53 * (a34 * a45 - a44 * a35)) -
               a32 * (a23 * (a44 * a55 - a54 * a45) - a43 * (a24 * a55 - a54 * a25) + a53 * (a24 * a45 - a44 * a25)) +
               a42 * (a23 * (a34 * a55 - a54 * a35) - a33 * (a24 * a55 - a54 * a25) + a53 * (a24 * a35 - a34 * a25)) -
               a52 * (a23 * (a34 * a45 - a44 * a35) - a33 * (a24 * a45 - a44 * a25) + a43 * (a24 * a35 - a34 * a25))) -
          a21 *
              (a12 * (a33 * (a44 * a55 - a54 * a45) - a43 * (a34 * a55 - a54 * a35) + a53 * (a34 * a45 - a44 * a35)) -
               a32 * (a13 * (a44 * a55 - a54 * a45) - a43 * (a14 * a55 - a54 * a15) + a53 * (a14 * a45 - a44 * a15)) +
               a42 * (a13 * (a34 * a55 - a54 * a35) - a33 * (a14 * a55 - a54 * a15) + a53 * (a14 * a35 - a34 * a15)) -
               a52 * (a13 * (a34 * a45 - a44 * a35) - a33 * (a14 * a45 - a44 * a15) + a43 * (a14 * a35 - a34 * a15))) +
          a31 *
              (a12 * (a23 * (a44 * a55 - a54 * a45) - a43 * (a24 * a55 - a54 * a25) + a53 * (a24 * a45 - a44 * a25)) -
               a22 * (a13 * (a44 * a55 - a54 * a45) - a43 * (a14 * a55 - a54 * a15) + a53 * (a14 * a45 - a44 * a15)) +
               a42 * (a13 * (a24 * a55 - a54 * a25) - a23 * (a14 * a55 - a54 * a15) + a53 * (a14 * a25 - a24 * a15)) -
               a52 * (a13 * (a24 * a45 - a44 * a25) - a23 * (a14 * a45 - a44 * a15) + a43 * (a14 * a25 - a24 * a15))) -
          a41 *
              (a12 * (a23 * (a34 * a55 - a54 * a35) - a33 * (a24 * a55 - a54 * a25) + a53 * (a24 * a35 - a34 * a25)) -
               a22 * (a13 * (a34 * a55 - a54 * a35) - a33 * (a14 * a55 - a54 * a15) + a53 * (a14 * a35 - a34 * a15)) +
               a32 * (a13 * (a24 * a55 - a54 * a25) - a23 * (a14 * a55 - a54 * a15) + a53 * (a14 * a25 - a24 * a15)) -
               a52 * (a13 * (a24 * a35 - a34 * a25) - a23 * (a14 * a35 - a34 * a15) + a33 * (a14 * a25 - a24 * a15))) +
          a51 *
              (a12 * (a23 * (a34 * a45 - a44 * a35) - a33 * (a24 * a45 - a44 * a25) + a43 * (a24 * a35 - a34 * a25)) -
               a22 * (a13 * (a34 * a45 - a44 * a35) - a33 * (a14 * a45 - a44 * a15) + a43 * (a14 * a35 - a34 * a15)) +
               a32 * (a13 * (a24 * a45 - a44 * a25) - a23 * (a14 * a45 - a44 * a15) + a43 * (a14 * a25 - a24 * a15)) -
               a42 * (a13 * (a24 * a35 - a34 * a25) - a23 * (a14 * a35 - a34 * a15) + a33 * (a14 * a25 - a24 * a15))));
}
//#endif

template<typename T_t, typename Ov_t>
void phmsd_det_impl(int nex, int const* iex, T_t const& T, Ov_t& ov)
{
  auto T_d = to_cuda_std_mdspan(T);
  auto ov_d = to_cuda_std_mdspan(ov);
  long ndet = ov.extent(0);
  long nwalk = ov.extent(1);
  switch(nex) 
  {
    case 1:
    {
      auto f = [=] __device__(long n) {
        long iw   = n/ndet; 
        long idet = n - iw*ndet; 
        ov_d(idet,iw) = T_d(iw,iex[2*idet+1],iex[2*idet]);
      };
      cub::DeviceFor::Bulk(nwalk*ndet,f);
      break;   
    }
    case 2:
    {
      auto f = [=] __device__(long n) {
        long iw   = n/ndet;
        long idet = n - iw*ndet;
        int const* x = iex + 4*idet; 
        ov_d(idet,iw) = T_d(iw,x[2],x[0])*T_d(iw,x[3],x[1]) - T_d(iw,x[2],x[1])*T_d(iw,x[3],x[0]);
      };
      cub::DeviceFor::Bulk(nwalk*ndet,f);
      break;
    }
    case 3:
    {
      auto f = [=] __device__(long n) {
        long iw   = n/ndet;
        long idet = n - iw*ndet;
        int const* x = iex + 6*idet; 
        ov_d(idet,iw) = _D3x3_(T_d(iw,x[3],x[0]),T_d(iw,x[3],x[1]),T_d(iw,x[3],x[2]),
                             T_d(iw,x[4],x[0]),T_d(iw,x[4],x[1]),T_d(iw,x[4],x[2]),
                             T_d(iw,x[5],x[0]),T_d(iw,x[5],x[1]),T_d(iw,x[5],x[2]));
      };
      cub::DeviceFor::Bulk(nwalk*ndet,f);
      break;
    }
    case 4:
    {
      auto f = [=] __device__(long n) {
        long iw   = n/ndet;
        long idet = n - iw*ndet;
        int const* x = iex + 8*idet; 
        ov_d(idet,iw) = D4x4(T_d(iw,x[4],x[0]),T_d(iw,x[4],x[1]),T_d(iw,x[4],x[2]),T_d(iw,x[4],x[3]),
                             T_d(iw,x[5],x[0]),T_d(iw,x[5],x[1]),T_d(iw,x[5],x[2]),T_d(iw,x[5],x[3]),
                             T_d(iw,x[6],x[0]),T_d(iw,x[6],x[1]),T_d(iw,x[6],x[2]),T_d(iw,x[6],x[3]),         
                             T_d(iw,x[7],x[0]),T_d(iw,x[7],x[1]),T_d(iw,x[7],x[2]),T_d(iw,x[7],x[3])); 
      };
      cub::DeviceFor::Bulk(nwalk*ndet,f);
      break;
    }
    case 5:
    {
      auto f = [=] __device__(long n) {
        long iw   = n/ndet;
        long idet = n - iw*ndet;
        int const* x = iex + 10*idet;
        ov_d(idet,iw) = D5x5(T_d(iw,x[5],x[0]),T_d(iw,x[5],x[1]),T_d(iw,x[5],x[2]),T_d(iw,x[5],x[3]),T_d(iw,x[5],x[4]),
                             T_d(iw,x[6],x[0]),T_d(iw,x[6],x[1]),T_d(iw,x[6],x[2]),T_d(iw,x[6],x[3]),T_d(iw,x[6],x[4]),
                             T_d(iw,x[7],x[0]),T_d(iw,x[7],x[1]),T_d(iw,x[7],x[2]),T_d(iw,x[7],x[3]),T_d(iw,x[7],x[4]),
                             T_d(iw,x[8],x[0]),T_d(iw,x[8],x[1]),T_d(iw,x[8],x[2]),T_d(iw,x[8],x[3]),T_d(iw,x[8],x[4]),
                             T_d(iw,x[9],x[0]),T_d(iw,x[9],x[1]),T_d(iw,x[9],x[2]),T_d(iw,x[9],x[3]),T_d(iw,x[9],x[4]));
      };
      cub::DeviceFor::Bulk(nwalk*ndet,f);
      break;
    }
    default:
    {
      // M[iwalk][idet][ip][iq]
      memory::buffered_array<DEVICE_MEMORY,nda::get_value_t<T_t>,4> M(ndet,nwalk,nex,nex);
      auto M_d = to_cuda_std_mdspan(M);
      long n1 = nwalk*nex*nex;
      long n2 = nex*nex;
      auto extract = [=] __device__(long n) {
        long idet = n/n1;
        n -= idet*n1;
        long iw   = n/n2;
        n -= iw*n2;
        long i   = n/nex;
        long j   = n - i*nex; 
        M_d(idet,iw,i,j) = T_d(iw,iex[2*nex*idet + nex + i],iex[2*nex*idet + j]);    
      };
      cub::DeviceFor::Bulk(ndet*nwalk*nex*nex,extract);
      memory::buffered_array<DEVICE_MEMORY,int,2> ipiv(ndet*nwalk,nex);
      memory::buffered_array<DEVICE_MEMORY,nda::get_value_t<T_t>,1> work;
      auto M3d = nda::reshape(M,std::array<long,3>{ndet*nwalk,nex,nex});
      nda::lapack::getrf(M3d,ipiv,work);
      math::log_determinant_from_getrf(M3d,ipiv,nda::flatten(ov));
    }
  };
}

template<typename T_t, typename R_t> 
void phmsd_compact_R_impl(int nex, int const* refc, int const* iex, T_t const& T, R_t &R) 
{
  using nda::range;
  auto all = range::all;
  // T(nw,nact,nel)
  auto T_d = to_cuda_std_mdspan(T);
  // R(nwalk,ndet,nex,nact) 
  auto R_d = to_cuda_std_mdspan(R);
  using value_t = std::decay_t<decltype(T_d(0,0,0))>;
  long nwalk = R.extent(0);
  long ndet = R.extent(1);
  long nel = T.extent(2);
  long nact = R.extent(3);
  memory::buffered_array<DEVICE_MEMORY,ComplexType,2> ov(nwalk,ndet);
  ov() = ComplexType(0);
  auto ov_d = to_cuda_std_mdspan(ov);
  memory::buffered_array<DEVICE_MEMORY,ComplexType,4> M(nwalk,ndet,nex,nex);
  auto M_d = to_cuda_std_mdspan(M);
  // Compute inverses 
  switch(nex)
  {
    case 1:
    {
      auto f = [=] __device__(long n) {
        long iw   = n/ndet;
        long idet = n - iw*ndet;
        ov_d(iw,idet) = T_d(iw,iex[2*idet+1],iex[2*idet]);
        if(abs(ov_d(iw,idet)) != 0) 
          M_d(iw,idet,0,0) = value_t(1.0)/ov_d(iw,idet);
      };
      cub::DeviceFor::Bulk(nwalk*ndet,f);
      break;
    }
    case 2:
    {
      auto f = [=] __device__(long n) {
        long iw   = n/ndet;
        long idet = n - iw*ndet;
        int const* x = iex + 4*idet;
        ov_d(iw,idet) = T_d(iw,x[2],x[0])*T_d(iw,x[3],x[1]) - T_d(iw,x[2],x[1])*T_d(iw,x[3],x[0]);
        if(abs(ov_d(iw,idet)) != 0) { 
          value_t scl = value_t(1.0)/ov_d(iw,idet);
          M_d(iw,idet,0,0) = T_d(iw,x[3],x[1]) * scl; 
          M_d(iw,idet,0,1) = -T_d(iw,x[2],x[1]) * scl; 
          M_d(iw,idet,1,0) = -T_d(iw,x[3],x[0]) * scl; 
          M_d(iw,idet,1,1) = T_d(iw,x[2],x[0]) * scl; 
        }
      };
      cub::DeviceFor::Bulk(nwalk*ndet,f);
      break;
    }
    case 3:
    {
      auto f = [=] __device__(long n) {
        long iw   = n/ndet;
        long idet = n - iw*ndet;
        int const* x = iex + 6*idet;
        ov_d(iw,idet) = _D3x3_(T_d(iw,x[3],x[0]),T_d(iw,x[3],x[1]),T_d(iw,x[3],x[2]),
                             T_d(iw,x[4],x[0]),T_d(iw,x[4],x[1]),T_d(iw,x[4],x[2]),
                             T_d(iw,x[5],x[0]),T_d(iw,x[5],x[1]),T_d(iw,x[5],x[2]));
        if(abs(ov_d(iw,idet)) != 0) { 
          value_t scl = value_t(1.0)/ov_d(iw,idet);
          I3x3(T_d(iw,x[3],x[0]),T_d(iw,x[3],x[1]),T_d(iw,x[3],x[2]),
               T_d(iw,x[4],x[0]),T_d(iw,x[4],x[1]),T_d(iw,x[4],x[2]),
               T_d(iw,x[5],x[0]),T_d(iw,x[5],x[1]),T_d(iw,x[5],x[2]),scl,&M_d(iw,idet,0,0));
        }
      };
      cub::DeviceFor::Bulk(nwalk*ndet,f);
      break;
    }
    case 4:
    {
      auto f = [=] __device__(long n) {
        long iw   = n/ndet;
        long idet = n - iw*ndet;
        int const* x = iex + 8*idet;
        ov_d(iw,idet) = D4x4(T_d(iw,x[4],x[0]),T_d(iw,x[4],x[1]),T_d(iw,x[4],x[2]),T_d(iw,x[4],x[3]),
                             T_d(iw,x[5],x[0]),T_d(iw,x[5],x[1]),T_d(iw,x[5],x[2]),T_d(iw,x[5],x[3]),
                             T_d(iw,x[6],x[0]),T_d(iw,x[6],x[1]),T_d(iw,x[6],x[2]),T_d(iw,x[6],x[3]),
                             T_d(iw,x[7],x[0]),T_d(iw,x[7],x[1]),T_d(iw,x[7],x[2]),T_d(iw,x[7],x[3]));
        if(abs(ov_d(iw,idet)) != 0) {
          value_t scl = value_t(1.0)/ov_d(iw,idet);
          I4x4(T_d(iw,x[4],x[0]),T_d(iw,x[4],x[1]),T_d(iw,x[4],x[2]),T_d(iw,x[4],x[3]),
               T_d(iw,x[5],x[0]),T_d(iw,x[5],x[1]),T_d(iw,x[5],x[2]),T_d(iw,x[5],x[3]),
               T_d(iw,x[6],x[0]),T_d(iw,x[6],x[1]),T_d(iw,x[6],x[2]),T_d(iw,x[6],x[3]),
               T_d(iw,x[7],x[0]),T_d(iw,x[7],x[1]),T_d(iw,x[7],x[2]),T_d(iw,x[7],x[3]),scl,&M_d(iw,idet,0,0));
        }
      };
      cub::DeviceFor::Bulk(nwalk*ndet,f);
      break;
    }
    default:
    {
      // M[iwalk][idet][ip][iq]
      long n1 = ndet*nex*nex;
      long n2 = nex*nex;
      auto extract = [=] __device__(long n) {
        long iw   = n/n1;
        n -= iw*n1;
        long idet = n/n2;
        n -= idet*n2;
        long i   = n/nex;
        long j   = n - i*nex;
        M_d(iw,idet,i,j) = T_d(iw,iex[2*nex*idet + nex + i],iex[2*nex*idet + j]);
      };
      cub::DeviceFor::Bulk(nwalk*n1,extract);
      auto M3d = nda::reshape(M,std::array<long,3>{ndet*nwalk,nex,nex});
      memory::buffered_array<DEVICE_MEMORY,int,2> ipiv(ndet*nwalk,nex);
      memory::buffered_array<DEVICE_MEMORY,nda::get_value_t<T_t>,1> work;
      nda::lapack::getrf(M3d,ipiv,work);
      math::log_determinant_from_getrf(M3d,ipiv,nda::flatten(ov));
      nda::lapack::getri_or_zero(M3d,ipiv,work);
    }
  }
  // construct compact R
  {
    long n1 = ndet*nex*nel;
    long n2 = nex*nel;
    auto f = [=] __device__(long n) {
      long iw = n/n1;
      n -= iw*n1;
      long idet = n/n2;
      n -= idet*n2;
      if(abs(ov_d(iw,idet)) != 0) {
        long p = n/nel;
        long i = n-p*nel; 
        int a = refc[i];   
        auto iex_ = iex + idet*2*nex;
        for (int q = 0; q < nex; ++q) {
          if(i == iex_[q]) 
            a = iex_[q+nex];
        }
        for (int q = 0; q < nex; ++q) {
          R_d(iw,idet,p,a) -= M_d(iw,idet,p,q) * T_d(iw,iex_[q+nex],i);
          if(i == iex_[q]) 
            R_d(iw,idet,p,a) += M_d(iw,idet,p,q); 
        }
      }
    };
    cub::DeviceFor::Bulk(nwalk*ndet*nex*nel,f);
  }
}

template<typename W_t, typename Rb_t, typename R_t>
void phmsd_reduce_R_impl(int nex, int const* refc, int const* iex, W_t const& wgt, Rb_t const& Rbuff, R_t &R)
{
  using nda::range;
  auto all = range::all;
  // Rbuff(nwalk,ndet,nex,nact) 
  auto Rb_d = to_cuda_std_mdspan(Rbuff);
  // R(nwalk,nel,nact) 
  auto R_d = to_cuda_std_mdspan(R);
  auto w_d = to_cuda_std_mdspan(wgt);
  long ndet = Rbuff.extent(1);
  long ndet_per_thread = 16;
  long nblk = (ndet + ndet_per_thread - 1)/ndet_per_thread;
  long nwalk = R.extent(0);
  long nel = R.extent(1);
  long nact = R.extent(2);
  using value_t = std::decay_t<decltype(R_d(0,0,0))>;

  long n1 = nblk*nel*nact;
  long n2 = nel*nact;
  auto f = [=] __device__(long n) {
    long iw = n/n1;
    n -= iw*n1; 
    long iblk = n/n2;
    n -= iblk*n2;
    long i = n/nact;
    long a = n-i*nact;
    int orb_i = refc[i];   
    value_t y(0);
    long max_ndet = min(ndet,(iblk+1)*ndet_per_thread);
    for(long idet=iblk*ndet_per_thread; idet<max_ndet; ++idet)
    {
      auto iex_ = iex + idet*2*nex;
      orb_i = refc[i];   
      for (int q = 0; q < nex; ++q) 
        if(i == iex_[q]) { 
          orb_i = iex_[q+nex];
          y += w_d(idet,iw) * Rb_d(iw,idet,q,a); 
        }
      if(a==orb_i) y += w_d(idet,iw);
    }
    sfqmc::arch::atomic_add(&R_d(iw, i, a), y);
  };
  cub::DeviceFor::Bulk(nwalk*nblk*nel*nact,f);
}

using memory::device_array_view;
using std::complex;

template<int Rank>
using basic_layout_t = typename nda::basic_layout<0, nda::C_stride_order<Rank>, nda::layout_prop_e::none>;

#define _inst_(T,V) \
template void phmsd_det_impl(int,int const*,V<const T,3,basic_layout_t<3>> const&,V<T,2,basic_layout_t<2>>&); \
template void phmsd_compact_R_impl(int,int const*,int const*,V<const T,3,basic_layout_t<3>> const&,V<T,4,basic_layout_t<4>>&); \
template void phmsd_reduce_R_impl(int,int const*,int const*,V<const T,2,basic_layout_t<2>>const&,V<const T,4,basic_layout_t<4>> const&,V<T,3,basic_layout_t<3>>&); \
template void phmsd_reduce_R_impl(int,int const*,int const*,V<const T,2,basic_layout_t<2>>const&,V<T,4,basic_layout_t<4>> const&,V<T,3,basic_layout_t<3>>&); 

_inst_(std::complex<double>,device_array_view)

} // kernels
