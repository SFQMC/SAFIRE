#include <cuda/std/complex>
#include <cuda/std/mdspan>

#include "arch/atomics.hpp"
#include "numerics/device_kernels/cuda/launch.cuh"
#include "numerics/device_kernels/device_api.hpp"

namespace kernels::device
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

// ov(ndet,nwalk), T(nwalk,nact,nel). Closed forms up to nex == phmsd_det_max_closed_form; beyond
// that the host wrapper drives getrf, so only the extract kernel below is needed.
void phmsd_det_small(int nex, int const* iex, view<std::complex<double> const, 3> T,
                     view<std::complex<double>, 2> ov)
{
  auto T_d  = T;
  auto ov_d = ov;
  long ndet = ov.extent(0);
  long nwalk = ov.extent(1);
  if(ndet * nwalk == 0) {
    return;
  }
  switch(nex)
  {
    case 1:
    {
      auto f = [=] __device__(long iw, long idet) {
        ov_d(idet,iw) = T_d(iw,iex[2*idet+1],iex[2*idet]);
      };
      for_each_extents<2>({nwalk,ndet},f);
      break;
    }
    case 2:
    {
      auto f = [=] __device__(long iw, long idet) {
        int const* x = iex + 4*idet;
        ov_d(idet,iw) = T_d(iw,x[2],x[0])*T_d(iw,x[3],x[1]) - T_d(iw,x[2],x[1])*T_d(iw,x[3],x[0]);
      };
      for_each_extents<2>({nwalk,ndet},f);
      break;
    }
    case 3:
    {
      auto f = [=] __device__(long iw, long idet) {
        int const* x = iex + 6*idet;
        ov_d(idet,iw) = _D3x3_(T_d(iw,x[3],x[0]),T_d(iw,x[3],x[1]),T_d(iw,x[3],x[2]),
                             T_d(iw,x[4],x[0]),T_d(iw,x[4],x[1]),T_d(iw,x[4],x[2]),
                             T_d(iw,x[5],x[0]),T_d(iw,x[5],x[1]),T_d(iw,x[5],x[2]));
      };
      for_each_extents<2>({nwalk,ndet},f);
      break;
    }
    case 4:
    {
      auto f = [=] __device__(long iw, long idet) {
        int const* x = iex + 8*idet;
        ov_d(idet,iw) = D4x4(T_d(iw,x[4],x[0]),T_d(iw,x[4],x[1]),T_d(iw,x[4],x[2]),T_d(iw,x[4],x[3]),
                             T_d(iw,x[5],x[0]),T_d(iw,x[5],x[1]),T_d(iw,x[5],x[2]),T_d(iw,x[5],x[3]),
                             T_d(iw,x[6],x[0]),T_d(iw,x[6],x[1]),T_d(iw,x[6],x[2]),T_d(iw,x[6],x[3]),
                             T_d(iw,x[7],x[0]),T_d(iw,x[7],x[1]),T_d(iw,x[7],x[2]),T_d(iw,x[7],x[3]));
      };
      for_each_extents<2>({nwalk,ndet},f);
      break;
    }
    case 5:
    {
      auto f = [=] __device__(long iw, long idet) {
        int const* x = iex + 10*idet;
        ov_d(idet,iw) = D5x5(T_d(iw,x[5],x[0]),T_d(iw,x[5],x[1]),T_d(iw,x[5],x[2]),T_d(iw,x[5],x[3]),T_d(iw,x[5],x[4]),
                             T_d(iw,x[6],x[0]),T_d(iw,x[6],x[1]),T_d(iw,x[6],x[2]),T_d(iw,x[6],x[3]),T_d(iw,x[6],x[4]),
                             T_d(iw,x[7],x[0]),T_d(iw,x[7],x[1]),T_d(iw,x[7],x[2]),T_d(iw,x[7],x[3]),T_d(iw,x[7],x[4]),
                             T_d(iw,x[8],x[0]),T_d(iw,x[8],x[1]),T_d(iw,x[8],x[2]),T_d(iw,x[8],x[3]),T_d(iw,x[8],x[4]),
                             T_d(iw,x[9],x[0]),T_d(iw,x[9],x[1]),T_d(iw,x[9],x[2]),T_d(iw,x[9],x[3]),T_d(iw,x[9],x[4]));
      };
      for_each_extents<2>({nwalk,ndet},f);
      break;
    }
    default:
      break;
  };
}

// M(ndet,nwalk,nex,nex) <- the nex x nex excitation block, ready for a batched getrf.
void phmsd_det_extract(int nex, int const* iex, view<std::complex<double> const, 3> T,
                       view<std::complex<double>, 4> M)
{
  auto T_d  = T;
  auto M_d  = M;
  auto extract = [=] __device__(long idet, long iw, long i, long j) {
    M_d(idet,iw,i,j) = T_d(iw,iex[2*nex*idet + nex + i],iex[2*nex*idet + j]);
  };
  for_each(M,extract);
}

// ov(nwalk,ndet) and its inverse M(nwalk,ndet,nex,nex), closed forms up to
// nex == phmsd_inverse_max_closed_form. Wherever the overlap is zero M is left untouched, and
// phmsd_compact_R_assemble skips those entries, so M needs no initialization.
void phmsd_compact_R_inverse_small(int nex, int const* iex, view<std::complex<double> const, 3> T,
                                   view<std::complex<double>, 2> ov,
                                   view<std::complex<double>, 4> M)
{
  auto T_d  = T;
  auto ov_d = ov;
  auto M_d  = M;
  using value_t = native_t<std::complex<double>>;
  long nwalk = ov.extent(0);
  long ndet  = ov.extent(1);
  if(nwalk * ndet == 0) {
    return;
  }
  switch(nex)
  {
    case 1:
    {
      auto f = [=] __device__(long iw, long idet) {
        ov_d(iw,idet) = T_d(iw,iex[2*idet+1],iex[2*idet]);
        if(abs(ov_d(iw,idet)) != 0)
          M_d(iw,idet,0,0) = value_t(1.0)/ov_d(iw,idet);
      };
      for_each(ov,f);
      break;
    }
    case 2:
    {
      auto f = [=] __device__(long iw, long idet) {
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
      for_each(ov,f);
      break;
    }
    case 3:
    {
      auto f = [=] __device__(long iw, long idet) {
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
      for_each(ov,f);
      break;
    }
    case 4:
    {
      auto f = [=] __device__(long iw, long idet) {
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
      for_each(ov,f);
      break;
    }
    default:
      break;
  }
}

// M(nwalk,ndet,nex,nex) <- the nex x nex excitation block. The index order differs from
// phmsd_det_extract: this one is walker-major.
void phmsd_compact_R_extract(int nex, int const* iex, view<std::complex<double> const, 3> T,
                             view<std::complex<double>, 4> M)
{
  auto T_d = T;
  auto M_d = M;
  auto extract = [=] __device__(long iw, long idet, long i, long j) {
    M_d(iw,idet,i,j) = T_d(iw,iex[2*nex*idet + nex + i],iex[2*nex*idet + j]);
  };
  for_each(M,extract);
}

// R(nwalk,ndet,nex,nact) from the inverse M and the overlaps ov
void phmsd_compact_R_assemble(int nex, int const* refc, int const* iex,
                              view<std::complex<double> const, 3> T,
                              view<std::complex<double> const, 2> ov,
                              view<std::complex<double> const, 4> M,
                              view<std::complex<double>, 4> R)
{
  auto T_d  = T;
  auto ov_d = ov;
  auto M_d  = M;
  auto R_d  = R;
  long nwalk = R.extent(0);
  long ndet  = R.extent(1);
  long nel   = T.extent(2);
  auto f = [=] __device__(long iw, long idet, long p, long i) {
    if(abs(ov_d(iw,idet)) != 0) {
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
  for_each_extents<4>({nwalk,ndet,nex,nel},f);
}

// Rbuff(nwalk,ndet,nex,nact), R(nwalk,nel,nact), wgt(ndet,nwalk)
void phmsd_reduce_R(int nex, int const* refc, int const* iex, view<std::complex<double> const, 2> wgt,
                    view<std::complex<double> const, 4> Rbuff, view<std::complex<double>, 3> R)
{
  auto Rb_d = Rbuff;
  auto R_d  = R;
  auto w_d  = wgt;
  long ndet = Rbuff.extent(1);
  long ndet_per_thread = 16;
  long nblk = (ndet + ndet_per_thread - 1)/ndet_per_thread;
  long nwalk = R.extent(0);
  long nel = R.extent(1);
  long nact = R.extent(2);
  if(nwalk * nel * nact == 0) {
    return;
  }
  using value_t = native_t<std::complex<double>>;

  auto f = [=] __device__(long iw, long iblk, long i, long a) {
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
  for_each_extents<4>({nwalk,nblk,nel,nact},f);
}

} // namespace kernels::device
