#include <cuda/std/complex>
#include <cuda/std/cmath>
#include <cuda/std/mdspan>

#include "arch/atomics.hpp"
#include "numerics/device_kernels/cuda/launch.cuh"
#include "numerics/device_kernels/device_api.hpp"

namespace kernels::device
{
namespace {
using ph_complex=cuda::std::complex<double>;
__device__ ph_complex ph_shuffle(ph_complex z,int lane) {
  return {__shfl_sync(0xffffffff,z.real(),lane),__shfl_sync(0xffffffff,z.imag(),lane)};
}

// One warp owns one small matrix. Rows live in registers, with partial
// pivoting across lanes. No pointer arrays, pivots, or LU workspace transfers.
template<int N,bool Inverse,typename T_t,typename O_t,typename M_t>
__global__ void kernel_ph_pivoted(int const* iex,T_t T,O_t ov,M_t M) {
  int lane=threadIdx.x%32;
  long batch=(long(blockIdx.x)*blockDim.x+threadIdx.x)/32;
  long nw=Inverse ? ov.extent(0) : ov.extent(1);
  long nd=Inverse ? ov.extent(1) : ov.extent(0);
  if(batch>=nw*nd) return; // uniform within the warp
  long d=batch/nw, iw=batch%nw;
  auto ex=iex+2*N*d;
  ph_complex a[N], inverse[N], det(1);
  int determinant_exponent=0;
#pragma unroll
  for(int j=0; j<N; ++j) {
    a[j]=lane<N ? T(iw,ex[N+lane],ex[j]) : ph_complex(0);
    if constexpr(Inverse) inverse[j]=ph_complex(lane==j ? 1 : 0);
  }
#pragma unroll
  for(int k=0; k<N; ++k) {
    double score=(lane>=k && lane<N) ? cuda::std::abs(a[k]) : -1.;
    int pivot_row=lane;
#pragma unroll
    for(int offset=16; offset>0; offset/=2) {
      double other=__shfl_down_sync(0xffffffff,score,offset);
      int other_row=__shfl_down_sync(0xffffffff,pivot_row,offset);
      if(lane+offset<32 && (other>score || (other==score && other_row<pivot_row))) {
        score=other; pivot_row=other_row;
      }
    }
    pivot_row=__shfl_sync(0xffffffff,pivot_row,0);
    int source=lane==k ? pivot_row : (lane==pivot_row ? k : lane);
#pragma unroll
    for(int j=0; j<N; ++j) {
      a[j]=ph_shuffle(a[j],source);
      if constexpr(Inverse) inverse[j]=ph_shuffle(inverse[j],source);
    }
    auto pivot=ph_shuffle(a[k],k);
    if(cuda::std::abs(pivot)==0.) {
      if(lane==0) {
        if constexpr(Inverse) ov(iw,d)=ph_complex(0);
        else ov(d,iw)=ph_complex(0);
      }
      if constexpr(Inverse) if(lane<N) {
#pragma unroll
        for(int j=0; j<N; ++j) M(iw,d,lane,j)=ph_complex(0);
      }
      return;
    }
    // Scale each pivot by a power of two so intermediate products cannot
    // overflow/underflow when the final determinant is representable.
    int exponent;
    cuda::std::frexp(cuda::std::fmax(cuda::std::abs(pivot.real()),cuda::std::abs(pivot.imag())),&exponent);
    ph_complex scaled(cuda::std::ldexp(pivot.real(),-exponent),cuda::std::ldexp(pivot.imag(),-exponent));
    determinant_exponent += exponent;
    det *= pivot_row==k ? scaled : -scaled;
    if constexpr(Inverse) {
      auto reciprocal=ph_complex(1)/pivot;
      if(lane==k) {
#pragma unroll
        for(int j=0; j<N; ++j) { a[j]*=reciprocal; inverse[j]*=reciprocal; }
      }
      auto factor=a[k];
#pragma unroll
      for(int j=0; j<N; ++j) {
        auto aj=ph_shuffle(a[j],k), bj=ph_shuffle(inverse[j],k);
        if(lane<N && lane!=k) { a[j]-=factor*aj; inverse[j]-=factor*bj; }
      }
    } else {
      auto factor=a[k]/pivot;
#pragma unroll
      for(int j=0; j<N; ++j) {
        auto aj=ph_shuffle(a[j],k);
        if(lane<N && lane>k && j>k) a[j]-=factor*aj;
      }
      if(lane>k) a[k]=ph_complex(0);
    }
  }
  if(lane==0) {
    det={cuda::std::ldexp(det.real(),determinant_exponent),cuda::std::ldexp(det.imag(),determinant_exponent)};
    if constexpr(Inverse) ov(iw,d)=det;
    else ov(d,iw)=det;
  }
  if constexpr(Inverse) if(lane<N) {
#pragma unroll
    for(int j=0; j<N; ++j) M(iw,d,lane,j)=inverse[j];
  }
}

template<bool Inverse>
void launch_ph_pivoted(int nex,int const* iex,view<std::complex<double> const,3> T,
                       view<std::complex<double>,2> ov,view<std::complex<double>,4> M) {
  long batches=ov.extent(0)*ov.extent(1);
  if(batches==0) return;
  long blocks=(batches+3)/4;
  switch(nex) {
    case 5: kernel_ph_pivoted<5,Inverse><<<blocks,128>>>(iex,T,ov,M); break;
    case 6: kernel_ph_pivoted<6,Inverse><<<blocks,128>>>(iex,T,ov,M); break;
    case 7: kernel_ph_pivoted<7,Inverse><<<blocks,128>>>(iex,T,ov,M); break;
    case 8: kernel_ph_pivoted<8,Inverse><<<blocks,128>>>(iex,T,ov,M); break;
  }
  sfqmc::cuda::cuda_check(cudaGetLastError(),"phmsd_pivoted");
}
} // namespace

void phmsd_det_pivoted(int nex,int const* iex,view<std::complex<double> const,3> T,
                       view<std::complex<double>,2> ov) {
  launch_ph_pivoted<false>(nex,iex,T,ov,view<std::complex<double>,4>{});
}
void phmsd_inverse_pivoted(int nex,int const* iex,view<std::complex<double> const,3> T,
                           view<std::complex<double>,2> ov,view<std::complex<double>,4> M) {
  launch_ph_pivoted<true>(nex,iex,T,ov,M);
}

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

namespace {
using density_complex=cuda::std::complex<double>;
__global__ void density_weight_sum(view<std::complex<double> const,2> ov,
    view<std::complex<double> const,2> weights, view<std::complex<double>,1> total)
{
  __shared__ double real[128],imag[128];
  long iw=blockIdx.x;
  density_complex sum(0);
  for(long d=threadIdx.x; d<ov.extent(1); d+=blockDim.x)
    if(cuda::std::abs(ov(iw,d))!=0.) sum+=weights(d,iw);
  real[threadIdx.x]=sum.real();
  imag[threadIdx.x]=sum.imag();
  __syncthreads();
  for(int offset=64; offset>0; offset/=2) {
    if(threadIdx.x<offset) {
      real[threadIdx.x]+=real[threadIdx.x+offset];
      imag[threadIdx.x]+=imag[threadIdx.x+offset];
    }
    __syncthreads();
  }
  if(threadIdx.x==0) total(iw)+=density_complex(real[0],imag[0]);
}
} // namespace

void phmsd_density_coefficients(int nex, int const* iex,
    view<std::complex<double> const,2> ov, view<std::complex<double> const,4> M,
    view<std::complex<double> const,2> weights, view<std::complex<double>,3> C,
    view<std::complex<double>,1> total)
{
  long nw=M.extent(0), nd=M.extent(1);
  auto f=[=] __device__(long iw,long d,long p,long q) {
    if(cuda::std::abs(ov(iw,d))!=0.) {
      auto ex=iex+2*nex*d;
      sfqmc::arch::atomic_add(&C(iw,ex[p],ex[nex+q]),weights(d,iw)*M(iw,d,p,q));
    }
  };
  for_each_extents<4>({nw,nd,nex,nex},f);
  if(nw>0 && nd>0) {
    density_weight_sum<<<nw,128>>>(ov,weights,total);
    sfqmc::cuda::cuda_check(cudaGetLastError(),"phmsd_density_coefficients");
  }
}

void phmsd_density_reference(int const* refc, view<std::complex<double> const,3> product,
    view<std::complex<double> const,1> total, view<std::complex<double>,3> R)
{
  auto f=[=] __device__(long iw,long i,long j) {
    R(iw,i,refc[j])+=(i==j ? total(iw) : density_complex(0))-product(iw,i,j);
  };
  for_each_extents<3>({R.extent(0),R.extent(1),R.extent(1)},f);
}

} // namespace kernels::device
