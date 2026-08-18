#include <cuda/std/complex>
#include <cuda/std/mdspan>

#include "cub/block/block_reduce.cuh"

#include "arch/CUDA/cuda_init.h"
#include "arch/atomics.hpp"
#include "numerics/device_kernels/device_api.hpp"

#define __NTHR__ 512 
#define __NCHOL__ 128
#define __BZ__ 32 

namespace kernels::device
{

template<typename S_t, typename R_t, typename W_t, typename E_t> 
__global__ void kernel_ph_excited_1body_energy(int const* iexcit, int const* refc, 
        S_t const S, R_t const R, W_t const w, E_t E) 
{
  using value_t = std::decay_t<decltype(E(0))>;
  //auto [nwalk,ndet,nex,nact] = R.extents();
  //dim3 grid_dim(nwalk*ndet,nex,1);
  int idet = blockIdx.x/R.extent(0);
  int iw = blockIdx.x - idet*R.extent(0); 
  int p = blockIdx.y;
  if(idet >= R.extent(1) or iw >= R.extent(0) or p >= R.extent(2))
    return;

  iexcit += idet*2*R.extent(2);
  value_t sum(0);

  int ip = iexcit[p];
  if(threadIdx.x < R.extent(3))
    for(int ia=threadIdx.x; ia<R.extent(3); ia+=__BZ__)
      sum += S(iw,ip,ia)*R(iw,idet,p,ia); 

  // Specialize BlockReduce for a 1D block of __BZ__ threads 
  using BlockReduce = cub::BlockReduce<value_t, __BZ__>;

  // Allocate shared memory for BlockReduce
  __shared__ typename BlockReduce::TempStorage temp_storage;

  // Compute the block-wide sum for thread0
  value_t aggregate = BlockReduce(temp_storage).Sum(sum);

  if (threadIdx.x == 0)
  {
    aggregate += S(iw,ip,iexcit[p+R.extent(2)]) - S(iw,ip,refc[ip]);
    aggregate *= w(idet,iw);
    sfqmc::arch::atomic_add(&E(iw), aggregate);
  }

}

// assuming single walker version for now
// T(nel,nchol,nact)
// R(ndet,nex,nact)
// wgt(ndet)
// KE(ndet,nchol) 
template<typename T_t, typename R_t, typename W_t, typename KE_t>
__global__ void kernel_ph_excited_energy_real_dense_chol_Tpna_first(int const* iexcit, int const* refc,
    T_t const T, R_t const R, W_t const wgt, cuda::std::complex<double>* EX, cuda::std::complex<double>* EJ, KE_t KE)
{
  using Type = cuda::std::complex<double>;
  __shared__ Type cache[2*__NTHR__+1];
  extern __shared__ int occps[];

  int idet = blockIdx.y;
  int n = blockIdx.x*blockDim.x + threadIdx.x;
  int nelec = T.extent(0);
  int nchol = T.extent(1);
  int nact = T.extent(2);
  int nex = R.extent(1);
  iexcit += idet*2*nex;
  Type keJ(0.0);

  cache[threadIdx.x] = Type(0.0);
  cache[threadIdx.x+blockDim.x] = Type(0.0);

  // calculate occ string
  {
    int i=threadIdx.x;
    while(i < nelec) {
      occps[i] = refc[i];
      for(int ie1=0; ie1<nex; ie1++) {
        int ip = iexcit[ie1];
        if(ip == i) {
          occps[i] = iexcit[ie1 + nex];
          break;
        }
      }
      i+=blockDim.x;
    }
  }

  // calculate eJ0
  if(blockIdx.x==0)
  {
    int c=threadIdx.x;
    while(c < nchol) {
      Type e_(0.0);
      for(int i=0; i<nelec; i++)
        e_ += T(i,c,refc[i]);
      cache[threadIdx.x] += e_*e_;
      c += blockDim.x;
    }
    __syncthreads();
    int i = blockDim.x / 2;
    while (i > 0)
    {
      if (threadIdx.x < i)
        cache[threadIdx.x] += cache[threadIdx.x + i];
      __syncthreads();
      i /= 2; //not sure bitwise operations are actually faster
    }
    if(threadIdx.x==0)
      cache[blockDim.x] -= cache[0];
    cache[threadIdx.x] = Type(0.0);
  } else {
    __syncthreads();
  }

  if(n < nchol) {
    for(int ie1=0; ie1<nex; ie1++) {
      int ip = iexcit[ie1];
      // ie1==ie2 term
      Type e1_(0.0);
      for(int a=0; a<nact; a++)
        e1_ += T(ip,n,a) * R(idet,ie1,a);
      cache[threadIdx.x] += e1_*e1_;
      keJ += e1_;

      // R[p]*R[q] terms
      for(int ie2=ie1+1; ie2<nex; ie2++) {
        int iq = iexcit[ie2];
        e1_=Type(0.0);
        for(int a=0; a<nact; a++)
          e1_ += T(iq,n,a) * R(idet,ie1,a);
        Type e2_(0.0);
        for(int a=0; a<nact; a++)
          e2_ += T(ip,n,a) * R(idet,ie2,a);
        cache[threadIdx.x] += Type(2.0) * e1_*e2_;
      }
    }

    // Rdiag-Rdiag terms
    for(int i=0; i<nelec; i++) {
      int Oi = occps[i];
      int ri = refc[i];
      if( not(Oi==ri) )
        cache[threadIdx.x] += T(i,n,Oi)*T(i,n,Oi) - T(i,n,ri)*T(i,n,ri);
      for(int j=i+1; j<nelec; j++) {
        int Oj = occps[j];
        int rj = refc[j];
        if( Oi!=ri or Oj!=rj )
          cache[threadIdx.x] += Type(2.0) * (T(i,n,Oj)*T(j,n,Oi) - T(i,n,rj)*T(j,n,ri));
      }
    }

    // R[diagonal]*R[diagonal] J-term
    for(int i=0; i<nelec; i++)
      keJ += T(i,n,occps[i]);

    KE(idet,n) = keJ;
    cache[threadIdx.x+blockDim.x] += keJ*keJ;
  }
  __syncthreads();
  n = blockDim.x / 2;
  while (n > 0)
  {
    if (threadIdx.x < n) {
      cache[threadIdx.x] += cache[threadIdx.x + n];
      cache[threadIdx.x+blockDim.x] += cache[blockDim.x + threadIdx.x + n];
    }
    __syncthreads();
    n /= 2; //not sure bitwise operations are actually faster
  }
  if (threadIdx.x == 0)
  {
    // ex[iw][d] *= -0.5*wgt[d][iw]
    cache[0] *= -Type(0.5)*wgt(idet);
    // eJ[iw][d] *= 0.5*wgt[d][iw]
    cache[blockDim.x] *= Type(0.5)*wgt(idet);
    // EX[iw] += eX[iw][d]
    sfqmc::arch::atomic_add(EX, cache[0]);
    // EJ[iw] += eJ[iw][d]
    sfqmc::arch::atomic_add(EJ, cache[blockDim.x]);
  }
}

// T(nel,nchol,nact)
// R(ndet,nex,nact)
// wgt(ndet)
template<typename T_t, typename R_t, typename W_t>
__global__ void kernel_ph_excited_energy_real_dense_chol_Tpna_second(int const* iexcit, int const* refc, T_t const T, R_t const R, W_t const wgt, cuda::std::complex<double>* EX)
{
  using Type = cuda::std::complex<double>;
  __shared__ Type cache[__BZ__];
  extern __shared__ int occps[];

  int idet = blockIdx.y;
  int nelec = T.extent(0);
  int nchol = T.extent(1);
  int nact = T.extent(2);
  int nex = R.extent(1);
  iexcit += idet*2*nex;

  cache[threadIdx.x] = Type(0.0);

  // calculate occ string
  {
    int i=threadIdx.x;
    while(i < nelec) {
      occps[i] = refc[i];
      for(int ie1=0; ie1<nex; ie1++) {
        int ip = iexcit[ie1];
        if(ip == i) {
          occps[i] = iexcit[ie1 + nex];
          break;
        }
      }
      i+=blockDim.x;
    }
  }
  __syncthreads();

  if(threadIdx.x < nact) {
    int n1 = min(nchol, (blockIdx.x+1)*__NCHOL__);
    for(int n=blockIdx.x*__NCHOL__; n<n1; n++)
    {
      for(int ie1=0; ie1<nex; ie1++) {
        int ip = iexcit[ie1];

       // R[p]*R[diagonal] term
        for(int j=0; j<nelec; j++) {
          int Oj = occps[j];
          Type e1_(0.0);
          for(int a=threadIdx.x; a<nact; a+=blockDim.x)
            e1_ += T(j,n,a) * R(idet,ie1,a);
          cache[threadIdx.x] += e1_*T(ip,n,Oj);
        }
      }
    }
  }
  __syncthreads();
  int n = blockDim.x / 2;
  while (n > 0)
  {
    if (threadIdx.x < n)
      cache[threadIdx.x] += cache[threadIdx.x + n];
    __syncthreads();
    n /= 2; //not sure bitwise operations are actually faster
  }
  if (threadIdx.x == 0)
  {
    // ex[iw][d] *= -0.5*wgt[d][iw]
    // factor of TWO from above cancels factor of HALF here!    
    cache[0] *= -wgt(idet);
    // EX[iw] += eX[iw][d]
    sfqmc::arch::atomic_add(EX, cache[0]);
  }
} 

// S(nwalk,nel,nact), R(nwalk,ndet,nex,nact), w(ndet,nwalk), E(nwalk)
void ph_excited_1body_energy(int const* refc, int const* iexcit, view<std::complex<double> const, 3> S,
                             view<std::complex<double> const, 4> R,
                             view<std::complex<double> const, 2> w, view<std::complex<double>, 1> E)
{
  long nwalk = R.extent(0);
  long ndet  = R.extent(1);
  long nex   = R.extent(2);
  if(nwalk * ndet * nex == 0) {
    return;
  }

  dim3 grid_dim((unsigned)(ndet * nwalk), (unsigned)nex, 1);
  kernel_ph_excited_1body_energy<<<grid_dim, __BZ__>>>(iexcit, refc, S, R, w, E);
  sfqmc::cuda::cuda_check(cudaGetLastError(), "ph_excited_1body_energy");
}

// One walker's contribution. The caller loops over walkers and slices T(iw,...), R(iw,...),
// w(:,iw) and KE(:,iw,:) with nda, which is why the slicing does not appear here.
//   T(nel,nchol,nact), R(ndet,nex,nact), w(ndet), KE(ndet,nchol)
void ph_excited_2body_energy_dense_cholesky_Tpna_walker(
    int const* refc, int const* iexcit, view<std::complex<double> const, 3> T,
    view<std::complex<double> const, 3> R, view<std::complex<double> const, 1> w,
    view<std::complex<double>, 1> EX, view<std::complex<double>, 1> EJ, long iw,
    view<std::complex<double>, 2> KE)
{
  long nelec = T.extent(0);
  long nchol = T.extent(1);
  long ndet  = R.extent(0);
  if(nelec * nchol * ndet == 0) {
    return;
  }

  long bsize = __NTHR__;
  long nc    = __NCHOL__;
  dim3 grid_dim((unsigned)((nchol + bsize - 1) / bsize), (unsigned)ndet, 1);
  dim3 grid_dim2((unsigned)((nchol + nc - 1) / nc), (unsigned)ndet, 1);
  int  sm_size(nelec * sizeof(int));

  kernel_ph_excited_energy_real_dense_chol_Tpna_first<<<grid_dim, __NTHR__, sm_size>>>(
      iexcit, refc, T, R, w, &EX(iw), &EJ(iw), KE);
  kernel_ph_excited_energy_real_dense_chol_Tpna_second<<<grid_dim2, __BZ__, sm_size>>>(iexcit, refc,
                                                                                       T, R, w,
                                                                                       &EX(iw));
  sfqmc::cuda::cuda_check(cudaGetLastError(), "ph_excited_2body_energy_dense_cholesky_Tpna");
}

} // namespace kernels::device
