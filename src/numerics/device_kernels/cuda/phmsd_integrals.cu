// Distributed under the Apache License, Version 2.0. See LICENSE.
#include <cuda/std/complex>
#include <cub/block/block_reduce.cuh>
#include "arch/atomics.hpp"
#include "numerics/device_kernels/cuda/launch.cuh"
#include "numerics/device_kernels/device_api.hpp"

namespace kernels::device {
namespace {
using C=cuda::std::complex<double>;
constexpr int nt=128;

template<typename X_t,typename F_t,typename R_t,typename W_t,typename E_t>
__global__ void integral_exchange(int const* refc,int const* iex,X_t X,F_t F,R_t R,W_t w,E_t EX) {
  long nd=R.extent(1), nx=R.extent(2), na=R.extent(3), ni=X.extent(1);
  long d=blockIdx.x%nd, iw=blockIdx.x/nd;
  auto ex=iex+2*nx*d;
  extern __shared__ C rc[];
  auto occ=reinterpret_cast<int*>(rc+nx*na);
  using Reduce=cub::BlockReduce<C,nt>;
  __shared__ typename Reduce::TempStorage scratch;
  for(long pa=threadIdx.x; pa<nx*na; pa+=nt) rc[pa]=R(iw,d,pa/na,pa%na);
  for(long i=threadIdx.x; i<ni; i+=nt) {
    int oi=refc[i];
    for(long p=0; p<nx; ++p) if(ex[p]==i) oi=ex[nx+p];
    occ[i]=oi;
  }
  __syncthreads();
  C sum(0);
  for(long ab=threadIdx.x; ab<na*na; ab+=nt) {
    long a=ab/na,b=ab%na;
    for(long p=0; p<nx; ++p)
      for(long q=0; q<nx; ++q)
        sum += rc[p*na+a]*rc[q*na+b]*X(iw,ex[q],a,ex[p],b);
  }
  for(long pa=threadIdx.x; pa<nx*na; pa+=nt) {
    long p=pa/na,a=pa%na;
    int hp=ex[p];
    C mixed=F(iw,hp,a);
    for(long q=0; q<nx; ++q)
      mixed += X(iw,ex[q],a,hp,ex[nx+q])-X(iw,ex[q],a,hp,refc[ex[q]]);
    sum += C(2)*rc[pa]*mixed;
  }
  for(long p=threadIdx.x; p<nx; p+=nt) {
    int hp=ex[p],op=ex[nx+p],rp=refc[hp];
    sum += X(iw,hp,op,hp,op)-X(iw,hp,rp,hp,rp);
    for(long j=0; j<ni; ++j) if(occ[j]==refc[j])
      sum += C(2)*(X(iw,hp,refc[j],j,op)-X(iw,hp,refc[j],j,rp));
    for(long q=p+1; q<nx; ++q)
      sum += C(2)*(X(iw,hp,ex[nx+q],ex[q],op)-X(iw,hp,refc[ex[q]],ex[q],rp));
  }
  C total=Reduce(scratch).Sum(sum);
  if(threadIdx.x==0) sfqmc::arch::atomic_add(&EX(iw),-C(.5)*w(d,iw)*total);
}

template<typename T_t,typename K0_t,typename R_t,typename W_t,typename E_t,typename K_t>
__global__ void integral_coulomb(int const* refc,int const* iex,T_t T,K0_t K0,R_t R,W_t w,E_t EJ,K_t KE) {
  long nd=R.extent(1),nx=R.extent(2),na=R.extent(3),nn=T.extent(3);
  long groups=(nn+nt-1)/nt,system=blockIdx.x/groups;
  long d=system%nd,iw=system/nd,n=(blockIdx.x%groups)*nt+threadIdx.x;
  auto ex=iex+2*nx*d;
  extern __shared__ C rc[];
  using Reduce=cub::BlockReduce<C,nt>;
  __shared__ typename Reduce::TempStorage scratch;
  for(long pa=threadIdx.x; pa<nx*na; pa+=nt) rc[pa]=R(iw,d,pa/na,pa%na);
  __syncthreads();
  C sum(0);
  if(n<nn) {
    C k=K0(iw,n);
    for(long p=0; p<nx; ++p) {
      long hp=ex[p];
      for(long a=0; a<na; ++a) k += rc[p*na+a]*T(iw,hp,a,n);
      k += T(iw,hp,ex[nx+p],n)-T(iw,hp,refc[hp],n);
    }
    KE(d,iw,n)=k;
    sum=k*k-K0(iw,n)*K0(iw,n);
  }
  C total=Reduce(scratch).Sum(sum);
  if(threadIdx.x==0) sfqmc::arch::atomic_add(&EJ(iw),C(.5)*w(d,iw)*total);
}
} // namespace

void phmsd_integral_reference(int const* refc,view<std::complex<double> const,4> T,
    view<std::complex<double> const,5> X,view<std::complex<double>,3> F,
    view<std::complex<double>,2> K0) {
  auto f=[=] __device__(long iw,long i,long a) {
    C sum(0);
    for(long j=0; j<T.extent(1); ++j) sum += X(iw,j,a,i,refc[j]);
    F(iw,i,a)=sum;
  };
  for_each(F,f);
  auto k=[=] __device__(long iw,long n) {
    C sum(0);
    for(long i=0; i<T.extent(1); ++i) sum += T(iw,i,refc[i],n);
    K0(iw,n)=sum;
  };
  for_each(K0,k);
}

void phmsd_integral_energy(int const* refc,int const* iex,
    view<std::complex<double> const,4> T,view<std::complex<double> const,5> X,
    view<std::complex<double> const,3> F,view<std::complex<double> const,2> K0,
    view<std::complex<double> const,4> R,view<std::complex<double> const,2> w,
    view<std::complex<double>,1> EX,view<std::complex<double>,1> EJ,
    view<std::complex<double>,3> KE) {
  long systems=R.extent(0)*R.extent(1);
  if(systems==0) return;
  size_t rbytes=R.extent(2)*R.extent(3)*sizeof(C);
  integral_exchange<<<systems,nt,rbytes+T.extent(1)*sizeof(int)>>>(refc,iex,X,F,R,w,EX);
  integral_coulomb<<<systems*((T.extent(3)+nt-1)/nt),nt,rbytes>>>(refc,iex,T,K0,R,w,EJ,KE);
  sfqmc::cuda::cuda_check(cudaGetLastError(),"phmsd_integral_energy");
}
} // namespace kernels::device
