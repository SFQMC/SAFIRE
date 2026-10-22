// Distributed under the Apache License, Version 2.0. See LICENSE.
#pragma once

namespace sfqmc::afqmc {

inline constexpr long phmsd_exchange_cache_bytes=64L*1024*1024;
inline long phmsd_exchange_cache_walkers(long nelec,long nact,long nchol=0) {
  long pair_size=nelec*nact;
  long elements=phmsd_exchange_cache_bytes/sizeof(ComplexType);
  if(pair_size<=0 || pair_size>elements/pair_size) return 0;
  long per_walker=pair_size*pair_size;
  if(nchol>0) {
    // Device cache also retains T(i,a,n), F(i,a) and the reference trace K0(n).
    if(pair_size>elements-per_walker) return 0;
    per_walker+=pair_size;
    if(nchol>(elements-per_walker)/(pair_size+1)) return 0;
    per_walker+=(pair_size+1)*nchol;
  }
  return elements/per_walker;
}

// CPU path for compact active spaces with many Cholesky vectors. X is the
// exact, non-Hermitian contraction X[i,a,j,b] = sum_n T[i,n,a] T[j,n,b].
// It is cached once per reference walker/spin, rather than once per determinant
// block. The caller bounds its storage and falls back to the Cholesky path.
void ph_excited_2body_energy_integrals(nda::MemoryVector auto const& iexcit,
    nda::MemoryVector auto const& refc, nda::MemoryArrayOfRank<4> auto const& Twina,
    nda::MemoryArrayOfRank<4> auto const& R, nda::MemoryMatrix auto const& wgt,
    nda::MemoryVector auto&& EX, nda::MemoryVector auto&& EJ,
    nda::MemoryArrayOfRank<3> auto&& KE, nda::MemoryArrayOfRank<5> auto const& Xw)
{
  using nda::range;
  auto all=range::all;
  static_assert(memory::get_memory_space<decltype(Twina)>() == HOST_MEMORY);
  auto [nw,nd,nx,na]=R.shape();
  long ni=Twina.extent(1), nn=Twina.extent(2);
  memory::buffered_array<HOST_MEMORY,ComplexType,2> F(ni,na);
  memory::buffered_array<HOST_MEMORY,ComplexType,1> tmp(na), mixed(na), k0(nn);
  nda::vector<int> occ(ni);
  for(long iw=0; iw<nw; ++iw) {
    auto T=Twina(iw,all,all,all);
    auto X=Xw(iw,all,all,all,all);
    F()=ComplexType(0); k0()=ComplexType(0);
    for(long j=0; j<ni; ++j) {
      k0() += T(j,all,refc(j));
      for(long i=0; i<ni; ++i) F(i,all) += X(j,all,i,refc(j));
    }
    auto ej0=nda::sum(k0*k0);
    for(long d=0; d<nd; ++d) {
      auto ex=iexcit.data()+2*nx*d;
      for(long i=0; i<ni; ++i) occ(i)=refc(i);
      for(long p=0; p<nx; ++p) occ(ex[p])=ex[nx+p];
      auto K=KE(d,iw,all);
      K()=k0;
      ComplexType exchange=0.;
      for(long p=0; p<nx; ++p) {
        int hp=ex[p], op=ex[nx+p], rp=refc(hp);
        auto rpvec=R(iw,d,p,all);
        nda::blas::gemv(ComplexType(1),T(hp,all,all),rpvec,ComplexType(1),K);
        K() += T(hp,all,op)-T(hp,all,rp);
        // The quadratic compact-R contribution; X is symmetric under exchange
        // of its (i,a) and (j,b) pairs, without complex conjugation.
        for(long q=p; q<nx; ++q) {
          nda::blas::gemv(ComplexType(1),nda::transpose(X(ex[q],all,hp,all)),
                         rpvec,ComplexType(0),tmp);
          exchange += ComplexType(p==q ? 1 : 2)*nda::blas::dot(tmp,R(iw,d,q,all));
        }
        mixed()=F(hp,all);
        for(long q=0; q<nx; ++q)
          mixed() += X(ex[q],all,hp,ex[nx+q])-X(ex[q],all,hp,refc(ex[q]));
        exchange += ComplexType(2)*nda::blas::dot(rpvec,mixed);

        // Only occupation pairs touching an excited hole can change.
        exchange += X(hp,op,hp,op)-X(hp,rp,hp,rp);
        for(long j=0; j<ni; ++j) if(occ(j)==refc(j))
          exchange += ComplexType(2)*(X(hp,refc(j),j,op)-X(hp,refc(j),j,rp));
        for(long q=p+1; q<nx; ++q)
          exchange += ComplexType(2)*(X(hp,ex[nx+q],ex[q],op)-X(hp,refc(ex[q]),ex[q],rp));
      }
      EX(iw) -= ComplexType(.5)*wgt(d,iw)*exchange;
      EJ(iw) += ComplexType(.5)*wgt(d,iw)*(nda::sum(K*K)-ej0);
    }
  }
}
} // namespace sfqmc::afqmc
