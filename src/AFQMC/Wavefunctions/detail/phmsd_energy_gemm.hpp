// Distributed under the Apache License, Version 2.0. See LICENSE.
#pragma once

namespace sfqmc::afqmc {

// Exact change of contraction order, without conjugation:
// B[d,p,i,n] = sum_a R[d,p,a] T[i,n,a].
// Contract a block of determinants with GEMM, then form the exchange and
// Coulomb corrections. Retain the caller's determinant/walker batching and
// additionally cap this temporary at 64 MiB (or one determinant).
void ph_excited_2body_energy_gemm(nda::MemoryVector auto const& iexcit,
    nda::MemoryVector auto const& refc, nda::MemoryArrayOfRank<4> auto const& Twina,
    nda::MemoryArrayOfRank<4> auto const& R, nda::MemoryMatrix auto const& wgt,
    nda::MemoryVector auto&& EX, nda::MemoryVector auto&& EJ,
    nda::MemoryArrayOfRank<3> auto&& KE)
{
  using nda::range;
  auto all = range::all;
  constexpr auto MEM = memory::get_memory_space<decltype(Twina)>();
  auto [nw, nd, nx, na] = R.shape();
  long ni = Twina.extent(1), nn = Twina.extent(2);
  if(nw*nd*nx*ni*nn == 0) return;
  constexpr long workspace_bytes = 64L*1024*1024;
  long block = std::min(nd, std::max(1L, workspace_bytes /
                                      (long(sizeof(ComplexType))*nx*ni*nn)));
  memory::buffered_array<MEM,ComplexType,4> buffer(block,nx,ni,nn);
  memory::buffered_array<MEM,ComplexType,3> Tan(ni,na,nn);
  nda::vector<int> occ;
  if constexpr(MEM == HOST_MEMORY) occ.resize(ni);
  for(long iw=0; iw<nw; ++iw) {
    auto T = Twina(iw,all,all,all);
    nda::tensor::add(ComplexType(1),T,"ina",ComplexType(0),Tan,"ian");
    auto tm = nda::reshape(T,std::array<long,2>{ni*nn,na});
    for(long first=0; first<nd; first+=block) {
      long count=std::min(block,nd-first);
      auto B = buffer(range(count),all,all,all);
      auto rm = nda::reshape(R(iw,range(first,first+count),all,all),
                             std::array<long,2>{count*nx,na});
      auto bm = nda::reshape(B,std::array<long,2>{count*nx,ni*nn});
      nda::blas::gemm(ComplexType(1),rm,nda::transpose(tm),ComplexType(0),bm);
      if constexpr(MEM == HOST_MEMORY) {
        memory::buffered_array<MEM,ComplexType,1> x(nn), k0(nn);
        k0()=ComplexType(0);
        for(long i=0; i<ni; ++i) k0() += Tan(i,refc(i),all);
        ComplexType j0=nda::sum(k0*k0);
        for(long d=0; d<count; ++d) {
          auto ex=iexcit.data()+2*nx*(first+d);
          for(long i=0; i<ni; ++i) occ(i)=refc(i);
          for(long p=0; p<nx; ++p) occ(ex[p])=ex[nx+p];
          auto k=KE(first+d,iw,all);
          x()=ComplexType(0); k()=ComplexType(0);
          for(long p=0; p<nx; ++p) {
            long hp=ex[p];
            auto v=B(d,p,hp,all);
            x() += v*v;
            k() += v;
            for(long q=p+1; q<nx; ++q)
              x() += ComplexType(2)*B(d,p,ex[q],all)*B(d,q,hp,all);
            for(long j=0; j<ni; ++j)
              x() += ComplexType(2)*B(d,p,j,all)*Tan(hp,occ(j),all);
          }
          for(long i=0; i<ni; ++i) {
            int oi=occ(i), ri=refc(i);
            auto ti=Tan(i,oi,all), tr=Tan(i,ri,all);
            k() += ti;
            if(oi!=ri) x() += ti*ti-tr*tr;
            for(long j=i+1; j<ni; ++j) {
              int oj=occ(j), rj=refc(j);
              if(oi!=ri || oj!=rj)
                x() += ComplexType(2)*(Tan(i,oj,all)*Tan(j,oi,all)-Tan(i,rj,all)*Tan(j,ri,all));
            }
          }
          EX(iw) -= ComplexType(.5)*wgt(first+d,iw)*nda::sum(x);
          EJ(iw) += ComplexType(.5)*wgt(first+d,iw)*(nda::sum(k*k)-j0);
        }
      } else {
#if defined(ENABLE_DEVICE)
        using kernels::device::to_view;
        kernels::device::ph_excited_energy_gemm_reduce(refc.data(),
            iexcit.data()+2*nx*first,to_view(Tan),to_view(B),
            to_view(wgt(range(first,first+count),iw)),to_view(EX),to_view(EJ),iw,
            to_view(KE(range(first,first+count),iw,all)));
#endif
      }
    }
  }
}
} // namespace sfqmc::afqmc
