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
// This file includes portions derived from work licensed under the
// University of Illinois/NCSA Open Source License. See the NOTICE file
// and LICENSES/NCSA.txt for details.
////////////////////////////////////////////////////////////////////////////////

#include <cmath>

#include "AFQMC/Wavefunctions/NOMSD_FT.hpp"

#include "AFQMC/SlaterDeterminantOperations/density_matrix.hpp"

namespace sfqmc
{
namespace afqmc
{


template<MEMORY_SPACE MEM, class devPsiT>
void NOMSD_FT<MEM,devPsiT>::runtime_optimization(WalkerSet<MEM>& wset)
{
  const int nw   = wset.size();
  const int nspin = walker_type==COLLINEAR ? 2 : 1;
  const int npol = walker_type==NONCOLLINEAR ? 2 : 1;
  memory::array<MEM,ComplexType,2> G(nw,nspin*npol*NMO*npol*NMO);
  // don't use buffered_array!!!
  HamOp.runtime_optimization(G);
}

/*
 * Calculates the bias potential.
 */
template<MEMORY_SPACE MEM, class devPsiT>
void NOMSD_FT<MEM,devPsiT>::vbias(WalkerSet<MEM>& wset, memory::array_view<MEM,ComplexType,2> v, double dt, int nt)
{
  memory::check_memory_space<MEM>(v);
  auto G_time = timers.G_for_vbias.start();
  int nspin = walker_type==COLLINEAR ? 2 : 1;
  int npol  = walker_type==NONCOLLINEAR ? 2 : 1;
  int nw = wset.size();
  nt = nt < 0? wset.getTauStep() : nt; // if time-slice is not given, use current time-slice
  int nc = nspin*npol*NMO*npol*NMO;
  utils::check(v.shape() == std::array<long,2>{nw,HamOp.number_of_cholesky_vectors()},
               "Shape mismatch");
  memory::buffered_array<MEM,ComplexType,2> G(nw,nc);
  memory::buffered_array<MEM,ComplexType,1> ovlp(nw);
  MixedDensityMatrix(wset, G, ovlp, nt);
  G_time.stop();

  auto vbias_time = timers.vbias.start();
  v() = 0.0;
  HamOp.vbias(G, v, dt);
  vbias_time.stop();
}

template<MEMORY_SPACE MEM, class devPsiT>
void NOMSD_FT<MEM,devPsiT>::Energy(WalkerSet<MEM>& wset, int nt)
{
  auto all = nda::range::all;
  int nw = wset.size();
  memory::buffered_array<MEM,ComplexType,1> ovlp(nw,ComplexType(0.0));
  memory::buffered_array<MEM,ComplexType,2> eloc(nw,3);
  eloc() = ComplexType(0.0);
  Energy(wset, eloc(), ovlp(), nt);
  wset.setProperty(OVLP, ovlp);
  wset.setProperty(E1_, eloc(all, 0));
  wset.setProperty(EXX_, eloc(all, 1));
  wset.setProperty(EJ_, eloc(all, 2));
}

template<MEMORY_SPACE MEM, class devPsiT>
void NOMSD_FT<MEM,devPsiT>::MixedDensityMatrix(WalkerSet<MEM> const& wset, memory::array_view<MEM,ComplexType,2> G, int nt)
{
  int nw = wset.size();
  memory::buffered_array<MEM,ComplexType,1> ovlp(nw,ComplexType(0.0));
  MixedDensityMatrix(wset, G, ovlp, nt);
}

template<MEMORY_SPACE MEM, class devPsiT>
void NOMSD_FT<MEM,devPsiT>::Log_Overlap(WalkerSet<MEM>& wset, int nt)
{
  int nw = wset.size();
  memory::buffered_array<MEM,ComplexType,1> ovlp(nw,ComplexType(0.0));
  Log_Overlap(wset, ovlp, nt);
  wset.setProperty(OVLP, ovlp);
}

/*
 * Calculates the local energy and overlaps of all the walkers in the set and 
 * returns them in the appropriate data structures
 */
template<MEMORY_SPACE MEM, class devPsiT>
void NOMSD_FT<MEM,devPsiT>::Energy(WalkerSet<MEM> const& wset, memory::array_view<MEM,ComplexType,2> E, memory::array_view<MEM,ComplexType,1> Ov, int nt)
{
  auto all = nda::range::all;
  memory::check_memory_space<MEM>(E,Ov);
  const int ndet = ci.size();
  const int nw   = wset.size();
  const int nspin = walker_type==COLLINEAR ? 2 : 1; 
  const int npol = walker_type==NONCOLLINEAR ? 2 : 1;
  nt = nt < 0 ? wset.getTauStep() : nt; // if time-slice is (default nt = -1) not given, use current time-slice 
  utils::check(Ov.size() == nw, "Size mismatch");
  utils::check(E.shape() == std::array<long,2>{nw,3}, "Size mismatch");
  // Log_Overlap accumulates!!!
  Ov() = ComplexType(0.0);
  E() = ComplexType(0.0);

  memory::buffered_array<MEM,ComplexType,2> G(nw,nspin*npol*NMO*npol*NMO); 

  if( ndet == 1) {

    DensityMatrix(wset, OrbMats(0,all,all), G, Ov, nt);
    HamOp.energy(E, G(), 0); 

  } else {

    utils::check(false, "finish multi-determinant Energy implementation for finite-T");
    /*
    memory::buffered_array<MEM,ComplexType,2> ovlp(ndet,nw); 
    memory::buffered_array<MEM,ComplexType,3> eloc(ndet,nw,3); 
    ovlp() = ComplexType(0.0);
    eloc() = ComplexType(0.0);

    for (int id = 0; id < ndet; id++)
    {
      DensityMatrix(wset, OrbMats(id,all,all), G, ovlp(id,all),nt); 
      HamOp.energy(eloc(id,all,all), G(), id);
    }

    // work on host for now
    auto ovlp_h = nda::to_host(ovlp);
    auto eloc_h = nda::to_host(eloc);
    nda::array<ComplexType,2> E_h(nw,3); 
    nda::array<ComplexType,1> sum(3,0.0);
    for(int i=0; i<nw; i++) {
      ComplexType log_m = *std::max_element(ovlp_h(all,i).begin(),ovlp_h(all,i).end());
      sum() = ComplexType(0.0);
      ComplexType deno(0.0);
      for(int d=0; d<ndet; ++d) {
        ComplexType expm = std::exp(ovlp_h(d,i) - log_m);
        sum() += std::conj(ci(d)) * eloc_h(d,i,all) * expm; 
        deno += std::conj(ci(d)) * expm;
      }
      E_h(i,all) = sum() / deno;  
    }
    // copy to device/host
    E() = E_h();
    */
  }
  
}

template<MEMORY_SPACE MEM, class devPsiT>
void NOMSD_FT<MEM,devPsiT>::MixedDensityMatrix(WalkerSet<MEM> const& wset, memory::array_view<MEM,ComplexType,2> G, memory::array_view<MEM,ComplexType,1> Ov, int nt)
{
  auto all = nda::range::all;
  memory::check_memory_space<MEM>(G,Ov);
  const int ndet  = ci.size();
  const int nw    = wset.size();
  const int nspin = walker_type==COLLINEAR ? 2 : 1;
  const int npol  = walker_type==NONCOLLINEAR ? 2 : 1;
  utils::check(Ov.size() == nw, "Size mismatch");
  const int nc    =  nspin*npol*NMO ;
  utils::check(G.shape() == std::array<long,2>{nw,nc*npol*NMO}, "Size mismatch");
  G() = ComplexType(0.0);
  Ov() = ComplexType(0.0);

  if(ndet == 1) {

    DensityMatrix(wset, OrbMats(0,all,all), G, Ov, nt);     
  
  } else {

    utils::check(false, "finish multi-determinant MixedDensityMatrix implementation for finite-T");

    /*
    // use the walker's current log_overlap as reference
    memory::buffered_array<MEM,ComplexType,1> log_m(nw); 
    wset.getProperty(OVLP, log_m);
    memory::buffered_array<MEM,ComplexType,1> Ot(nw); 
    memory::buffered_array<MEM,ComplexType,2> Gt(G.shape()); 

    for(int d=0; d<ndet; ++d) {

      DensityMatrix(wset, OrbMats(d,all,all), Gt, Ot, nt);     
      // G += conj(ci) * exp(Ot-log_m) * Gt
      // Ov += conj(ci) * exp(Ot-log_m)
      // doing in host for now!!!
      
      nda::tensor::add(ComplexType(-1.0),log_m,"w",ComplexType(1.0),Ot,"w");
      nda::tensor::scale(std::conj(ci(d)),Ot,nda::tensor::unary_op::EXP);
      nda::tensor::add(ComplexType(1.0),Ot,"w",ComplexType(1.0),Ov,"w");
      if constexpr (MEM==HOST_MEMORY) {
        for(int w=0; w<nw; ++w) 
          G(w,all) += Gt(w,all) * Ot(w);
      } else {
        // is this doing the righ thing???
        nda::tensor::contract(ComplexType(1.0),Ot,"w",Gt,"wi",ComplexType(1.0),G,"wi");
      }

    }

    if constexpr (MEM==HOST_MEMORY) {
      for(int w=0; w<nw; ++w) 
        G(w,all) /= Ov(w); 
    } else {
      Ot() = Ov();
      nda::tensor::scale(1.0,Ot,nda::tensor::unary_op::RCP);
      // is this doing the righ thing???
      nda::tensor::elementwise(1.0,Ot,"w",1.0,G,"wi",nda::tensor::binary_op::PROD);
    }
    */
  }
}
template<MEMORY_SPACE MEM, class devPsiT>
void NOMSD_FT<MEM,devPsiT>::Log_Overlap(WalkerSet<MEM> const& wset, memory::array_view<MEM,ComplexType,1> Ov, int nt)
{
  memory::check_memory_space<MEM>(Ov);
  const int ndet = ci.size();
  const int nw   = wset.size();
  nt = nt < 0 ? wset.getTauStep() : nt; // if time-slice is not given (default nt = -1), use current time-slice
  utils::check(Ov.size() == nw, "Size mismatch");
  // Log_Overlap accumulates!!!
  Ov() = ComplexType(0.0);

  if(ndet == 1) {

    memory::buffered_array<MEM,ComplexType,1> sclR_up(nw);
    wset.getProperty(LOGSCL_UP, sclR_up(nda::range::all));

    memory::buffered_array<MEM,ComplexType,1> sclL_up(1,sclL(0));

    using T = typename std::decay<decltype(OrbMats(0,0,0))>::type;
    // MixedDensityMatrix does not accept sparse trial wavefunctions
    if constexpr (math::sparse::CSRMatrix<T>){
      auto ULup_dense(math::sparse::to_array<'N'>(OrbMats(0,0,0)()));
      auto DLup_dense(math::sparse::to_array<'N'>(OrbMats(0,0,1)()));
      auto VLup_dense(math::sparse::to_array<'N'>(OrbMats(0,0,2)()));

      det_ops::Log_Overlap(ULup_dense,DLup_dense(nt,nda::range::all),VLup_dense,
                        wset.UMatrices(Alpha),wset.DMatrices(Alpha),wset.VMatrices(Alpha),
                        sclL_up, sclR_up, Ov);
    //if(walker_type == CLOSED) {
    //  nda::tensor::scale(ComplexType(2.0),Ov);
      if (walker_type == COLLINEAR)  {
        memory::buffered_array<MEM,ComplexType,1> sclR_dn(nw);
        wset.getProperty(LOGSCL_DN, sclR_dn(nda::range::all));

        memory::buffered_array<MEM,ComplexType,1> sclL_dn(1,sclL(1));

        auto ULdn_dense(math::sparse::to_array<'N'>(OrbMats(0,1,0)()));
        auto DLdn_dense(math::sparse::to_array<'N'>(OrbMats(0,1,1)()));
        auto VLdn_dense(math::sparse::to_array<'N'>(OrbMats(0,1,2)()));

        det_ops::Log_Overlap(ULdn_dense,DLdn_dense(nt,nda::range::all),VLdn_dense,
                wset.UMatrices(Beta),wset.DMatrices(Beta),wset.VMatrices(Beta),
                sclL_dn, sclR_dn, Ov);
      }
    }
    else{
      
      // FIX: how to pass a view of OrbMats(0,0,1), instead of copying to DLup?
      auto ULup(OrbMats(0,0,0)());
      auto DLup(OrbMats(0,0,1)());
      auto VLup(OrbMats(0,0,2)());

      det_ops::Log_Overlap(ULup,DLup(nt,nda::range::all),VLup,
                        wset.UMatrices(Alpha),wset.DMatrices(Alpha),wset.VMatrices(Alpha),
                        sclL_up, sclR_up, Ov);
      if(walker_type == CLOSED) utils::check(false, "Closed walkers not implemented for finite-T");

      if (walker_type == COLLINEAR)  {
        memory::buffered_array<MEM,ComplexType,1> sclR_dn(nw);
        wset.getProperty(LOGSCL_DN, sclR_dn(nda::range::all));

        memory::buffered_array<MEM,ComplexType,1> sclL_dn(1,sclL(1));

        // FIX: how to pass a view of OrbMats(0,0,1), instead of copying to DLdn?
        auto ULdn(OrbMats(0,1,0)());
        auto DLdn(OrbMats(0,1,1)());
        auto VLdn(OrbMats(0,1,2)());

        det_ops::Log_Overlap(ULdn,DLdn(nt,nda::range::all),VLdn,
                wset.UMatrices(Beta),wset.DMatrices(Beta),wset.VMatrices(Beta),
                sclL_dn, sclR_dn, Ov);
      }

    }

  } else {
    utils::check(false, "multi-determinant Log_Overlap is not implemented for finite-T");
    /*
    // compute separate, assemble afterwards
    memory::buffered_array<MEM,ComplexType,2> log_ovlps(ndet,nw);
    log_ovlps() = ComplexType(0.0);
    for(int d=0; d<ndet; ++d) {
      det_ops::Log_Overlap(OrbMats(d,0)(),wset.template SlaterMatrices<MEM>(Alpha),log_ovlps(d,all));
      if(walker_type == CLOSED) {
        nda::tensor::scale(ComplexType(2.0),log_ovlps(d,all));
      } else if (walker_type == COLLINEAR)  {
        det_ops::Log_Overlap(OrbMats(d,1)(),wset.template SlaterMatrices<MEM>(Beta),log_ovlps(d,all));
      }
      ComplexType val( std::log( (std::abs(ci(d)) < 1e-12 ? ComplexType(1e-12) : std::conj(ci(d)) ) ) );
      memory::buffered_array<MEM,ComplexType,1> buff(nw,val);
      // log_ovlp(d,w) = log( ci(d) ) + log( Ov(d, Beta) ) + log( Ov(d, Alpha) )
      nda::tensor::add(buff,log_ovlps(d,all));
    }
  
    // now assemble avoiding round-off problems 
    // Ov(w) = sum_d log_ovlp(d,w) 
    Ov() = ComplexType(0.0);
    math::log_of_sum(0,log_ovlps,Ov);
    */ 
  }
}
/*
 * Full green function of the trial wave function.
 */
template<MEMORY_SPACE MEM, class devPsiT>
memory::const_shared_array<HOST_MEMORY,ComplexType,3> NOMSD_FT<MEM,devPsiT>::G_MF()
{
  using nda::range;
  auto all = range::all;

  int nspin = walker_type==COLLINEAR ? 2 : 1;
  int npol  = walker_type==NONCOLLINEAR ? 2 : 1;
  int ndet  = ci.size();

  if (ndet == 1)
  {
    return memory::share_from_root(*mpi, [&] {
      nda::array<ComplexType,3> gMF(nspin,npol*NMO,npol*NMO);
      memory::buffered_array<MEM,ComplexType,1> Ov(1);
      gMF() = ComplexType(0.0);

      //memory::buffered_array<MEM,ComplexType,3> PsiT(1,npol*NMO,nup);
      //PsiT(0,all,all) = nda::dagger(math::sparse::to_array<ComplexType>(OrbMats(0,0)()));

      memory::buffered_array<MEM,ComplexType,1> sclR(1,ComplexType(0.0));
      memory::buffered_array<MEM,ComplexType,1> sclL0_up(1,sclL(0));

      auto ULup = math::sparse::to_array<'N'>(OrbMats(0,0,0)());
      auto ULdn = math::sparse::to_array<'N'>(OrbMats(0,1,0)());

      auto DLup = math::sparse::to_array<'N'>(OrbMats(0,0,1)());
      auto DLdn = math::sparse::to_array<'N'>(OrbMats(0,1,1)());

      auto VLup = math::sparse::to_array<'N'>(OrbMats(0,0,2)());
      auto VLdn = math::sparse::to_array<'N'>(OrbMats(0,1,2)());

      memory::buffered_array<MEM,ComplexType,3> IMat3D(1,npol*NMO,npol*NMO);
      IMat3D(0,all,all) = nda::eye<ComplexType>(npol*NMO);
      memory::buffered_array<MEM,ComplexType,2> IVec2D(1,npol*NMO);
      IVec2D(0,all) = ComplexType(1.0);

      if constexpr (MEM==HOST_MEMORY){
        auto G3d = nda::reshape(gMF,std::array<long,3>{1,nspin*npol*NMO,npol*NMO});
        det_ops::MixedDensityMatrix(ULup, DLup(0,all), VLup, IMat3D, IVec2D, 
            IMat3D, G3d(all,range(npol*NMO),all), Ov, sclL0_up, sclR, true, true);
        if( walker_type == COLLINEAR ) {
          memory::buffered_array<MEM,ComplexType,1> sclL0_dn(1,sclL(1));
          det_ops::MixedDensityMatrix(ULdn, DLdn(0,all), VLdn, IMat3D, IVec2D, 
               IMat3D, G3d(all,nda::range(npol*NMO,nspin*npol*NMO),all), Ov, sclL0_dn, sclR, true, true);
        }
      } else {
        memory::array<MEM,ComplexType,3> gt(1,npol*NMO,npol*NMO);
        det_ops::MixedDensityMatrix(ULup, DLup(0,all), VLup, IMat3D, IVec2D, 
                IMat3D, gt, Ov, sclL0_up, sclR, true, true);
        gMF(0,all,all) = gt(0,all,all);
        if( walker_type == COLLINEAR ) { 
          memory::buffered_array<MEM,ComplexType,1> sclL0_dn(1,sclL(1));
          det_ops::MixedDensityMatrix(ULdn, DLdn(0,all), VLdn, IMat3D, IVec2D, 
                IMat3D, gt, Ov, sclL0_dn, sclR, true, true);
          gMF(1,all,all) = gt(0,all,all);
        }
      }

      return gMF;
    });

  }
  else {

    utils::check(false, "finish multi-determinant G_MF implementation for finite-T");
    /*
    RealType Osum(0.0);
    memory::buffered_array<MEM,ComplexType,2> Gsum(1,nspin*npol*NMO*npol*NMO);
    auto[n0, n1] = itertools::chunk_range(0, ndet*(ndet+1)/2, mpi->comm.size(), mpi->comm.rank());
    int last_p = -1;
    bool found = false;
    Gsum() = ComplexType(0.0);

    { // control buffer lifetime
      memory::buffered_array<MEM,ComplexType,1> Ov(1);
      memory::buffered_array<MEM,ComplexType,2> G(1,nspin*npol*NMO*npol*NMO);
      auto G3d = nda::reshape(G,std::array<long,3>{1,nspin*npol*NMO,npol*NMO});
      memory::buffered_array<MEM,ComplexType,3> PsiT(1,npol*NMO,nup);
      memory::buffered_array<MEM,ComplexType,3> PsiTB;
      if( walker_type == COLLINEAR ) 
        PsiTB.resize(1,npol*NMO,ndown);
      auto Ov_h = nda::to_host(Ov);

      for(int p=0, pq=0; p<ndet; ++p) {
        for(int q=p; q<ndet; ++q, ++pq) {

          if( pq < n0 ) continue;
          if( pq >= n1 ) break; 
          if( last_p != p ) {
            last_p = p;
            PsiT(0,all,all) = nda::dagger(math::sparse::to_array<ComplexType>(OrbMats(p,0)()));
            if( walker_type == COLLINEAR ) 
              PsiTB(0,all,all) = nda::dagger(math::sparse::to_array<ComplexType>(OrbMats(p,1)()));
          }

          Ov() = ComplexType(0.0);
          G() = ComplexType(0.0);
          det_ops::MixedDensityMatrix(OrbMats(q,0)(),PsiT,G3d(all,nda::range(nup),all),Ov,false);
          if( walker_type == COLLINEAR ) 
            det_ops::MixedDensityMatrix(OrbMats(q,1)(),PsiTB,G3d(all,nda::range(nup,nel),all),Ov,false);
          Ov_h() = nda::to_host(Ov);
          if( std::abs(Ov_h(0)) == ComplexType(0.0) and not found ) {
            found = true;
            app_warning(" WARNING: Found orthogonal determinants in trial wave function of NOMSD.");
            app_warning("          The mean-field substraction potential is potentially wrong. !");
          }
          RealType scl = std::real(std::conj(ci(q)) * ci(p) * std::exp(Ov_h(0))); 
          Osum += scl; 
          nda::tensor::add(ComplexType(scl),G,"wi",ComplexType(1.0),Gsum,"wi");
        }
      }
    }

    mpi->all_reduce(Gsum,std::plus<>{});
    mpi->comm.all_reduce_in_place_n(&Osum,1,std::plus<>{});
    nda::tensor::scale(ComplexType(1.0/Osum),Gsum); 
    auto Gv = nda::reshape(Gsum,std::array<long,3>{nspin,npol*NMO,npol*NMO});
    if(mpi->node_comm.root()) gMF() = Gv();
    mpi->comm.barrier();
    */
    return memory::host_const_shared_array<ComplexType,3>{};
  }
}
/*
 * Calculate mean field expectation value of Cholesky potentials
 */
template<MEMORY_SPACE MEM, class devPsiT>
void NOMSD_FT<MEM,devPsiT>::vMF(memory::array_view<MEM,ComplexType,1> v, double dt)
{
  using nda::range;
  auto all = range::all;
  utils::check(v.size() == number_of_cholesky_vectors(), "Size mismatch");
  utils::check(v.strides()[0] == 1, "Strides mismatch");
  auto v2d = nda::reshape(v,std::array<long,2>{1,v.size()});   
  v() = ComplexType(0.0);

  int nspin = walker_type==COLLINEAR ? 2 : 1;
  int npol  = walker_type==NONCOLLINEAR ? 2 : 1;
  int ndet  = ci.size();

  if (ndet == 1)
  {
    memory::buffered_array<MEM,ComplexType,1> Ov(1); 
    memory::buffered_array<MEM,ComplexType,2> G(1,nspin*npol*NMO*npol*NMO); 
    auto G3d = nda::reshape(G,std::array<long,3>{1,nspin*npol*NMO,npol*NMO});


    memory::buffered_array<MEM,ComplexType,1> sclR(1,ComplexType(0.0));
    memory::buffered_array<MEM,ComplexType,1> sclL0_up(1,sclL(0));

    auto ULup = math::sparse::to_array<'N'>(OrbMats(0,0,0)());
    auto ULdn = math::sparse::to_array<'N'>(OrbMats(0,1,0)());

    auto DLup = math::sparse::to_array<'N'>(OrbMats(0,0,1)());
    auto DLdn = math::sparse::to_array<'N'>(OrbMats(0,1,1)());

    auto VLup = math::sparse::to_array<'N'>(OrbMats(0,0,2)());
    auto VLdn = math::sparse::to_array<'N'>(OrbMats(0,1,2)());

    memory::buffered_array<MEM,ComplexType,3> IMat3D(1,npol*NMO,npol*NMO);
    IMat3D(0,all,all) = nda::eye<ComplexType>(npol*NMO);
    memory::buffered_array<MEM,ComplexType,2> IVec2D(1,npol*NMO);
    IVec2D(0,all) = ComplexType(1.0);
    
    det_ops::MixedDensityMatrix(ULup, DLup(0,all), VLup, IMat3D, IVec2D, 
            IMat3D, G3d(all,range(npol*NMO),all), Ov, sclL0_up, sclR, true, true);
    
    if( walker_type == COLLINEAR ) {
        memory::buffered_array<MEM,ComplexType,1> sclL0_dn(1,sclL(1)); 
        det_ops::MixedDensityMatrix(ULdn, DLdn(0,all), VLdn, IMat3D, IVec2D, 
               IMat3D, G3d(all,nda::range(npol*NMO,nspin*npol*NMO),all), Ov, sclL0_dn, sclR, true, true);
    }

    HamOp.vbias(G, v2d, dt);

  } else {

    utils::check(false, "finish multi-determinant vMF implementation for finite-T");
    /*
    RealType Osum(0.0);
    memory::buffered_array<MEM,ComplexType,2> Gsum(1,nspin*npol*NMO*npol*NMO);
    auto[n0, n1] = itertools::chunk_range(0, ndet*(ndet+1)/2, mpi->comm.size(), mpi->comm.rank());
    int last_p = -1;
    bool found = false;
    Gsum() = ComplexType(0.0);

    { // control buffer lifetime
      memory::buffered_array<MEM,ComplexType,1> Ov(1);
      memory::buffered_array<MEM,ComplexType,2> G(1,nspin*npol*NMO*npol*NMO);
      auto G3d = nda::reshape(G,std::array<long,3>{1,nspin*npol*NMO,npol*NMO});
      memory::buffered_array<MEM,ComplexType,3> PsiT(1,npol*NMO,nup);
      memory::buffered_array<MEM,ComplexType,3> PsiTB;
      if( walker_type == COLLINEAR ) 
        PsiTB.resize(1,npol*NMO,ndown);
      auto Ov_h = nda::to_host(Ov);

      for(int p=0, pq=0; p<ndet; ++p) {
        for(int q=p; q<ndet; ++q, ++pq) {

          if( pq < n0 ) continue;
          if( pq >= n1 ) break; 
          if( last_p != p ) {
            last_p = p;
            PsiT(0,all,all) = nda::dagger(math::sparse::to_array<ComplexType>(OrbMats(p,0)()));
            if( walker_type == COLLINEAR ) 
              PsiTB(0,all,all) = nda::dagger(math::sparse::to_array<ComplexType>(OrbMats(p,1)()));
          }

          Ov() = ComplexType(0.0);
          G() = ComplexType(0.0);
          det_ops::MixedDensityMatrix(OrbMats(q,0)(),PsiT,G3d(all,nda::range(nup),all),Ov,false);
          if( walker_type == COLLINEAR ) 
            det_ops::MixedDensityMatrix(OrbMats(q,1)(),PsiTB,G3d(all,nda::range(nup,nel),all),Ov,false);
          Ov_h() = nda::to_host(Ov);
          if( std::abs(Ov_h(0)) == ComplexType(0.0) and not found ) {
            found = true;
            app_warning(" WARNING: Found orthogonal determinants in trial wave function of NOMSD.");
            app_warning("          The mean-field substraction potential is potentially wrong. !");
          }
          RealType scl = std::real(std::conj(ci(q)) * ci(p) * std::exp(Ov_h(0))); 
          Osum += scl; 
          nda::tensor::add(ComplexType(scl),G,"wi",ComplexType(1.0),Gsum,"wi");
        }
      }
    }

    mpi->all_reduce(Gsum,std::plus<>{});
    mpi->comm.all_reduce_in_place_n(&Osum,1,std::plus<>{});
    nda::tensor::scale(ComplexType(1.0/Osum),Gsum); 
    HamOp.vbias(Gsum, v2d, dt);
    */
  }
}
/*
template<MEMORY_SPACE MEM, class devPsiT>
template<class WlkSet, nda::MemoryArrayOfRank<1> TVec>
void NOMSD_FT<MEM,devPsiT>::Log_Overlap(const WlkSet& wset, TVec&& Ov)
{
    utils::check(false,"Log_Overlap not implemented for finite-T");
}
*/

/*
template<MEMORY_SPACE MEM, class devPsiT>
inline void NOMSD<MEM,devPsiT>::recompute_ci()
{
  if (walker_type == NONCOLLINEAR)
    APP_ABORT(" Error: NOMSD::recompute_ci not implemented with NONCOLLINEAR.");
  double LogOverlapFactor(0.0);
  int ndets = ci.size();
  if (ndets == 1)
  {
    ci[0] = ComplexType(1.0, 0.0);
    return;
  }

  if (TG.getNGroupsPerTG() > 1)
  {
    APP_ABORT(" Error: rediag not implemented with distributed wavefunction.");
  }
  else
  {
    size_t nt = (1 + dm_size(false) + NMO * nup);
    StaticSHMVector shmbuff(iextensions<1u>{nt}, ComplexType(0.0),
                            shm_buffer_manager.get_generator().template get_allocator<ComplexType>());

    boost::multi::array<ComplexType, 2, shared_allocator<ComplexType>> H({ndets, ndets}, TG.Node());
    boost::multi::array<ComplexType, 2, shared_allocator<ComplexType>> S({ndets, ndets}, TG.Node());
    boost::multi::array<ComplexType, 1> energy(iextensions<1u>{2});
    CMatrix Psia({0, 0});
    CMatrix Psib({0, 0});
    if (TG.TG_local().root())
    {
      // only TG_local().root() calculates G for now
      Psia = CMatrix({NMO, nup});
      if (walker_type == COLLINEAR)
        Psib = CMatrix({NMO, ndown});
    }
    using std::fill_n;
    fill_n(H.origin(), H.num_elements(), ComplexType(0.0));
    fill_n(S.origin(), S.num_elements(), ComplexType(0.0));
    fill_n(energy.origin(), energy.num_elements(), ComplexType(0.0));

    //ComplexType zero(0.0);
    auto Gsize = dm_size(false);
    int nr = Gsize, nc = 1;
    if (transposed_G_for_E_)
      std::swap(nr, nc);
    CVector_ref ov_(make_device_ptr(shmbuff.origin()), iextensions<1u>{1});
    CMatrix_ref G(ov_.origin() + 1, {nr, nc});
    StaticMatrix eloc2({1, 3},  ComplexType(0.0),
		       buffer_manager.get_generator().template get_allocator<ComplexType>());

    for (int jdet = 0, ji = 0; jdet < ndets; jdet++)
    {
      ComplexType cjdet = ci[jdet];
      for (int idet = jdet; idet < ndets; idet++, ji++)
      {
        if (ji % TG.getNumberOfTGs() == TG.getTGNumber())
        {
          ComplexType cidet = ci[idet];
          if (TG.TG_local().root())
          {
            CMatrix_ref Ga(G.origin(), {nup, NMO});
            ma::Matrix2MA('H', OrbMats[nspins * jdet], Psia);
            ov_[0] = SDetOp.MixedDensityMatrix(OrbMats[nspins * idet], Psia, Ga, LogOverlapFactor, true);
            if (walker_type == COLLINEAR)
            {
              CMatrix_ref Gb(Ga.origin() + Ga.num_elements(), {ndown, NMO});
              ma::Matrix2MA('H', OrbMats[nspins * jdet + 1], Psib);
              ov_[0] *= SDetOp.MixedDensityMatrix(OrbMats[nspins * idet + 1], Psib, Gb, LogOverlapFactor, true);
            }
            else if (walker_type == CLOSED)
            {
              ov_[0] *= ComplexType(ov_[0]);
            }
          }
          TG.TG_local().barrier();
          HamOp.energy(eloc2, G, idet, TG.TG_local().root());
          if (TG.TG_local().size() > 1)
            TG.TG_local().all_reduce_in_place_n(raw_pointer_cast(eloc2.origin()), 3, std::plus<>());
          if (TG.TG_local().root())
          {
            if (idet != jdet)
            {
              //ComplexType ovij = static_cast<ComplexType>(ov_[0]);
              H[idet][jdet] = ComplexType(ov_[0]) * (ComplexType(eloc2[0][0]) + ComplexType(eloc2[0][1]) + ComplexType(eloc2[0][2]));
              S[idet][jdet] = ComplexType(ov_[0]);
              S[idet][jdet] = ComplexType(ov_[0]);
              H[jdet][idet] = ma::conj(ComplexType(H[idet][jdet]));
              S[jdet][idet] = ma::conj(ComplexType(S[idet][jdet]));
              energy[0] += ma::conj(cidet) * cjdet * ComplexType(H[idet][jdet]) 
			+ ma::conj(cjdet) * cidet * ComplexType(H[jdet][idet]);
              energy[1] += ma::conj(cidet) * cjdet * ComplexType(S[idet][jdet]) 
			+ ma::conj(cjdet) * cidet * ComplexType(S[jdet][idet]);
            }
            else
            {
              H[idet][idet] =
                  ComplexType(real(ComplexType(ov_[0]) * (ComplexType(eloc2[0][0]) + ComplexType(eloc2[0][1]) + ComplexType(eloc2[0][2]))), 
                              0.0);
              S[idet][idet] = ComplexType(real(ComplexType(ov_[0])), 0.0);
              energy[0] += ma::conj(cidet) * cjdet * ComplexType(H[idet][jdet]);
              energy[1] += ma::conj(cidet) * cjdet * ComplexType(S[idet][jdet]);
            }
          }
          TG.TG_local().barrier();
        }
      }
    }
    TG.Global().barrier();
    if (TG.Node().root())
      TG.Cores().all_reduce_in_place_n(raw_pointer_cast(H.origin()), H.num_elements(), std::plus<>());
    if (TG.Node().root())
      TG.Cores().all_reduce_in_place_n(raw_pointer_cast(S.origin()), S.num_elements(), std::plus<>());
    TG.Global().all_reduce_in_place_n(energy.origin(), 2, std::plus<>());

    app_log(1," - Variational energy of trial wavefunction: {}",  energy[0] / energy[1]);
    app_log(1," - Diagonalizing CI matrix.");
    using RVector = boost::multi::array<RealType, 1>;
    #if defined(ENABLE_DEVICE)
    using CMatrix = boost::multi::array<ComplexType, 2>;
    #endif
    // Want a "unique" solution for all cores/nodes.
    if (TG.Global().rank() == 0)
    {
      boost::multi::array_ref<ComplexType, 2> H_(raw_pointer_cast(H.origin()), {ndets, ndets});
      boost::multi::array_ref<ComplexType, 2> S_(raw_pointer_cast(S.origin()), {ndets, ndets});
      std::pair<RVector, CMatrix> Sol = ma::genEigSelect<RVector, CMatrix>(H_, S_, 1);
      app_log(1," - Updating CI coefficients. ");
      app_log(1," - Recomputed coefficient of first determinant: {}", Sol.second[0][0]); 
      for (int idet = 0; idet < ndets; idet++)
      {
        ComplexType ci_ = Sol.second[0][idet];
        // Do we want this much output?
        app_log(1, "{} old: {}, new: {} ",idet,ci[idet],ci_);
        ci[idet] = ci_;
      }
      app_log(1," - Recomputed variational energy of trial wavefunction: {}",Sol.first[0]);
    }
    if (TG.Global().size() > 1)
      TG.Global().broadcast_n(raw_pointer_cast(ci.data()), ci.size(), 0);
  }
}
*/

template class NOMSD_FT<HOST_MEMORY, PsiT_Matrix<HOST_MEMORY>>;
template class NOMSD_FT<HOST_MEMORY, memory::const_shared_array<HOST_MEMORY,ComplexType,2>>;
#if defined(ENABLE_DEVICE)
template class NOMSD_FT<DEVICE_MEMORY, PsiT_Matrix<DEVICE_MEMORY>>;
template class NOMSD_FT<DEVICE_MEMORY, memory::const_shared_array<DEVICE_MEMORY,ComplexType,2>>;
#endif

} // namespace afqmc

} // namespace sfqmc
