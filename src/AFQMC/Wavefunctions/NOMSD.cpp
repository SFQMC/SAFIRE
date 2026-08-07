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

#include "AFQMC/Wavefunctions/NOMSD.hpp"

#include "AFQMC/SlaterDeterminantOperations/density_matrix.hpp"

namespace sfqmc {
namespace afqmc {

template<MEMORY_SPACE MEM, class devPsiT>
void NOMSD<MEM,devPsiT>::runtime_optimization(WalkerSet<MEM>& wset) {
  const int nw   = wset.size();
  const int nel = (walker_type==COLLINEAR ? nup+ndown : nup );
  const int npol = (walker_type==NONCOLLINEAR ? 2 : 1 );
  memory::array<MEM,ComplexType,2> G(nw,nel*npol*NMO);
  // don't use buffered_array!!!
  HamOp.runtime_optimization(G);
}

/*
 * Calculates the bias potential.
 */
template<MEMORY_SPACE MEM, class devPsiT>
void NOMSD<MEM,devPsiT>::vbias(WalkerSet<MEM>& wset, memory::array_view<MEM,ComplexType,2> v,
                               double dt, [[maybe_unused]] int nt) {
  memory::check_memory_space<MEM>(v);
  AFQMCTimer.start(G_for_vbias_timer);
  bool compact_G_for_vbias = (ci.size() == 1);
  int nel   = (walker_type==COLLINEAR ? nup+ndown : nup);
  int nspin = (walker_type==COLLINEAR ? 2 : 1);
  int npol  = (walker_type==NONCOLLINEAR ? 2 : 1);
  int nw = wset.size();
  int nc = (compact_G_for_vbias ? nel*npol*NMO : nspin*npol*NMO*npol*NMO );
  utils::check(v.shape() == std::array<long,2>{nw,HamOp.number_of_cholesky_vectors()},
               "Shape mismatch");
  memory::buffered_array<MEM,ComplexType,2> G(nw,nc);
  memory::buffered_array<MEM,ComplexType,1> ovlp(nw);
  MixedDensityMatrix(wset, G, ovlp, compact_G_for_vbias);
  AFQMCTimer.stop(G_for_vbias_timer);
  AFQMCTimer.start(vbias_timer);
  v() = ComplexType(0.0);
  HamOp.vbias(G, v, dt);
  AFQMCTimer.stop(vbias_timer);
}

/*
 * Calculates the local energy and overlaps of all the walkers in the set and stores
 * them in the wset data
 */
template<MEMORY_SPACE MEM, class devPsiT>
void NOMSD<MEM,devPsiT>::Energy(WalkerSet<MEM>& wset, [[maybe_unused]] int nt) {
  auto all = nda::range::all;
  int nw = wset.size();
  memory::buffered_array<MEM,ComplexType,1> ovlp(nw,ComplexType(0.0));
  memory::buffered_array<MEM,ComplexType,2> eloc(nw,3);
  eloc() = ComplexType(0.0);
  Energy(wset, eloc(), ovlp());
  wset.setProperty(OVLP, ovlp);
  wset.setProperty(E1_, eloc(all, 0));
  wset.setProperty(EXX_, eloc(all, 1));
  wset.setProperty(EJ_, eloc(all, 2));
}

template<MEMORY_SPACE MEM, class devPsiT>
void NOMSD<MEM,devPsiT>::MixedDensityMatrix(WalkerSet<MEM> const& wset,
                                            memory::array_view<MEM,ComplexType,2> G, bool compact) {
  int nw = wset.size();
  memory::buffered_array<MEM,ComplexType,1> ovlp(nw,ComplexType(0.0));
  MixedDensityMatrix(wset, G, ovlp, compact);
}

/*
 * Calculates the overlaps of all walkers in the set. Updates values in wset.
 */
template<MEMORY_SPACE MEM, class devPsiT>
void NOMSD<MEM,devPsiT>::Log_Overlap(WalkerSet<MEM>& wset) {
  int nw = wset.size();
  memory::buffered_array<MEM,ComplexType,1> ovlp(nw,ComplexType(0.0));
  Log_Overlap(wset, ovlp);
  wset.setProperty(OVLP, ovlp);
}

/*
 * Returns the reference Slater Matrices needed for back propagation.
 */
template<MEMORY_SPACE MEM, class devPsiT>
void NOMSD<MEM,devPsiT>::getReferences(memory::buffered_array<MEM,ComplexType,3>& Refs) const {
  using nda::range;
  auto all = range::all;
  memory::check_memory_space<MEM>(Refs);
  int number_of_references = OrbMats.extent(0);

  int nel = nup + (walker_type == COLLINEAR ? ndown : 0);
  int npol = (walker_type == NONCOLLINEAR ? 2 : 1);
  Refs.resize(number_of_references, npol*NMO, nel);
  if constexpr (math::sparse::CSRMatrix<devPsiT>) {
    for(int i=0; i<number_of_references; ++i) {
      Refs(i,all,range(nup)) = math::sparse::to_array<'H'>(OrbMats(i,0)());
      if(walker_type == COLLINEAR)
        Refs(i,all,range(nup,nel)) = math::sparse::to_array<'H'>(OrbMats(i,1)());
    }
  } else {
    for(int i=0; i<number_of_references; ++i) {
      if(nup != 0) {
        nda::tensor::add(nda::conj(OrbMats(i,0)()),"ji",Refs(i,all,range(nup)),"ij");
      }
      if(walker_type == COLLINEAR && ndown != 0) {
        nda::tensor::add(nda::conj(OrbMats(i,1)()),"ji",Refs(i,all,range(nup,nel)),"ij");
      }
    }
  }
}

/*
 * Calculates the local energy and overlaps of all the walkers in the set and 
 * returns them in the appropriate data structures
 */
template<MEMORY_SPACE MEM, class devPsiT>
void NOMSD<MEM,devPsiT>::Energy(WalkerSet<MEM> const& wset,
                                memory::array_view<MEM,ComplexType,2> E,
                                memory::array_view<MEM,ComplexType,1> Ov, int nt) {
  auto all = nda::range::all;
  memory::check_memory_space<MEM>(E,Ov);
  const int ndet = ci.size();
  const int nw   = wset.size();
  const int nel = (walker_type==COLLINEAR ? nup+ndown : nup ); 
  const int npol = (walker_type==NONCOLLINEAR ? 2 : 1 ); 
  utils::check(Ov.size() == nw, "Size mismatch");
  utils::check(E.shape() == std::array<long,2>{nw,3}, "Size mismatch");
  // Log_Overlap accumulates!!!
  Ov() = ComplexType(0.0);
  E() = ComplexType(0.0);

  memory::buffered_array<MEM,ComplexType,2> G(nw,nel*npol*NMO); 

  if(ndet == 1) {
    DensityMatrix(wset, OrbMats(0,all), G, Ov);
    HamOp.energy(E, G(), 0);
    ComplexType val = std::log(std::abs(ci(0)) < 1e-12 ? ComplexType(1e-12) : std::conj(ci(0)));
    memory::buffered_array<MEM,ComplexType,1> buff(nw,val);
    nda::tensor::add(1.0, buff,"w", 1.0, Ov,"w");

  } else {
    memory::buffered_array<MEM,ComplexType,2> ovlp(ndet,nw); 
    memory::buffered_array<MEM,ComplexType,3> eloc(ndet,nw,3); 
    ovlp() = ComplexType(0.0);
    eloc() = ComplexType(0.0);

    for (int id = 0; id < ndet; id++) {
      DensityMatrix(wset, OrbMats(id,all), G, ovlp(id,all)); 
      HamOp.energy(eloc(id,all,all), G(), id);
    }

    // work on host for now
    auto ovlp_h = nda::to_host(ovlp);
    auto eloc_h = nda::to_host(eloc);
    nda::array<ComplexType,2> E_h(nw,3); 
    nda::array<ComplexType,1> Ov_h(nw,ComplexType(0.0)); 
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
      Ov_h(i) = std::log(deno) + log_m;
    }
    // copy to device/host
    E() = E_h();
    Ov() = Ov_h();
  }
}

template<MEMORY_SPACE MEM, class devPsiT>
void NOMSD<MEM,devPsiT>::MixedDensityMatrix(WalkerSet<MEM> const& wset,
                                            memory::array_view<MEM,ComplexType,2> G,
                                            memory::array_view<MEM,ComplexType,1> Ov, bool compact) {
  auto all = nda::range::all;
  memory::check_memory_space<MEM>(G,Ov);
  const int ndet  = ci.size();
  const int nw    = wset.size();
  const int nel   = (walker_type==COLLINEAR ? nup+ndown : nup );
  const int nspin = (walker_type==COLLINEAR ? 2 : 1 );
  const int npol  = (walker_type==NONCOLLINEAR ? 2 : 1 );
  utils::check(Ov.size() == nw, "Size mismatch");
  const int nc    = ( compact ? nel : nspin*npol*NMO );
  auto expected_shape = std::to_array<long>({nw, nc * npol * NMO});
  utils::check(G.shape() == expected_shape, "Size mismatch: given {} != {} expected", G.shape(), expected_shape);
  G() = ComplexType(0.0);
  Ov() = ComplexType(0.0);

  if(ndet == 1) {
    DensityMatrix(wset, OrbMats(0,all), G, Ov, compact);     
  } else {
    // use the walker's current log_overlap as reference
    memory::buffered_array<MEM,ComplexType,1> log_m(nw); 
    wset.getProperty(OVLP, log_m);
    memory::buffered_array<MEM,ComplexType,1> Ot(nw); 
    memory::buffered_array<MEM,ComplexType,2> Gt(G.shape()); 

    for(int d=0; d<ndet; ++d) {

      DensityMatrix(wset, OrbMats(d,all), Gt, Ot, compact);     
      // G += conj(ci) * exp(Ot-log_m) * Gt
      // Ov += conj(ci) * exp(Ot-log_m)
      // doing in host for now!!!
      
      nda::tensor::add(ComplexType(-1.0),log_m,"w",ComplexType(1.0),Ot,"w");
      nda::apply(std::conj(ci(d)),Ot,nda::tensor::op::EXP);
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
      nda::apply(ComplexType(1.0),Ot,nda::tensor::op::RCP);
      // is this doing the righ thing???
      nda::tensor::elementwise(ComplexType(1.0),Ot,"w",ComplexType(1.0),G,"wi",nda::tensor::op::MUL);
    }
    // Ov(iw) += log_ov(iw)
    nda::tensor::add(ComplexType(1.0),log_m,"w",ComplexType(1.0),Ov,"w");
 
  }
}
template<MEMORY_SPACE MEM, class devPsiT>
void NOMSD<MEM,devPsiT>::Log_Overlap(WalkerSet<MEM> const& wset,
                                     memory::array_view<MEM,ComplexType,1> Ov, int nt) {
  auto all = nda::range::all;
  memory::check_memory_space<MEM>(Ov);
  const int ndet = ci.size();
  const int nw   = wset.size();
  utils::check(Ov.size() == nw, "Size mismatch");
  // Log_Overlap accumulates!!!
  Ov() = ComplexType(0.0);

  if(ndet == 1) {
    det_ops::Log_Overlap(OrbMats(0,0)(),wset.SlaterMatrices(Alpha),Ov);
    if(walker_type == CLOSED) {
      nda::tensor::scale(ComplexType(2.0),Ov);
    } else if (walker_type == COLLINEAR)  {
      det_ops::Log_Overlap(OrbMats(0,1)(),wset.SlaterMatrices(Beta),Ov);
    }
    ComplexType val = std::log(std::abs(ci(0)) < 1e-12 ? 1e-12 : std::conj(ci(0)));
    memory::buffered_array<MEM,ComplexType,1> buff(nw,val);
    nda::tensor::add(1.0, buff,"w", 1.0, Ov,"w");
  } else {
    // compute separate, assemble afterwards
    memory::buffered_array<MEM,ComplexType,2> log_ovlps(ndet,nw);
    log_ovlps() = ComplexType(0.0);
    for(int d=0; d<ndet; ++d) {
      det_ops::Log_Overlap(OrbMats(d,0)(),wset.SlaterMatrices(Alpha),log_ovlps(d,all));
      if(walker_type == CLOSED) {
        nda::tensor::scale(ComplexType(2.0),log_ovlps(d,all));
      } else if (walker_type == COLLINEAR)  {
        det_ops::Log_Overlap(OrbMats(d,1)(),wset.SlaterMatrices(Beta),log_ovlps(d,all));
      }
      ComplexType val( std::log( (std::abs(ci(d)) < 1e-12 ? ComplexType(1e-12) : std::conj(ci(d)) ) ) );
      memory::buffered_array<MEM,ComplexType,1> buff(nw,val);
      // log_ovlp(d,w) = log( ci(d) ) + log( Ov(d, Beta) ) + log( Ov(d, Alpha) )
      nda::tensor::add(1.0, buff,"w", 1.0, log_ovlps(d,all),"w");
    }
  
    // now assemble avoiding round-off problems 
    // Ov(w) = sum_d log_ovlp(d,w) 
    auto log_h = nda::to_host(log_ovlps);
    nda::array<ComplexType,1> ov_h(nw);
    for(int i=0; i<nw; ++i) {
      auto ai = log_h(nda::range::all,i);
      // find largest determinant for each walker to use as reference
      ComplexType log_m = *std::max_element(ai.begin(),ai.end());
      ComplexType sum = 0.0;
      for(int p=0; p<ndet; ++p)
        sum += std::exp( ai(p) - log_m );
      ov_h(i) = log_m + std::log(sum);
    }
    Ov() = ov_h();
  }
}
/*
 * Full green function of the trial wave function.
 */
template<MEMORY_SPACE MEM, class devPsiT>
memory::const_shared_array<HOST_MEMORY,ComplexType,3> NOMSD<MEM,devPsiT>::G_MF() {
  using nda::range;
  auto all = range::all;

  int nel   = (walker_type==COLLINEAR ? nup+ndown : nup);
  int nspin = (walker_type==COLLINEAR ? 2 : 1);
  int npol  = (walker_type==NONCOLLINEAR ? 2 : 1);
  int ndet  = ci.size();

  if (ndet == 1)
  {
    return memory::share_from_root(*mpi, [&] {
      nda::array<ComplexType,3> gMF(nspin,npol*NMO,npol*NMO);
      memory::buffered_array<MEM,ComplexType,1> Ov(1);
      gMF() = ComplexType(0.0);

      memory::buffered_array<MEM,ComplexType,3> PsiT(1,npol*NMO,nup);
      PsiT(0,all,all) = math::sparse::to_array<'H'>(OrbMats(0,0)());
      if constexpr (MEM==HOST_MEMORY) {
        auto G3d = nda::reshape(gMF,std::array<long,3>{1,nspin*npol*NMO,npol*NMO});
        det_ops::MixedDensityMatrix(OrbMats(0,0)(),PsiT,G3d(all,nda::range(npol*NMO),all),Ov,false);
        if( walker_type == COLLINEAR ) {
          PsiT(0,all,range(ndown)) = math::sparse::to_array<'H'>(OrbMats(0,1)());
          det_ops::MixedDensityMatrix(OrbMats(0,1)(),PsiT(all,all,range(ndown)),G3d(all,nda::range(npol*NMO,nspin*npol*NMO),all),Ov,false);
        }
      } else {
        memory::array<MEM,ComplexType,3> gt(1,npol*NMO,npol*NMO);
        det_ops::MixedDensityMatrix(OrbMats(0,0)(),PsiT,gt,Ov,false);
        gMF(0,all,all) = gt(0,all,all);
        if( walker_type == COLLINEAR ) {
          PsiT(0,all,range(ndown)) = math::sparse::to_array<'H'>(OrbMats(0,1)());
          det_ops::MixedDensityMatrix(OrbMats(0,1)(),PsiT(all,all,range(ndown)),gt,Ov,false);
          gMF(1,all,all) = gt(0,all,all);
        }
      }
      return gMF;
    });
  } else {

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
            PsiT(0,all,all) = math::sparse::to_array<'H'>(OrbMats(p,0)());
            if( walker_type == COLLINEAR ) 
              PsiTB(0,all,all) = math::sparse::to_array<'H'>(OrbMats(p,1)());
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
    return memory::share_from_root(*mpi, [&] {
      nda::array<ComplexType,3> gMF(nspin,npol*NMO,npol*NMO);
      gMF() = Gv();
      return gMF;
    });
  }
}
/*
 * Calculate mean field expectation value of Cholesky potentials
 */
template<MEMORY_SPACE MEM, class devPsiT>
void NOMSD<MEM,devPsiT>::vMF(memory::array_view<MEM,ComplexType,1> v, double dt) {
  using nda::range;
  auto all = range::all;
  utils::check(v.size() == number_of_cholesky_vectors(), "Size mismatch");
  utils::check(v.strides()[0] == 1, "Strides mismatch");
  auto v2d = nda::reshape(v,std::array<long,2>{1,v.size()});   
  v() = ComplexType(0.0);

  int nel   = (walker_type==COLLINEAR ? nup+ndown : nup);
  int nspin = (walker_type==COLLINEAR ? 2 : 1);
  int npol  = (walker_type==NONCOLLINEAR ? 2 : 1);
  int ndet  = ci.size();

  if (ndet == 1)
  {
    memory::buffered_array<MEM,ComplexType,1> Ov(1); 
    memory::buffered_array<MEM,ComplexType,2> G(1,nel*npol*NMO); 
    auto G3d = nda::reshape(G,std::array<long,3>{1,nel,npol*NMO});

    memory::buffered_array<MEM,ComplexType,3> PsiT(1,npol*NMO,nup);
    PsiT(0,all,all) = math::sparse::to_array<'H'>(OrbMats(0,0)());
    det_ops::MixedDensityMatrix(OrbMats(0,0)(),PsiT,G3d(all,nda::range(nup),all),Ov);
    if( walker_type == COLLINEAR ) { 
      PsiT(0,all,range(ndown)) = math::sparse::to_array<'H'>(OrbMats(0,1)());
      det_ops::MixedDensityMatrix(OrbMats(0,1)(),PsiT(all,all,range(ndown)),G3d(all,nda::range(nup,nel),all),Ov);
    }
    HamOp.vbias(G, v2d, dt);

  } else {

    RealType Osum(0.0);
    memory::buffered_array<MEM,ComplexType,2> Gsum(1,nspin*npol*NMO*npol*NMO);
    auto[n0, n1] = itertools::chunk_range(0, ndet*(ndet+1)/2, mpi->comm.size(), mpi->comm.rank());
    int last_p = -1;
    bool found = false;
    Gsum() = ComplexType(0.0);

    { // control buffer lifetime
      memory::buffered_array<MEM,ComplexType,1> Ov(1);
      memory::buffered_array<MEM,ComplexType,2> G(1,nspin*npol*NMO*npol*NMO);
      auto G4d = nda::reshape(G,std::array<long,4>{1,nspin,npol*NMO,npol*NMO});
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
            PsiT(0,all,all) = math::sparse::to_array<'H'>(OrbMats(p,0)());
            if( walker_type == COLLINEAR ) 
              PsiTB(0,all,all) = math::sparse::to_array<'H'>(OrbMats(p,1)());
          }

          Ov() = ComplexType(0.0);
          G() = ComplexType(0.0);
          det_ops::MixedDensityMatrix(OrbMats(q,0)(),PsiT,G4d(all,0,all,all),Ov,false);
          if( walker_type == COLLINEAR ) 
            det_ops::MixedDensityMatrix(OrbMats(q,1)(),PsiTB,G4d(all,1,all,all),Ov,false);
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

  }
}

template class NOMSD<HOST_MEMORY, PsiT_Matrix<HOST_MEMORY>>;
template class NOMSD<HOST_MEMORY, memory::const_shared_array<HOST_MEMORY,ComplexType,2>>;
#if defined(ENABLE_DEVICE)
template class NOMSD<DEVICE_MEMORY, PsiT_Matrix<DEVICE_MEMORY>>;
template class NOMSD<DEVICE_MEMORY, memory::const_shared_array<DEVICE_MEMORY,ComplexType,2>>;
#endif

} // namespace afqmc

} // namespace sfqmc
