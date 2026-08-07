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

#include <vector>
#include <tuple>
#include <algorithm>

#include "AFQMC/config.h"
#include "AFQMC/Wavefunctions/PHMSD.hpp"
#include "AFQMC/SlaterDeterminantOperations/density_matrix.hpp"
#include "numerics/operations/product.hpp"
#include "numerics/operations/determinants.hpp"
#if defined(ENABLE_DEVICE)
#include "numerics/device_kernels/cuda/phmsd_routines.cuh"
#endif

namespace sfqmc
{
namespace afqmc
{


template<MEMORY_SPACE MEM>
void PHMSD<MEM>::runtime_optimization(WalkerSet<MEM>& wset)
{
  const int nw   = wset.size();
  const int nel = (walker_type==COLLINEAR ? nup+ndown : nup );
  const int npol = (walker_type==NONCOLLINEAR ? 2 : 1 );
  memory::array<MEM,ComplexType,2> G(nw,nel*npol*NMO);
  // don't use buffered_array!!!
// This needs to depend on algorithm!!!
  HamOp.runtime_optimization(G);
}

/*
 * Calculates the bias potential.
 */
template<MEMORY_SPACE MEM>
void PHMSD<MEM>::vbias(WalkerSet<MEM>& wset, memory::array_view<MEM,ComplexType,2> v, double dt, [[maybe_unused]] int nt)
{
  memory::check_memory_space<MEM>(v);
  AFQMCTimer.start(G_for_vbias_timer);
  int nact  = OrbMats(0).extent(0) + (walker_type==COLLINEAR ? OrbMats(OrbMats.extent(0)-1).extent(0) : 0);
  int npol  = (walker_type==NONCOLLINEAR ? 2 : 1);
  int nw = wset.size();
  utils::check(v.shape() == std::array<long,2>{nw,HamOp.number_of_cholesky_vectors()},
               "Shape mismatch");
  memory::buffered_array<MEM,ComplexType,2> G(nw,nact*npol*NMO);
  memory::buffered_array<MEM,ComplexType,1> ovlp(nw);
  MixedDensityMatrix(wset, G, ovlp);
  AFQMCTimer.stop(G_for_vbias_timer);
  AFQMCTimer.start(vbias_timer);
  v() = ComplexType(0.0);
  HamOp.vbias(G, v, dt);
  AFQMCTimer.stop(vbias_timer);
}

template<MEMORY_SPACE MEM>
void PHMSD<MEM>::Energy(WalkerSet<MEM>& wset, [[maybe_unused]] int nt)
{
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

template<MEMORY_SPACE MEM>
void PHMSD<MEM>::Energy(WalkerSet<MEM> const& wset, memory::array_view<MEM,ComplexType,2> E, memory::array_view<MEM,ComplexType,1> Ov, [[maybe_unused]] int nt)
{
  if(energy_algorithm == PHMSDEnergyAlgorithm::reference)
    energy_alg0(wset,E,Ov);
  else
    energy_alg1(wset,E,Ov);
}

template<MEMORY_SPACE MEM>
void PHMSD<MEM>::MixedDensityMatrix(WalkerSet<MEM> const& wset, memory::array_view<MEM,ComplexType,2> G, bool compact)
{
  int nw = wset.size();
  memory::buffered_array<MEM,ComplexType,1> ovlp(nw,ComplexType(0.0));
  MixedDensityMatrix(wset, G, ovlp, compact);
}

template<MEMORY_SPACE MEM>
void PHMSD<MEM>::Log_Overlap(WalkerSet<MEM>& wset)
{
  int nw = wset.size();
  memory::buffered_array<MEM,ComplexType,1> ovlp(nw,ComplexType(0.0));
  Log_Overlap(wset, ovlp);
  wset.setProperty(OVLP, ovlp);
}

/*
 * Calculates the local energy and overlaps of all the walkers in the set and 
 * returns them in the appropriate data structures
 * alg0: Loop over unique configurations and do a full calculation of the GF and E.
 * Reference implementation, uses the least amount of memory, slow/inefficient. 
 */
template<MEMORY_SPACE MEM>
void PHMSD<MEM>::energy_alg0(WalkerSet<MEM> const& wset, memory::array_view<MEM,ComplexType,2> E, memory::array_view<MEM,ComplexType,1> Ov)
{
  using nda::range;
  auto all = range::all;
  int nspin = (walker_type==COLLINEAR?2:1);
  long nkev = HamOp.number_of_ke_vectors();
  int nwalk = wset.size();
  utils::check(E.extent(0) == nwalk and E.extent(1) == 3, "Size mismatch");
  utils::check(Ov.extent(0) == nwalk, "Size mismatch");

  E() = ComplexType(0.0);
  Ov() = ComplexType(0.0);

  long nexcit[2] = {long(abij.number_of_unique_excitations()[0]), 
              (walker_type==COLLINEAR ? long(abij.number_of_unique_excitations()[1]) : 0)};
  long maxn_unique_confg = std::max(nexcit[0],nexcit[1]);
  int nspin_in_orbmat = OrbMats.extent(0);
  long nact[2] = {OrbMats(0).extent(0),OrbMats( nspin_in_orbmat-1 ).extent(0)};
  long nel[2] = {nup,(walker_type==COLLINEAR?ndown:0)};
  // logarithm of base determinant
  memory::buffered_array<MEM,ComplexType,2> log_ov(2,nwalk); 
  // Opposite spin J contribution to Eloc
  memory::buffered_array<MEM,ComplexType,1> opSpinEJ(nwalk,ComplexType(0.0)); 
  // Determinant ratio wrt idet=0, e.g. Ov(spin,idet,iw)/Ov(spin,0,iw)
  memory::buffered_array<MEM,ComplexType,3> Ovmsd(nspin,maxn_unique_confg,nwalk); 
  // Local energy components from each unique determinant
  memory::buffered_array<MEM,ComplexType,4> Emsd(nspin,maxn_unique_confg,nwalk,3); 
  log_ov() = ComplexType(0.0);
  Ovmsd() = ComplexType(0.0);
  Emsd() = ComplexType(0.0);

  // 1. calculate eneries for unique determinants
  //    - Emsd[spin][nd_unique][iw][{0:E1, 1:EXX, 2:--}]
  //    - Ovmsd[spin][nd_unique][iw]
  if (walker_type == NONCOLLINEAR)
  {
/*
    nda::array<int,1> confg(nup);
    memory::array<MEM,int,1> confg_dev(nup);
    //ComplexType ov0;
    auto SM=wset.SlaterMatrices(Alpha);
    auto Gdims     = dm_dims(false, SpinTypes(0));
    auto Gdims_ref = dm_dims_ref(false, SpinTypes(0));
    int nr = Gdims.first * Gdims.second, nc = nwalk;
    if (transposed_G_for_E_)
      std::swap(nr, nc);
    StaticSHMMatrix G({nr, nc}, 
                   shm_buffer_manager.get_generator().template get_allocator<ComplexType>()); 
    // compute [nwalk][...] and transpose at end
    StaticSHM3Tensor Gwork({nwalk, Gdims_ref.first, Gdims_ref.second}, 
                   shm_buffer_manager.get_generator().template get_allocator<ComplexType>());
    int ix = ( transposed_G_for_E_ ? 0 : 1 );  
    Static3Tensor Gt({ix*(w1-w0), Gdims.first, Gdims.second},
                   buffer_manager.get_generator().template get_allocator<ComplexType>());
    
    for (int nd = 0; nd < int(nexcit[0]); ++nd)
    {
      abij.get_configuration(0, nd, confg);
      copy_n(confg.data(), confg.size(), confg_dev.origin());
      {
        int n0, n1;
        std::tie(n0, n1) = FairDivideBoundary(TG.TG_local().rank(), 
                                              int(G.num_elements()), 
                                              TG.TG_local().size());
        fill_n(G.origin()+n0, n1-n0, ComplexType(0.0));
        TG.TG_local().barrier();
      }
      // calculate G and overlap
      SDetOp.BatchedMixedDensityMatrixFromConfiguration(OrbMats[0], SM.sliced(w0,w1),
                      Gwork.sliced(w0, w1), LogOverlapFactor, Ovmsd[0][nd].sliced(w0,w1), 
                      confg_dev.origin(), true);
      // transpose if needed
      if(transposed_G_for_E_) 
      {
        fill_n(G[w0].origin(), (w1-w0)*G[w0].num_elements(), ComplexType(0.0));
        auto G3D = G.rotated().partitioned(Gdims.first).unrotated();  
        ma::copy_select(Gwork.sliced(w0,w1),G3D.sliced(w0,w1),confg_dev);
      } 
      else
      {
        fill_n(Gt.origin(), Gt.num_elements(), ComplexType(0.0));
        ma::copy_select(Gwork.sliced(w0,w1),Gt,confg_dev);
        ma::transpose( Gt.rotated().flatted().unrotated(), G(boost::multi::ALL, {w0,w1}) ); 
      }
      TG.TG_local().barrier();
      HamOp.energy(eloc2, G, 0, TG.TG_local().root(), true, true);
      if(TG.TG_local().size() > 1 ) {
        // reduce_n since Emsd is in shared memory (doing this instead of copy with mutex)
        TG.TG_local().reduce_n(raw_pointer_cast(eloc2.origin()), 3 * nwalk, 
                               raw_pointer_cast(Emsd[0][nd].origin()), std::plus<>(), 0);
        TG.TG_local().barrier();
      } else {
        copy_n(eloc2.origin(), 3 * nwalk, Emsd[0][nd].origin()); 
      }
    }  // nd
*/
  }
  else
  {
    memory::buffered_array<MEM,ComplexType,2> KEleft(nwalk,nkev); 
    memory::buffered_array<MEM,ComplexType,3> KEright(maxn_unique_confg,nwalk,nkev); 
    auto KEr2d = nda::reshape(KEright,std::array<long,2>{maxn_unique_confg,nwalk*nkev});
    memory::buffered_array<MEM,ComplexType,1> eloc(nwalk); 

    for (int spin = 0; spin < nspin; ++spin)
    {
      auto SM=wset.SlaterMatrices( (spin == 0 ? Alpha : Beta) );
      nda::array<int,1> confg(nel[spin]);
      memory::array<MEM,int,1> confg_dev(nel[spin]);

      memory::buffered_array<MEM,ComplexType,2> G(nwalk,nact[spin]*NMO); 
      auto G3d = nda::reshape(G,std::array<long,3>{nwalk,nact[spin],NMO});
      memory::buffered_array<MEM,ComplexType,2> Gwork(nwalk,nel[spin]*NMO); 
      auto Gwork3d = nda::reshape(Gwork, std::array<long,3>{nwalk,nel[spin],NMO});

      for (int nd = 0; nd < int(nexcit[spin]); ++nd)
      {
        abij.get_configuration(spin, nd, confg);
        confg_dev() = confg();
        G() = ComplexType(0.0);
        // calculate G and overlap
        det_ops::MixedDensityMatrixFromConfiguration(OrbMats(spin%nspin_in_orbmat),SM,Gwork3d,Ovmsd(spin,nd,all),confg_dev,true);
        if(nd == 0) {
          log_ov(spin,all) = Ovmsd(spin,nd,all); 
          nda::tensor::set(ComplexType(1.0),Ovmsd(spin,nd,all));
        } else {
          // log_ov(n) - log_ov(0)
          nda::tensor::add(ComplexType(-1.0),log_ov(spin,all),"w",ComplexType(1.0),Ovmsd(spin,nd,all),"w");
          // Ovmsd = exp(log_ov(n) - log_ov(0)) -> Ov(n)/Ov(0)
          nda::apply(ComplexType(1.0),Ovmsd(spin,nd,all),nda::tensor::op::EXP);
        }
        // kernel for gpu???
        for(int a=0; a<nel[spin]; ++a)
          G3d(all,confg(a),all) = Gwork3d(all,a,all);
        if (spin == 0) {
          HamOp.energy(Alpha, Emsd(spin,nd,all,all), G, 0,  KEright(nd,all,all));
        }
        else
        {
          HamOp.energy(Beta, Emsd(spin,nd,all,all), G, 0,  KEleft);
          // when spin == 1, KEright already contains contraction over unique up configurations
          // opSpinEJ(iw) += (sum_n KEr[nd][iw][n] KEl[iw][n] ) * Ovmsd[spin][nd][iw]  
          nda::tensor::contract(ComplexType(1.0),KEright(nd,all,all),"wn",KEleft,"wn",
                                ComplexType(0.0),eloc,"w"); 
          // tmp[iw] *= Ovmsd[spin][nd][iw]
          nda::tensor::elementwise(ComplexType(1.0),Ovmsd(spin,nd,all),"w",
                                   ComplexType(1.0),eloc,"w",nda::tensor::op::MUL);
          // opSpinEJ[iw] += tmp[iw]
          nda::tensor::add(ComplexType(1.0),eloc,"w",ComplexType(1.0),opSpinEJ,"w");  
        }
      }  // nd
      if (spin == 0) {
        // scale KE vectors by overlaps:  KEright[nd][iw][n] *= Ovmsd[0][nd][iw] 
        if constexpr (MEM==HOST_MEMORY)
          for(int d=0; d<nexcit[0]; ++d)
            for(int w=0; w<nwalk; ++w)
              KEright(d,w,all) *= Ovmsd(0,d,w);
        else
          nda::tensor::elementwise(ComplexType(1.0),Ovmsd(0,range(nexcit[0]),all),"dw",
                                   ComplexType(1.0),KEright(range(nexcit[0]),all,all),"dwn",nda::tensor::op::MUL);
        // Tdn[idet_down][iw][n] = DetCouplings[idet_down][idet_up] * KEright[idet_up][iw][n] 
        memory::buffered_array<MEM,ComplexType,2> Tdn(nexcit[0],nwalk*nkev); 
        Tdn() = KEr2d(range(nexcit[0]),all);
        math::sparse::csrmm<'N'>(OpSpinDetCouplings(1),Tdn,KEr2d(range(nexcit[1]),all));
      }
    } // spin
  }

  // 2. assemble sum over configurations
  if (walker_type == COLLINEAR)
  {
    // weight[iw]_u/d = sum_d/u Ov[u][iw] *OpSpinDetCouplings[u][d] Ov[d][iw] 
    // E[iw] += sum_u weight[iw]_u * E[u][iw] + sum_d weight[iw]_d * E[d][iw]
    // Ovmsd( {nspin, maxn_unique_confg, nwalk}    )
    // Emsd ( {nspin, maxn_unique_confg, nwalk, 3} )
    memory::buffered_array<MEM,ComplexType,2> wgt(maxn_unique_confg, nwalk);
    // spin up
    // W[u][iw] = sum_d OpSpinDetCouplings[u][d] Ov[d][iw]   
    math::sparse::csrmm<'N'>(OpSpinDetCouplings(0),Ovmsd(1,range(nexcit[1]),all),wgt(range(nexcit[0]),all));
    // Ov[u][iw] = sum_u wgt(u,iw) * Ovmsd[0][u][iw]   
    nda::tensor::contract(ComplexType(1.0),Ovmsd(0,range(nexcit[0]),all),"uw",
             wgt(range(nexcit[0]),all),"uw",ComplexType(0.0),Ov,"w");

    // W[u][iw] *= Ovmsd[0][u][iw]   
    nda::tensor::elementwise(ComplexType(1.0),Ovmsd(0,range(nexcit[0]),all),"uw",
                         ComplexType(1.0),wgt(range(nexcit[0]),all),"uw",nda::tensor::op::MUL);
    if constexpr (MEM==HOST_MEMORY) {
      for(int u=0; u<nexcit[0]; ++u) 
        for(int w=0; w<nwalk; ++w) 
          Emsd(0,u,w,all) *= wgt(u,w);
    } else { 
      nda::tensor::elementwise(ComplexType(1.0),wgt(range(nexcit[0]),all),"uw",
              ComplexType(1.0),Emsd(0,range(nexcit[0]),all,all),"uwc",nda::tensor::op::MUL);
    }

    // spin down
    // W[d][iw] = sum_u OpSpinDetCouplings[d][u] Ov[u][iw]   
    math::sparse::csrmm<'N'>(OpSpinDetCouplings(1),Ovmsd(0,range(nexcit[0]),all),wgt(range(nexcit[1]),all));
    // W[u][iw] *= Ovmsd[0][u][iw]   
    nda::tensor::elementwise(ComplexType(1.0),Ovmsd(1,range(nexcit[1]),all),"uw",
                         ComplexType(1.0),wgt(range(nexcit[1]),all),"uw",nda::tensor::op::MUL);
    if constexpr (MEM==HOST_MEMORY) {
      for(int u=0; u<nexcit[1]; ++u)
        for(int w=0; w<nwalk; ++w) 
          Emsd(1,u,w,all) *= wgt(u,w);
    } else {
      nda::tensor::elementwise(ComplexType(1.0),wgt(range(nexcit[1]),all),"uw",
              ComplexType(1.0),Emsd(1,range(nexcit[1]),all,all),"uwc",nda::tensor::op::MUL);
    }
  } else {
/*
    // Ovmsd[0][u][iw] *= OpSpinDetCouplings[d][u]
    Vector_ref_<pointer> Xu(make_device_ptr(OpSpinDetCouplings[1].non_zero_values_data(0)), 
			    iextensions<1u>{nexcit[0]});
    ma::elementwise(ma::TOp_MUL, 0, Xu, Ovmsd[0].sliced(0, nexcit[0]));

    // Ov[iw] = sum_u Ovmsd[0][u][iw]   
    ma::accumulate(0, Ovmsd[0].sliced(0, nexcit[0]), Ov); 

    // Emsd[0][u][iw][0:3] *= W[u][iw] 
    ma::elementwise(ma::TOp_MUL, 0, Ovmsd[0].sliced(0, nexcit[0]).flatted(),
                        Emsd[0].sliced(0, nexcit[0]).flatted());     
*/
  }
  if constexpr (MEM==HOST_MEMORY)  {
    for(int is=0; is<nspin; is++) 
      for(int n=0; n<maxn_unique_confg; n++) 
        E() += Emsd(is,n,all,all);
    if(walker_type == COLLINEAR) 
      E(all,2) += opSpinEJ();
    for(int iw=0; iw<nwalk; iw++) 
      E(iw,all) /= Ov(iw);
  } else {
    // implemented in nda_functions.hpp
    nda::tensor::reduce(ComplexType(1.0),Emsd,"snwc",ComplexType(0.0),E,"wc",nda::tensor::op::SUM);
    if(walker_type == COLLINEAR) 
      nda::tensor::add(ComplexType(1.0),opSpinEJ,"w",ComplexType(1.0),E(all,2),"w");
    memory::buffered_array<MEM,ComplexType,1> Ot(Ov);
    nda::apply(ComplexType(1.0),Ot,nda::tensor::op::RCP);
    nda::tensor::elementwise(ComplexType(1.0),Ot,"w",
                             ComplexType(1.0),E,"wc",nda::tensor::op::MUL);
  }
  // Ov -> log( Ov(n)/Ov(0) ) 
  nda::apply(ComplexType(1.0),Ov,nda::tensor::op::LOG);
  // Ov += log_ov(0) -> log(Ov)
  if(walker_type == COLLINEAR) 
    nda::tensor::add(ComplexType(1.0),log_ov(1,all),"w",ComplexType(1.0),log_ov(0,all),"w");
  nda::tensor::add(ComplexType(1.0),log_ov(0,all),"w",ComplexType(1.0),Ov,"w");
}

/*
 * Calculates the local energy and overlaps of all the walkers in the set and 
 * returns them in the appropriate data structures
 */
template<MEMORY_SPACE MEM>
void PHMSD<MEM>::energy_alg1(WalkerSet<MEM> const& wset, memory::array_view<MEM,ComplexType,2> E, memory::array_view<MEM,ComplexType,1> Ov)
{
  using nda::range;
  auto all = range::all;
  ComplexType one(1.0),zero(0.0);
  long nkev = HamOp.number_of_ke_vectors();
  int nwalk = wset.size();
  utils::check(E.extent(0) == nwalk and E.extent(1) == 3, "Size mismatch");
  utils::check(Ov.extent(0) == nwalk, "Size mismatch");

  E() = zero;
  Ov() = zero;

  long nexcit[2] = {long(abij.number_of_unique_excitations()[0]),
              (walker_type==COLLINEAR ? long(abij.number_of_unique_excitations()[1]) : 0)};
  long maxn_unique_confg = std::max(nexcit[0],nexcit[1]);
  int nspin_in_orbmat = OrbMats.extent(0);
  int nactA = OrbMats(0).extent(0);
  int nactB = (walker_type==COLLINEAR ? OrbMats(nspin_in_orbmat-1).extent(0) : 0);
  memory::buffered_array<MEM,ComplexType,1> log_ov(nwalk,zero); 

  if (walker_type != COLLINEAR)
  {
    utils::check(false,"Error: Finish implementation of PHMSD for CLOSED/NONCOLLINEAR walkers.");
  } 
  else  
  {

    // Walker-block size: process nwalk_block_size walkers per energy batch so the KEright/Tdn
    // device temporaries stay bounded regardless of the total walker count. Per-walker energies
    // are independent, so the block size is numerically irrelevant (nwalk_block_size input).
    int nbatch__ = std::min(nwalk, std::max(1, nwalk_block_size));
    auto SMup=wset.SlaterMatrices(Alpha);
    auto SMdn=wset.SlaterMatrices(Beta);

    for (int iw = 0; iw < nwalk; iw+=nbatch__)
    {
      int nw = std::min(nbatch__, nwalk - iw);  
      nda::range w_rng(iw,iw+nw);

      memory::buffered_array<MEM,ComplexType,3> KEright(maxn_unique_confg,nw,nkev);
      auto KEr2d = nda::reshape(KEright,std::array<long,2>{maxn_unique_confg,nw*nkev});
      memory::buffered_array<MEM,ComplexType,2> eloc(nw,3);

      memory::buffered_array<MEM,ComplexType,3> GA(nw,nup,NMO);
      auto GA2d = nda::reshape(GA,std::array<long,2>{nw,nup*NMO});
      memory::buffered_array<MEM,ComplexType,3> QQ0inv0(nw,nactA,nup);
      memory::buffered_array<MEM,ComplexType,2> ovlp_ratios_up(nexcit[0],nw);

      memory::buffered_array<MEM,ComplexType,3> GB(nw,ndown,NMO);
      auto GB2d = nda::reshape(GB,std::array<long,2>{nw,ndown*NMO});
      memory::buffered_array<MEM,ComplexType,3> QQ0inv1(nw,nactB,ndown);
      memory::buffered_array<MEM,ComplexType,2> ovlp_ratios_dn(nexcit[1],nw);

      // 1. calculate list of overlaps
      det_ops::MixedDensityMatrixForWoodbury(OrbMats(0),SMup(w_rng,all,all),GA,
          log_ov(w_rng),QQ0inv0,abij.get_reference_configuration_device(0),true);
      calculate_overlaps(0, abij, QQ0inv0, ovlp_ratios_up);

      det_ops::MixedDensityMatrixForWoodbury(OrbMats(nspin_in_orbmat-1),SMdn(w_rng,all,all),GB,
          log_ov(w_rng),QQ0inv1,abij.get_reference_configuration_device(1),true);
      calculate_overlaps(1, abij, QQ0inv1, ovlp_ratios_dn);

      { // spin up
        // 2a. calculate weights
        memory::buffered_array<MEM,ComplexType,2> wgt(nexcit[0], nw);
        // W[u][iw] = sum_d OpSpinDetCouplings[u][d] ovlp_ratio[d][iw]   
        math::sparse::csrmm<'N'>(OpSpinDetCouplings(0),ovlp_ratios_dn,wgt);
        // Ov[iw] = sum_u W(u,iw) * ovlp_ratio(u,iw)  
        nda::tensor::contract(one,ovlp_ratios_up,"uw",wgt,"uw",zero,Ov(w_rng),"w");
        // W(u,iw) *= ovlp_ratio(u,iw)   
        nda::tensor::elementwise(one,ovlp_ratios_up,"uw",one,wgt,"uw",nda::tensor::op::MUL);

        // 3a. Reference energy and temporary matrices
        HamOp.ph_reference_energy(Alpha, eloc, GA2d, KEright(0,all,all));

        // MAM: ph_excited_energies calculates and adds sum_d wgt[d] * (E[d]-Eref)
        // so I need to add here: sum_d wgt[d] Eref = Ov * Eref
        // E[iw][:] *= Ov[iw]   
        if constexpr (MEM==HOST_MEMORY) {
          for(int i=0; i<nw; ++i)
            eloc(i,all) *= Ov(iw+i);
        } else {
          nda::tensor::elementwise(one,Ov(w_rng),"w",one,eloc,"wi",nda::tensor::op::MUL);
        }
        nda::tensor::add(one, eloc, "wi", one, E(w_rng,all), "wi");

        // 4a. calculate R matrices and evaluate E, KEright 
        ph_excited_energies_first_step(abij, wgt, QQ0inv0, eloc,
                                       KEright(range(nexcit[0]),all,all), HamOp, ndet_block_size);
        nda::tensor::add(one, eloc, "wi", one, E(w_rng,all), "wi");

        memory::buffered_array<MEM,ComplexType,3> Tdn(nexcit[0], nw, nkev);
        Tdn() = zero;
        // scale KE vectors by overlaps:  KEright[nd][iw][n] *= Ovmsd[0][nd][iw] 
        if constexpr (MEM==HOST_MEMORY) {
          for(int i=0; i<nexcit[0]; i++)
            for(int w=0; w<nw; w++)
              KEright(i,w,all) *= ovlp_ratios_up(i,w);
        } else {
          nda::tensor::elementwise(one,ovlp_ratios_up,"nw",
                   one,KEright(range(nexcit[0]),all,all),"nwk",nda::tensor::op::MUL);
        }

        // Tdn[idet_down][iw][n] = DetCouplings[idet_down][idet_up] * KEright[idet_up][iw][n] 
        Tdn() = KEright(range(nexcit[0]),all,all); 
        auto Tdn2d = nda::reshape(Tdn,std::array<long,2>{nexcit[0],nw*nkev}); 
        math::sparse::csrmm<'N'>(OpSpinDetCouplings(1),Tdn2d,KEr2d(range(nexcit[1]),all));
        // scale KE vectors by overlaps:  KEright[nd][iw][n] *= Ovmsd[1][nd][iw] 
        if constexpr (MEM==HOST_MEMORY) {
          for(int i=0; i<nexcit[1]; i++)
            for(int w=0; w<nw; w++)
              KEright(i,w,all) *= ovlp_ratios_dn(i,w);
        } else {
          nda::tensor::elementwise(one,ovlp_ratios_dn,"nw",
                   one,KEright(range(nexcit[1]),all,all),"nwk",nda::tensor::op::MUL);
        }
      }

      { // spin down
        // 2b. calculate weights
        memory::buffered_array<MEM,ComplexType,2> wgt(nexcit[1], nw);
        // W[u][iw] = sum_d OpSpinDetCouplings[u][d] Ov[d][iw]   
        math::sparse::csrmm<'N'>(OpSpinDetCouplings(1),ovlp_ratios_up,wgt);
        // W[u][iw] *= Ovlps[0][u][iw]   
        nda::tensor::elementwise(one,ovlp_ratios_dn,"uw",one,wgt,"uw",nda::tensor::op::MUL);

        // 3b. Reference energy and temporary matrices
        memory::buffered_array<MEM,ComplexType,2> KEleft(nw, nkev);
        HamOp.ph_reference_energy(Beta, eloc, GB2d, KEleft);

        // E[iw][:] *= Ov[iw] ; see note above about factor of Ov (instead of wgt)
        if constexpr (MEM==HOST_MEMORY) {
          for(int i=0; i<nw; ++i)
            eloc(i,all) *= Ov(iw+i);
        } else {
          nda::tensor::elementwise(one,Ov(w_rng),"w",one,eloc,"wi",nda::tensor::op::MUL);
        }
        nda::tensor::add(one, eloc, "wi", one, E(w_rng,all), "wi");

        memory::buffered_array<MEM,ComplexType,1> tmp(nw);
        // tmp(iw) = sum_n KEr[nd][iw][n] KEl[iw][n]
        nda::tensor::contract(one,KEright(0,all,all),"wn",KEleft,"wn",zero,tmp,"w"); 
        // opSpinEJ[iw] += tmp[iw]
        nda::tensor::add(one,tmp,"w",one,E(w_rng,2),"w");

        //4b. calculate R matrices and evaluate E, KEright
        ph_excited_energies_second_step(abij, wgt, QQ0inv1, eloc,
                                        KEright(range(nexcit[1]),all,all), HamOp, ndet_block_size);
        nda::tensor::add(one, eloc, "wi", one, E(w_rng,all), "wi");
      }
    }
  }

  if constexpr (MEM==HOST_MEMORY)  {
    for(int iw=0; iw<nwalk; iw++)
      E(iw,all) /= Ov(iw);
  } else {
    memory::buffered_array<MEM,ComplexType,1> Ot(Ov);
    nda::apply(ComplexType(1.0),Ot,nda::tensor::op::RCP);
    nda::tensor::elementwise(ComplexType(1.0),Ot,"w",
                             ComplexType(1.0),E,"wc",nda::tensor::op::MUL);
  }
  // Ov -> log( Ov(n)/Ov(0) ) 
  nda::apply(ComplexType(1.0),Ov,nda::tensor::op::LOG);
  // Ov += log_ov(0) -> log(Ov)
  nda::tensor::add(ComplexType(1.0),log_ov,"w",ComplexType(1.0),Ov,"w");
}

/*
 * Computes the mixed density matrix
 */
template<MEMORY_SPACE MEM>
void PHMSD<MEM>::MixedDensityMatrix(WalkerSet<MEM> const& wset, memory::array_view<MEM,ComplexType,2> G, memory::array_view<MEM,ComplexType,1> Ov, bool compact)
{
  using nda::range;
  auto all = nda::range::all;
  memory::check_memory_space<MEM>(G,Ov);
  const int nspin = (walker_type==COLLINEAR?2:1);
  const int npol  = (walker_type==NONCOLLINEAR?2:1);
  const int nwalk = wset.size();
  utils::check(Ov.size() == nwalk, "Size mismatch");
  const int nspin_in_orbmat = OrbMats.extent(0); 
  const int nactA = OrbMats(0).extent(0);
  const int nactB = (walker_type==COLLINEAR ? OrbMats(nspin_in_orbmat-1).extent(0) : 0);
  const int nact = nactA+nactB;
  const int nc    = ( compact ? nact : nspin*npol*NMO );
  utils::check(G.shape() == std::array<long,2>{nwalk,nc*npol*NMO}, "Size mismatch");
  G() = ComplexType(0.0);
  Ov() = ComplexType(0.0);

  if (walker_type == NONCOLLINEAR)
  {
    utils::check(false, "noncollinear PHMSD MixedDensityMatrix not implemented");
/*
    // always calculate compact and multiply by OrbMat at the end if full
    auto Gdims      = dm_dims(false, Alpha);
    auto Gdims_full = dm_dims(true, Alpha);
    if (compact)
      Gdims_full = {0, 0};
    auto Gdims0 = dm_dims_ref(false, Alpha);
    //size_t cnt   = 0;
    const int ntasks_percore      = nw / TG.getNCoresPerTG();
    const int ntasks_total_serial = ntasks_percore * TG.getNCoresPerTG();
    const int nextra              = nw - ntasks_total_serial;
    if(nextra != 0) 
      APP_ABORT(" Error in PHMSD: ncores must divide #walkers.");	

    // each processor does ntasks_percore_serial overlaps serially
    const int w0 = TG.getLocalTGRank() * ntasks_percore;
    const int wN = (TG.getLocalTGRank() + 1) * ntasks_percore;
    const int nbatch__  = std::min((wN-w0), (nbatch < 0 ? (wN-w0) : nbatch));

    StaticVector overlaps(iextensions<1u>{nbatch__}, ComplexType(0.0), 
                       buffer_manager.get_generator().template get_allocator<ComplexType>());
    // storage for reference Green functions
    Static3Tensor GA0({nbatch__, Gdims0.first, Gdims0.second}, 
                       buffer_manager.get_generator().template get_allocator<ComplexType>());
    Static3Tensor QQ0inv0_buff({nbatch__, OrbMats[0].size(0), nup}, 
                     buffer_manager.get_generator().template get_allocator<ComplexType>());

    auto SM=wset.SlaterMatrices(Alpha);

    for (int iw = w0; iw < wN; iw+=nbatch__)
    {
      int nb = std::min(nbatch__, wN - iw);  
      auto GA= GA0.sliced(0, nb);  
      auto local_QQ0inv0= QQ0inv0_buff.sliced(0, nb);  

      // 1. calculate list of overlaps
      SDetOp.BatchedMixedDensityMatrixForWoodbury(OrbMats[0], SM.sliced(iw,iw+nb), GA, 
            LogOverlapFactor, overlaps.sliced(0,nb), refc_dev.origin(), local_QQ0inv0, true);

#if defined(ENABLE_CUDA)
      calculate_overlaps(0, 1, 0, abij, local_QQ0inv0, Ovlps[0].sliced(0,nexcit[0]));
#else
      for(int i=0; i<nb; ++i)
        calculate_overlaps(0, 1, 0, abij, local_QQ0inv0[i], Ovlps[0]({0,nexcit[0]}, iw+i));
#endif

      // multiply Ovlps by overlaps
      // Ovlps[s][u][iw] *= overlaps[iw]   
      ma::elementwise(ma::TOp_MUL, 1, overlaps.sliced(0,nb), Ovlps[0]({0,nexcit[0]},{iw,iw+nb}));

      // 2. generate R[Nact,Nel] and generate G
      {
        // Ovlps[0][u][iw] *= OpSpinDetCouplings[d][u]
        Vector_ref_<pointer> Xu(make_device_ptr(OpSpinDetCouplings[1].non_zero_values_data(0)), 
			    iextensions<1u>{nexcit[0]});
        ma::elementwise(ma::TOp_MUL, 0, Xu, Ovlps[0]({0, nexcit[0]},{iw,iw+nb}));

        // Ov[iw] = sum_u Ovlps[0][u][iw]   
        ma::accumulate(0, Ovlps[0]({0, nexcit[0]},{iw,iw+nb}), Ov.sliced(iw,iw+nb)); 

        Static3Tensor Ra({nb, nup, long(OrbMats[0].size(0))}, ComplexType(0.0), 
                       buffer_manager.get_generator().template get_allocator<ComplexType>());
#if defined(ENABLE_CUDA)
        calculate_R(0, 1, 0, abij, local_QQ0inv0, Ovlps[0]({0, nexcit[0]},{iw,iw+nb}), Ra);
#else
        for(int i=0; i<nb; ++i)
          calculate_R(0, 1, 0, abij, local_QQ0inv0[i], Ovlps[0](boost::multi::ALL,i), Ra[i]);
#endif

        if (transpose)
        {
          if (compact)
          { 
            Array_ref<ComplexType, 3, pointer> Gw(make_device_ptr(G[iw].origin()), 
                                               {nb, Gdims.first, Gdims.second});
            ma::productStridedBatched(T(Ra), GA, Gw(Gw.extension(0), {0,Gdims.first}, Gw.extension(2)));
          }
          else
          {
            Array_ref<ComplexType, 3, pointer> Gw(make_device_ptr(G[iw].origin()), 
                                    {nb, Gdims_full.first, Gdims_full.second});
            Static3Tensor GAt({nb, Gdims.first, Gdims.second}, 
                         buffer_manager.get_generator().template get_allocator<ComplexType>());
            ma::productStridedBatched(T(Ra), GA, GAt);
            ma::productStridedBatched(T(OrbMats[0]), GAt, Gw); 
          }
        }
        else
        {
          Static3Tensor GAt({nb, Gdims.first, Gdims.second}, 
                         buffer_manager.get_generator().template get_allocator<ComplexType>());
          if (compact)
          {
            ma::productStridedBatched(T(Ra), GA, GAt);
            ma::transpose(GAt.rotated().flatted().unrotated(),
                          G({0, Gdims.first * Gdims.second}, {iw, iw+nb}));
          }
          else
          {
            Static3Tensor Gfulla({nb, Gdims_full.first, Gdims_full.second}, 
                       buffer_manager.get_generator().template get_allocator<ComplexType>());
            ma::productStridedBatched(T(Ra), GA, GAt);
            ma::productStridedBatched(T(OrbMats[0]), GAt, Gfulla);
            ma::transpose(Gfulla.rotated().flatted().unrotated(),
                          G({0, Gdims_full.first * Gdims_full.second}, {iw, iw+nb}));
          }
        }
      }
    }
*/
  }
  else
  {
    const auto& nexcit = abij.number_of_unique_excitations();
    const int nbatch__  = nwalk; 

    auto SMup=wset.SlaterMatrices(Alpha);
    auto SMdn=wset.SlaterMatrices(Beta);

    for (int iw = 0; iw < nwalk; iw+=nbatch__)
    {
      int nw = std::min(nbatch__, nwalk - iw);  
      memory::buffered_array<MEM,ComplexType,1> log_ov(nw); 
      log_ov() = ComplexType(0.0);

      memory::buffered_array<MEM,ComplexType,3> GA(nw,nup,NMO); 
      memory::buffered_array<MEM,ComplexType,3> QQ0inv0(nw,nactA,nup); 
      memory::buffered_array<MEM,ComplexType,2> ovlp_ratios_up(nexcit[0],nw);

      memory::buffered_array<MEM,ComplexType,3> GB(nw,ndown,NMO); 
      memory::buffered_array<MEM,ComplexType,3> QQ0inv1(nw,nactB,ndown); 
      memory::buffered_array<MEM,ComplexType,2> ovlp_ratios_dn(nexcit[1],nw);

      // 1. calculate list of overlaps
      det_ops::MixedDensityMatrixForWoodbury(OrbMats(0),SMup(range(iw,iw+nw),all,all),GA,
          log_ov,QQ0inv0,abij.get_reference_configuration_device(0),true);
      calculate_overlaps(0, abij, QQ0inv0, ovlp_ratios_up);

      det_ops::MixedDensityMatrixForWoodbury(OrbMats(nspin_in_orbmat-1),SMdn(range(iw,iw+nw),all,all),GB,
          log_ov,QQ0inv1,abij.get_reference_configuration_device(1),true);
      calculate_overlaps(1, abij, QQ0inv1, ovlp_ratios_dn);

      // 2. generate R[Nact,Nel] and generate G
      {
        // log(Ov(iw)) = log_ov(iw) + std::log( sum_n c(n) * ovlp_ratios(n,iw) ) 
        memory::buffered_array<MEM,ComplexType,2> wgt(nexcit[0], nw);
        // W[u][iw] = sum_d OpSpinDetCouplings[u][d] Ov[d][iw]   
        math::sparse::csrmm<'N'>(OpSpinDetCouplings(0),ovlp_ratios_dn,wgt);
        // 1. Ov(iw) = sum_n wgt(n,iw) * ov_up(n,iw) 
        nda::tensor::contract(ComplexType(1.0),ovlp_ratios_up,"nw",wgt,"nw",ComplexType(0.0),Ov(range(iw,iw+nw)),"w");

        // wgt() = wgt() * ovlp_ratios_up()  
        nda::tensor::elementwise(ComplexType(1.0),ovlp_ratios_up,"nw",
                                 ComplexType(1.0),wgt,"nw",nda::tensor::op::MUL);

        // calculate R
        memory::buffered_array<MEM,ComplexType,3> Ra(nw,nup,nactA);
        Ra() = ComplexType(0.0);
        calculate_R(0, abij, QQ0inv0, wgt, Ra);

        if (compact)
        { 
          auto G3d = nda::reshape(G,std::array<long,3>{nwalk,nact,NMO}); 
          if constexpr (MEM==HOST_MEMORY) {
            for(int i=0; i<nw; ++i)
              nda::blas::gemm(nda::transpose(Ra(i,all,all)),GA(i,all,all),
                              G3d(iw+i,range(nactA),all));
          } else {
            nda::tensor::contract(ComplexType(1.0), Ra, "wia", GA, "wik", 
                ComplexType(0.0), G3d(range(iw,iw+nw),range(nactA),all), "wak");
          }
        }
        else
        {
          auto G4d = nda::reshape(G,std::array<long,4>{nwalk,nspin,NMO,NMO}); 
          memory::buffered_array<MEM,ComplexType,3> Gtmp(nw,nactA,NMO); 
          nda::tensor::contract(ComplexType(1.0), Ra, "wia", GA, "wik", 
              ComplexType(0.0), Gtmp, "wak");
          math::sparse::csrmm<'T'>(OrbMats(0),Gtmp,G4d(range(iw,iw+nw),0,all,all));
        }
      }

      {  // repeat for beta
        memory::buffered_array<MEM,ComplexType,2> wgt(nexcit[1], nw);
        // W[u][iw] = sum_d OpSpinDetCouplings[u][d] Ov[d][iw]   
        math::sparse::csrmm<'N'>(OpSpinDetCouplings(1),ovlp_ratios_up,wgt);
        // wgt() = wgt() * ovlp_ratios_up()  
        nda::tensor::elementwise(ComplexType(1.0),ovlp_ratios_dn,"nw",
                                 ComplexType(1.0),wgt,"nw",nda::tensor::op::MUL);

        // calculate R
        memory::buffered_array<MEM,ComplexType,3> Rb(nw,ndown,nactB);
        Rb() = ComplexType(0.0);
        calculate_R(1, abij, QQ0inv1, wgt, Rb);

        if (compact)
        {
          auto G3d = nda::reshape(G,std::array<long,3>{nwalk,nact,NMO});
          math::product<'T','N'>(Rb, GB, G3d(range(iw,iw+nw), range(nactA,nact), all));
        }
        else
        {
          auto G4d = nda::reshape(G,std::array<long,4>{nwalk,nspin,NMO,NMO});
          memory::buffered_array<MEM,ComplexType,3> Gtmp(nw,nactB,NMO);
          nda::tensor::contract(ComplexType(1.0), Rb, "wia", GB, "wik",
              ComplexType(0.0), Gtmp, "wak");
          math::sparse::csrmm<'T'>(OrbMats(nspin_in_orbmat-1),Gtmp,G4d(range(iw,iw+nw),1,all,all));
        }
      }

      // G -> G/Ov, where both G and Ov are missing a factor of ov(n=0)
      if constexpr (MEM==HOST_MEMORY) { 
        for(int i=0; i<nw; ++i)
          G(iw+i,all) /= Ov(iw+i); 
        // Ov -> log( Ov ) + log_ov(n=0) 
      } else {
        memory::buffered_array<MEM,ComplexType,1> Ot(nw);
        Ot() = Ov(range(iw,iw+nw)); 
        nda::apply(ComplexType(1.0),Ot,nda::tensor::op::RCP);
        nda::tensor::elementwise(ComplexType(1.0),Ot,"w",ComplexType(1.0),G(range(iw,iw+nw),all),"wi",nda::tensor::op::MUL);
      }
      // Ov -> log( Ov ) + log_ov(n=0) 
      nda::apply(ComplexType(1.0),Ov(range(iw,iw+nw)),nda::tensor::op::LOG);
      // 3. Ov(iw) += log_ov(iw)
      nda::tensor::add(ComplexType(1.0),log_ov,"w",ComplexType(1.0),Ov(range(iw,iw+nw)),"w");
    }
  }
}

/*
 * Calculates the overlaps of all walkers in the set. Returns values in arrays. 
 * Ov is assumed to be local to the core
 */
template<MEMORY_SPACE MEM>
void PHMSD<MEM>::Log_Overlap(WalkerSet<MEM> const& wset, memory::array_view<MEM,ComplexType,1> Ov, [[maybe_unused]] int nt)
{
  using nda::range;
  auto all = range::all;
  int nspin_in_orbmat = OrbMats.extent(0);
  const int nwalk = wset.size();
  utils::check(Ov.extent(0) == nwalk, "");
  Ov() = ComplexType(0.0);
  if (walker_type == NONCOLLINEAR)
  {
    utils::check(false, "noncollinear PHMSD MixedDensityMatrix not implemented");
/*
    long nexcit = long(abij.number_of_unique_excitations()[0]);
    StaticVector ov0(iextensions<1u>{nbatch__}, ComplexType(0.0),
                       buffer_manager.get_generator().template get_allocator<ComplexType>());
    StaticMatrix ovlp_list({nexcit, nbatch__}, ComplexType(0.0),
                     buffer_manager.get_generator().template get_allocator<ComplexType>());
    Static3Tensor QQ0inv0_buff({nbatch__, OrbMats[0].size(0), nup}, 
                     buffer_manager.get_generator().template get_allocator<ComplexType>());

    auto SM=wset.SlaterMatrices(Alpha);

    for (int iw = w0; iw < wN; iw+=nbatch__)
    {
      int nb = std::min(nbatch__, wN - iw);
      auto local_QQ0inv0= QQ0inv0_buff.sliced(0, nb) ;

      // alpha  
      SDetOp.BatchedOverlapForWoodbury(OrbMats[0], SM.sliced(iw,iw+nb), LogOverlapFactor, 
                ov0.sliced(0,nb), refc_dev.origin(), local_QQ0inv0);
#if defined(ENABLE_CUDA)
      calculate_overlaps(0, 1, 0, abij, local_QQ0inv0, ovlp_list.sliced(0,nexcit));
#else
      for(int i=0; i<nb; ++i)
        calculate_overlaps(0, 1, 0, abij, local_QQ0inv0[i], ovlp_list({0,nexcit}, i));
#endif

      // multiply Ovlps by overlaps
      // Ovlps[s][u][iw] *= ov0[iw]   
      ma::elementwise(ma::TOp_MUL, 1, ov0.sliced(0,nb), ovlp_list({0,nexcit[0]},{0,nb}));

      // Ovlps[0][u][iw] *= OpSpinDetCouplings[d][u]
      Vector_ref_<pointer> Xu(make_device_ptr(OpSpinDetCouplings[1].non_zero_values_data(0)), 
			    iextensions<1u>{nexcit[0]});
      ma::elementwise(ma::TOp_MUL, 0, Xu, ovlp_list({0, nexcit[0]},{0,nb}));

      // Ov[iw] = sum_u ovlp_list[u][iw]   
      ma::accumulate(0, ovlp_list({0, nexcit[0]},{0,nb}), Ov.sliced(iw,iw+nb)); 
    }
*/
  }
  else
  {
    auto nexcit = std::to_array<long>({
      abij.number_of_unique_excitations()[0],
      abij.number_of_unique_excitations()[1]
    });
    int nwbatch = nwalk; // tune later

    auto SMup=wset.SlaterMatrices(Alpha);
    auto SMdn=wset.SlaterMatrices(Beta);

    for (int iw = 0; iw < nwalk; iw+=nwbatch)
    {
      int nw = std::min(nwbatch, nwalk-iw);
      memory::buffered_array<MEM,ComplexType,1> log_ov(nw, ComplexType(0.0));
      memory::buffered_array<MEM,ComplexType,2> ovlp_ratios_up(nexcit[0],nw);
      memory::buffered_array<MEM,ComplexType,2> ovlp_ratios_dn(nexcit[1],nw);

      {
        // alpha  
        memory::buffered_array<MEM,ComplexType,3> QQ0inv0(nw,OrbMats(0).extent(0),nup);
        det_ops::Log_OverlapForWoodbury(OrbMats(0),SMup(range(iw,iw+nw),all,all),log_ov,QQ0inv0,abij.get_reference_configuration_device(0));
        calculate_overlaps(0, abij, QQ0inv0, ovlp_ratios_up);
      }

      {
        // beta
        memory::buffered_array<MEM,ComplexType,3> QQ0inv1(nw,OrbMats(nspin_in_orbmat-1).extent(0),ndown);
        det_ops::Log_OverlapForWoodbury(OrbMats(nspin_in_orbmat-1),SMdn(range(iw,iw+nw),all,all),log_ov,QQ0inv1,abij.get_reference_configuration_device(1));
        calculate_overlaps(1, abij, QQ0inv1, ovlp_ratios_dn);
      }

      // log(Ov(iw)) = log_ov(iw) + std::log( sum_n c(n) * ovlp_ratios(n,iw) ) 
      memory::buffered_array<MEM,ComplexType,2> wgt(nexcit[0], nw);
      // W[u][iw] = sum_d OpSpinDetCouplings[u][d] Ov[d][iw]   
      math::sparse::csrmm<'N'>(OpSpinDetCouplings(0),ovlp_ratios_dn,wgt);
      // 1. Ov(iw) = sum_n wgt(n,iw) * ov_up(n,iw) 
      nda::tensor::contract(ComplexType(1.0),ovlp_ratios_up,"nw",wgt,"nw",ComplexType(0.0),Ov(range(iw,iw+nw)),"w");
      // 2. Ov(iw) = log( Ov(iw) ) = log (sum_n wgt(n,iw) * ov_up(n,iw)) 
      nda::apply(ComplexType(1.0),Ov(range(iw,iw+nw)),nda::tensor::op::LOG);
      // 3. Ov(iw) += log_ov(iw)
      nda::tensor::add(ComplexType(1.0),log_ov,"w",ComplexType(1.0),Ov(range(iw,iw+nw)),"w");
    }
  }
}
/*
template<bool MP>
template<class WlkSet, class TVec, class Mat1, class Mat2, class Mat3, class Observable>
void PHMSD<MP>::accumulate_estimators(int iav, WlkSet& wset, TVec& wgt,
        std::vector<Observable>& properties_1body, std::vector<Observable>& properties,
        Mat1 const& X, Mat2 const& Yc, Mat3 const& M, bool time_evolved, bool importanceSampling)
{
  if(TG.TG_local().size() > 1)
    APP_ABORT(" Error: energy algorithm 1 not yet implemented with ncore>1.\n\n");
  using ma::conj;
  using std::get;
  using std::fill_n;
  using std::copy_n;
  int nspins  = (walker_type==COLLINEAR?2:1);
  int npol((walker_type == NONCOLLINEAR) ? 2 : 1);
  int nwalk = wset.size();
  double LogOverlapFactor(wset.getLogOverlapFactor());

  // what to do!
  bool calculate_G = (properties_1body.size()>0);
  bool loop_over_unique = (properties.size()>0);
  bool loop_over_confgs = false; //(properties_full.size()>0);

  // if only 1body properties, call MixedDensityMatrix 
  if(calculate_G and not loop_over_unique and not loop_over_confgs) {

    StaticSHM4Tensor Gfull({nwalk, nspins, npol*NMO, npol*NMO},
          shm_buffer_manager.get_generator().template get_allocator<ComplexType>());
    // Gfull = M + ma::T(X) * ma::T(OrbMats[spin]) * Gc * conj(Y),
    //   where Yc = conj(Y), Yc already comes with the conjugate!

    int w0, wN;
    std::tie(w0, wN) = FairDivideBoundary(TG.TG_local().rank(), nwalk, TG.TG_local().size());
    TG.TG_local().barrier();

    // 0. Gfull = M
    // assumes M is contiguous
    if(time_evolved)
      ma::copy(M.sliced(w0,wN).flatted().flatted().flatted(),
             Gfull.sliced(w0,wN).flatted().flatted().flatted() );
    TG.TG_local().barrier();

    int nact[2] = {int(OrbMats[0].size(0)), int(OrbMats[1].size(0))};
    StaticSHM3Tensor Gc({nwalk, nact[0]+nact[1], npol*NMO}, 
          shm_buffer_manager.get_generator().template get_allocator<ComplexType>());

    // 1. Calculate compact, mixed density matrix
    MixedDensityMatrix(wset, Gc.rotated().flatted().unrotated(), true,  true);

    for(int ispin=0; ispin<nspins; ispin++) {

      auto Gis=Gc.sliced(w0,wN).rotated().sliced(ispin*nact[0], nact[0]+ispin*nact[1]).unrotated();

      if(time_evolved) {
        StaticSHM3Tensor GYc(Gis.extensions(), 
            shm_buffer_manager.get_generator().template get_allocator<ComplexType>());

        // GYc = Gc * Yc
        ma::productStridedBatched(Gis, Yc.sliced(w0,wN).rotated()[ispin].unrotated(), GYc);

        // reuse Gis: Gis = S * X 
        ma::productStridedBatched(OrbMats[ispin], X.sliced(w0,wN).rotated()[ispin].unrotated(),
                                Gis);

        // Gfull += T(Gis) * GYc
        ma::productStridedBatched(ComplexType(1.0), ma::T(Gis), GYc,
                              ComplexType(1.0), Gfull.sliced(w0,wN).rotated()[ispin].unrotated() );

      } else {

        // Gfull = T(OrbMats) * Gc
        ma::productStridedBatched(ma::T(OrbMats[ispin]), Gis,
                                Gfull.sliced(w0,wN).rotated()[ispin].unrotated() );

      }

    }
    TG.TG_local().barrier();

    // 3. copy to host 
    // trying allocating on the fly, change if too slow!
    long i0, iN;
    std::tie(i0, iN) = FairDivideBoundary(long(TG.TG_local().rank()),
                    long(Gfull.num_elements()), long(TG.TG_local().size()));
    Array<ComplexType, 4, shared_allocator<ComplexType>> G4D_host(Gfull.extensions(),
                                        shared_allocator<ComplexType>{TG.TG_local()});
    copy_n(make_device_ptr(Gfull.origin()) + i0, iN - i0, raw_pointer_cast(G4D_host.origin()) + i0);
    TG.TG_local().barrier();

    // 4. accumulate 
    if(time_evolved) {
      for (auto& v : properties_1body)
        v.accumulate(iav, Gfull, G4D_host, wgt, importanceSampling);
    }
    else {
      for (auto& v : properties_1body)
        v.accumulate(iav, 
                    OrbMats[0],
                    Gc.sliced(w0,wN).rotated().sliced(0, nact[0]).unrotated(),
                    OrbMats[1],
                    Gc.sliced(w0,wN).rotated().sliced(nact[0], nact[0]+nact[1]).unrotated(),    
                    Gfull, G4D_host, wgt, importanceSampling);
    }
    
    return;
  }
  APP_ABORT("Error: PHMSD::accumulate_estimator only working for onerdm, finish! ");

  long nexcit[2] = {long(abij.number_of_unique_excitations()[0]),
                    long(abij.number_of_unique_excitations()[1])};
  std::vector<int> confg(nup);
  IVector confg_dev(iextensions<1u>{nup});

  if (walker_type != COLLINEAR)
  {
    APP_ABORT("Error: Finish implementation of PHMSD for CLOSED/NONCOLLINEAR walkers.");
  }
  else
  {

    // generalize later
    int nbatch__ = nwalk;

    Array<ComplexType, 1> Xw(iextensions<1u>{nwalk});
    Array<ComplexType, 1> scl_wgt(wgt);
    StaticVector Ov(iextensions<1u>{nwalk}, ComplexType(0.0),
                       buffer_manager.get_generator().template get_allocator<ComplexType>());
    StaticSHM3Tensor ph_wgt({2, maxn_unique_confg, nwalk}, ComplexType(0.0),
                       shm_buffer_manager.get_generator().template get_allocator<ComplexType>());
    StaticSHM3Tensor Ovlps({2, maxn_unique_confg, nwalk}, ComplexType(0.0),
                       shm_buffer_manager.get_generator().template get_allocator<ComplexType>());

    auto SMup=wset.SlaterMatrices(Alpha);
    auto SMdn=wset.SlaterMatrices(Beta);

    // 1. calculate Overlaps
    for (int iw = 0; iw < nwalk; iw+=nbatch__)
    {
      int nb = std::min(nbatch__, nwalk - iw);
      StaticVector overlaps(iextensions<1u>{nb}, ComplexType(0.0),
                       buffer_manager.get_generator().template get_allocator<ComplexType>());
      for(int ispin=0; ispin<nspins; ispin++) { 

        Static3Tensor QQ0inv({nb, OrbMats[ispin].size(0), (ispin==0?nup:ndown)}, 
                     buffer_manager.get_generator().template get_allocator<ComplexType>());

        if(ispin==0)
          SDetOp.BatchedOverlapForWoodbury(OrbMats[ispin], SMup.sliced(iw,iw+nb), LogOverlapFactor,
                overlaps, refc_dev.origin() + ispin*nup, QQ0inv);
        else
          SDetOp.BatchedOverlapForWoodbury(OrbMats[ispin], SMdn.sliced(iw,iw+nb), LogOverlapFactor,
                overlaps, refc_dev.origin() + ispin*nup, QQ0inv);
#if defined(ENABLE_CUDA)
        calculate_overlaps(0, 1, ispin, abij, QQ0inv, Ovlps[ispin]({0,nexcit[ispin]},{iw,iw+nb}));
#else
        for(int i=0; i<nb; ++i)
          calculate_overlaps(0, 1, ispin, abij, QQ0inv[i], Ovlps[ispin]({0,nexcit[ispin]}, iw+i));
#endif
        // multiply Ovlps by overlaps
        // Ovlps[s][u][iw] *= overlaps[iw]   
        ma::elementwise(ma::TOp_MUL, 1, overlaps, Ovlps[ispin]({0,nexcit[ispin]}, {iw, iw+nb}));
      } // ispin
    }  // iw

    // 2. calculate full overlap, ph_wgts and normalize walker weights
    // W[u][iw] = sum_d OpSpinDetCouplings[u][d] Ov[d][iw]   
    ma::product(OpSpinDetCouplings[0], Ovlps[1].sliced(0, nexcit[1]), ph_wgt[0].sliced(0, nexcit[0]));
    ma::product(OpSpinDetCouplings[1], Ovlps[0].sliced(0, nexcit[0]), ph_wgt[1].sliced(0, nexcit[1]));
    // W[u][iw] *= Ovlps[0][u][iw]   
    ma::elementwise(ma::TOp_MUL, Ovlps[0].sliced(0,nexcit[0]), ph_wgt[0].sliced(0, nexcit[0]));
    ma::elementwise(ma::TOp_MUL, Ovlps[1].sliced(0,nexcit[1]), ph_wgt[1].sliced(0, nexcit[1]));
    // Ov[iw] = sum_u W[u][iw]  
    ma::accumulate(0, ph_wgt[0].sliced(0, nexcit[0]), Ov);

    // scl_wgt[iw] = wgt[iw] / Ov[iw]
    copy_n(Ov.origin(), nwalk, Xw.origin());
    for(int i=0; i<nwalk; i++)
        scl_wgt[i] /= Xw[i];

    // 4. loop over excitations, calculate Rpa and call properties
    for (int iw = 0; iw < nwalk; iw+=nbatch__)
    { 
      int nb = std::min(nbatch__, nwalk - iw);
      StaticVector overlaps(iextensions<1u>{nb}, ComplexType(0.0),
                       buffer_manager.get_generator().template get_allocator<ComplexType>());
      for(int ispin=0; ispin<nspins; ispin++) {
        
        auto& SM = ( (ispin==0) ? SMup : SMdn );
        Static3Tensor GA({nb, SM.size(2), npol*NMO},
                       buffer_manager.get_generator().template get_allocator<ComplexType>());
        Static3Tensor QQ0inv({nb, OrbMats[ispin].size(0), SM.size(2)}, 
                     buffer_manager.get_generator().template get_allocator<ComplexType>());
        
        SDetOp.BatchedMixedDensityMatrixForWoodbury(OrbMats[ispin], SM.sliced(iw,iw+nb), GA,
            LogOverlapFactor, overlaps, refc_dev.origin() + ispin*nup, QQ0inv, true);

/ *
         //
         if( ispin==0 ) {
           ph_excited_properties_first_step(nup, OrbMats[0].size(0), abij, ph_wgt[0],
                    QQ0inv, properties);       
         } else {
           ph_excited_properties_second_step(ndown, OrbMats[1].size(0), abij, ph_wgt[1],
                    QQ0inv, properties);       
         }
* /

      } // ispin

      // properties.ph_complete?

    }  // iw

  }
}
*/
/*
 * Calculate mean field Green function 
 */
template<MEMORY_SPACE MEM>
memory::const_shared_array<HOST_MEMORY,ComplexType,3> PHMSD<MEM>::G_MF()
{
  constexpr MEMORY_SPACE M = HOST_MEMORY;
  using nda::range;
  auto all = range::all;
  int nspin  = (walker_type==COLLINEAR?2:1);
  int npol    = (walker_type==NONCOLLINEAR?2:1);
  int nelec[2] = { nup, (walker_type==COLLINEAR ? ndown : 0) };

  long nexcit[2] = {long(abij.number_of_unique_excitations()[0]),
              (walker_type==COLLINEAR ? long(abij.number_of_unique_excitations()[1]) : 0)};
  long maxn_unique_confg = std::max(nexcit[0],nexcit[1]);
  int nspin_in_orbmat = OrbMats.extent(0);
  long nact[2] = {OrbMats(0).extent(0),
                 ( walker_type==COLLINEAR ? OrbMats(nspin_in_orbmat-1).extent(0) : 0)};

  std::vector<int> exct(2 * nup);
  std::vector<int> Iwork(2 * nup);
  nda::array<int,1> confg(nup);
  nda::array<int,1> confgB(nup);

  auto Orbs = dense_orbs();

  memory::array<M,ComplexType,3> A(1,npol*NMO,nup);
  auto A2d = nda::reshape(A,std::array<long,2>{npol*NMO,nup});

  // 1. Overlaps
  memory::buffered_array<M,ComplexType,2> ovlps(nspin,maxn_unique_confg);
  ovlps() = ComplexType(0.0);
  for (int spin = 0, nc = 0; spin < nspin; ++spin)
  {
    confg.resize((spin == 0) ? nup : ndown);
    auto As = A(all,all,range(nelec[spin]));
    auto As2d = A2d(all,range(nelec[spin]));
    for (int nd = 0; nd < int(abij.number_of_unique_excitations()[spin]); ++nd, ++nc)
    {
      if(nc%mpi->comm.size() != mpi->comm.rank()) continue;
      abij.get_configuration(spin, nd, confg);
      // A(:,i) = Orbs[spin](:,confg(i))
      nda::copy_select(false,1,confg,ComplexType(1.0),Orbs[spin%nspin_in_orbmat](),ComplexType(0.0),As2d);
      det_ops::Log_Overlap(As2d,As,ovlps(spin,range(nd,nd+1)),false);
    }
  }
  mpi->all_reduce(ovlps,std::plus<>{});

  // log(Ov(n)) -> Ov(n)/Ov(n=0)
  for(int is=0; is<nspin; ++is) {
    for(int d=1; d<nexcit[is]; ++d)
      ovlps(is,d) = std::exp(ovlps(is,d) - ovlps(is,0));
    ovlps(is,0) = 1.0;
  }

  // Overlap 
  ComplexType ov(0.0);
  if(walker_type == COLLINEAR)
    for (auto it = abij.configurations_begin(); it < abij.configurations_end(); ++it)
      ov += std::norm(std::get<2>(*it)) * ovlps(0,std::get<0>(*it)) * ovlps(1,std::get<1>(*it));
  else
    for (auto it = abij.configurations_begin(); it < abij.configurations_end(); ++it)
      ov += std::norm(std::get<2>(*it)) * ovlps(0,std::get<0>(*it));

  memory::buffered_array<M,ComplexType,2> Gfull(nact[0]+nact[1],npol*NMO);
  // 2. Diagonal and off-diagonal components
  for (int spin = 0, is0=0; spin < nspin; ++spin, is0 += nact[0])
  {
    int other_spin    = 1 - spin;
    confg.resize((spin == 0) ? nup : ndown);

    memory::buffered_array<M,ComplexType,1> ova(1,ComplexType(0.0));
    memory::array<M,ComplexType,3> G3d(1,nelec[spin],npol*NMO);
    auto As = A(all,all,range(nelec[spin]));
    auto As2d = A2d(all,range(nelec[spin]));
    auto G2d = nda::reshape(G3d,std::array<long,2>{nelec[spin],npol*NMO});
    auto Gs = Gfull(range(is0, is0+nact[spin]), all);

    // diagonal contribution
    for (int nd = 0; nd < int(abij.number_of_unique_excitations()[spin]); ++nd)
    {
      if (nd % mpi->comm.size() == mpi->comm.rank())
      {
        abij.get_configuration(spin, nd, confg);
        // A(:,i) = Orbs[spin](:,confg(i))
        nda::copy_select(false,1,confg,ComplexType(1.0),Orbs[spin%nspin_in_orbmat](),ComplexType(0.0),As2d);
        det_ops::MixedDensityMatrix(As2d,As,G3d,ova,true,false);
        ComplexType wgt(0.0);
        if( walker_type == COLLINEAR )
        {
          auto D = OpSpinDetCouplings[spin][nd];
          long nnz = D.nnz();
          auto val = nda::to_host(D.values());
          auto jdet = nda::to_host(D.columns());
          for (size_t n=0; n<nnz; ++n)
            wgt += ovlps(other_spin,jdet(n)) * std::norm(val(n));
          wgt *= ovlps(spin,nd);
        } else {
          auto D = OpSpinDetCouplings[1][nd];
          auto val = nda::to_host(D.values());
          //wgt = ovlps(spin,nd) * ComplexType(OpSpinDetCouplings[1].values(nd));
          wgt = ovlps(spin,nd) * ComplexType(val(0));
        }
        nda::copy_select(true,0,confg,wgt,G2d,ComplexType(1.0),Gs);
      }
    }

    // off-diagonal contribution
    ComplexType dummy(0.0);
    confgB.resize((spin == 0) ? nup : ndown);
    if(walker_type == COLLINEAR)
    {
      for (int nd = 0; nd < int(abij.number_of_unique_excitations()[other_spin]); ++nd)
      {
        if (nd % mpi->comm.size() == mpi->comm.rank())
        {
          auto D = OpSpinDetCouplings[other_spin][nd];
          long nnz = D.nnz();
          auto val = nda::to_host(D.values());
          auto jdet = nda::to_host(D.columns());
          for (long n1=0; n1<nnz; ++n1)
          {
            long cf1 = long(jdet(n1));
            abij.get_configuration(spin, cf1, confg);
            std::sort(confg.begin(), confg.end());
            for (long n2=n1+1; n2<nnz; ++n2)
            {
              long cf2 = long(jdet[n2]);
              abij.get_configuration(spin, cf2, confgB);
              std::sort(confgB.begin(), confgB.end());
              exct.clear();
              int np = get_excitation_number(true, confg, confgB, exct, dummy, Iwork);
              if (np == 1)
              {
                ComplexType wgt = val(n1) * std::conj(val(n2));
                /*
                 * exct: [0]: location of orbital being excited, [1]: excited orbital 
                 * confg[exct[0]]: occupied orbital being excited
                 * WARNING!!! Assumes orthogonal states, needs overlap factor!!!
                 * Either calculate it (expensive) or demand orthogonality!!!   
                 */
                exct[0] = confg[exct[0]];
                Gs(exct[1],all) += wgt*nda::conj(Orbs[spin%nspin_in_orbmat]()(all,exct[0]));
                Gs(exct[0],all) += std::conj(wgt)*nda::conj(Orbs[spin%nspin_in_orbmat]()(all,exct[1]));
              }
            }
          }
        }
      }
    } else {
//MAM: needs checking!!!
      auto D = OpSpinDetCouplings[1][0];
      long nunique = D.nnz();
      auto val = nda::to_host(D.values());
      for (size_t n1=0, n12=0; n1<nunique; ++n1)
      {
        abij.get_configuration(spin, n1, confg);
        std::sort(confg.begin(), confg.end());
        for (size_t n2=n1+1; n2<nunique; ++n2, ++n12)
        {
          if (n12 % mpi->comm.size() != mpi->comm.rank()) continue;
          abij.get_configuration(spin, n2, confgB);
          std::sort(confgB.begin(), confgB.end());
          exct.clear();
          int np = get_excitation_number(true, confg, confgB, exct, dummy, Iwork);
          if (np == 1)
          {
            ComplexType wgt = val(n1) * std::conj(val(n2));
            /*
             * exct: [0]: location of orbital being excited, [1]: excited orbital 
             * confg[exct[0]]: occupied orbital being excited
             * WARNING!!! Assumes orthogonal states, needs overlap factor!!!
             * Either calculate it (expensive) or demand orthogonality!!!   
             */
            exct[0] = confg[exct[0]];
            Gs(exct[1],all) += wgt*nda::conj(Orbs[spin%nspin_in_orbmat]()(all,exct[0]));
            Gs(exct[0],all) += std::conj(wgt)*nda::conj(Orbs[spin%nspin_in_orbmat]()(all,exct[1]));
          } // np==1
        } // n2
      } // n1
    } // walker_type == COLLINEAR
  }

  mpi->all_reduce(Gfull,std::plus<>{});
  Gfull() *= (1.0/ov);
  return memory::share_from_root(*mpi, [&] {
    nda::array<ComplexType,3> gMF(nspin,npol*NMO,npol*NMO);
    gMF() = ComplexType(0.0);
    if constexpr (MEM==HOST_MEMORY) {
      math::sparse::csrmm<'T'>(OrbMats(0),Gfull(range(nact[0]),all),gMF(0,all,all));
      if(walker_type==COLLINEAR) {
        math::sparse::csrmm<'T'>(OrbMats(nspin_in_orbmat-1),Gfull(range(nact[0],nact[0]+nact[1]),all),gMF(1,all,all));
      }
    } else {
      memory::array<MEM,ComplexType,2> Gdev(Gfull);
      memory::array<MEM,ComplexType,2> gt(npol*NMO,npol*NMO);
      math::sparse::csrmm<'T'>(OrbMats(0),Gdev(range(nact[0]),all),gt);
      gMF(0,all,all) = gt();
      if(walker_type==COLLINEAR) {
        math::sparse::csrmm<'T'>(OrbMats(nspin_in_orbmat-1),Gdev(range(nact[0],nact[0]+nact[1]),all),gt);
        gMF(1,all,all) = gt();
      }
    }
    return gMF;
  });
}

/*
 * Calculate mean field expectation value of Cholesky potentials
 * This is a collective call.
 */ 
template<MEMORY_SPACE MEM>
void PHMSD<MEM>::vMF(memory::array_view<MEM,ComplexType,1> v, double dt)
{
  // Implementing this routine on HOST_MEMORY regardless of MEM!!!
  // Move to device if it is too slow!!!
  constexpr MEMORY_SPACE M = HOST_MEMORY; 
  using nda::range;
  auto all = range::all;
  utils::check(v.size() == number_of_cholesky_vectors(), "Size mismatch");
  utils::check(v.strides()[0] == 1, "Strides mismatch");
  auto v2d = nda::reshape(v,std::array<long,2>{1,v.size()});
  v() = ComplexType(0.0);

  // collective call!!!
  mpi->comm.barrier();

  int nspin_in_orbmat = OrbMats.extent(0);
  int nspin = (walker_type==COLLINEAR ? 2 : 1);
  int npol  = (walker_type==NONCOLLINEAR ? 2 : 1);
  int nelec[2] = { nup, (walker_type==COLLINEAR ? ndown : 0) };

  long nexcit[2] = {long(abij.number_of_unique_excitations()[0]), 
              (walker_type==COLLINEAR ? long(abij.number_of_unique_excitations()[1]) : 0)};
  long maxn_unique_confg = std::max(nexcit[0],nexcit[1]);
  long nact[2] = {OrbMats(0).extent(0),
                 ( walker_type==COLLINEAR ? OrbMats(nspin_in_orbmat-1).extent(0) : 0)}; 

  std::vector<int> exct(2 * nup);
  std::vector<int> Iwork(2 * nup);
  nda::array<int,1> confg(nup);
  nda::array<int,1> confgB(nup);

  auto Orbs = dense_orbs();

  memory::array<M,ComplexType,3> A(1,npol*NMO,nup);
  auto A2d = nda::reshape(A,std::array<long,2>{npol*NMO,nup});

  // 1. Overlaps
  memory::buffered_array<M,ComplexType,2> ovlps(nspin,maxn_unique_confg);
  ovlps() = ComplexType(0.0);
  for (int spin = 0, nc = 0; spin < nspin; ++spin)
  {
    confg.resize((spin == 0) ? nup : ndown);
    auto As = A(all,all,range(nelec[spin]));
    auto As2d = A2d(all,range(nelec[spin]));
    for (int nd = 0; nd < int(abij.number_of_unique_excitations()[spin]); ++nd, ++nc)
    {
      if(nc%mpi->comm.size() != mpi->comm.rank()) continue;
      abij.get_configuration(spin, nd, confg);
      // A(:,i) = Orbs[spin](:,confg(i))
      nda::copy_select(false,1,confg,ComplexType(1.0),Orbs[spin%nspin_in_orbmat](),ComplexType(0.0),As2d);
      det_ops::Log_Overlap(As2d,As,ovlps(spin,range(nd,nd+1)),false);
    }
  }
  mpi->all_reduce(ovlps,std::plus<>{});

  // log(Ov(n)) -> Ov(n)/Ov(n=0)
  for(int is=0; is<nspin; ++is) {
    for(int d=1; d<nexcit[is]; ++d)
      ovlps(is,d) = std::exp(ovlps(is,d) - ovlps(is,0));
    ovlps(is,0) = 1.0;
  }

  // Overlap 
  ComplexType ov(0.0);
  if(walker_type == COLLINEAR) 
    for (auto it = abij.configurations_begin(); it < abij.configurations_end(); ++it)
      ov += std::norm(std::get<2>(*it)) * ovlps(0,std::get<0>(*it)) * ovlps(1,std::get<1>(*it));
  else
    for (auto it = abij.configurations_begin(); it < abij.configurations_end(); ++it)
      ov += std::norm(std::get<2>(*it)) * ovlps(0,std::get<0>(*it)); 

  memory::buffered_array<M,ComplexType,2> Gfull(nact[0]+nact[1],npol*NMO);
  // 2. Diagonal and off-diagonal components
  for (int spin = 0, is0=0; spin < nspin; ++spin, is0 += nact[0])
  {
    int other_spin    = 1 - spin;
    confg.resize((spin == 0) ? nup : ndown);

    memory::buffered_array<M,ComplexType,1> ova(1,ComplexType(0.0));
    memory::array<M,ComplexType,3> G3d(1,nelec[spin],npol*NMO);
    auto As = A(all,all,range(nelec[spin]));
    auto As2d = A2d(all,range(nelec[spin]));
    auto G2d = nda::reshape(G3d,std::array<long,2>{nelec[spin],npol*NMO});
    auto Gs = Gfull(range(is0, is0+nact[spin]), all);
    // diagonal contribution
    for (int nd = 0; nd < int(abij.number_of_unique_excitations()[spin]); ++nd)
    {
      if (nd % mpi->comm.size() == mpi->comm.rank())
      {
        abij.get_configuration(spin, nd, confg);
        // A(:,i) = Orbs[spin](:,confg(i))
        nda::copy_select(false,1,confg,ComplexType(1.0),Orbs[spin%nspin_in_orbmat](),ComplexType(0.0),As2d);
        det_ops::MixedDensityMatrix(As2d,As,G3d,ova,true,false);
        ComplexType wgt(0.0);
        if( walker_type == COLLINEAR ) 
	{
          auto D = OpSpinDetCouplings[spin][nd];     
          long nnz = D.nnz();
	  auto val = nda::to_host(D.values()); 
	  auto jdet = nda::to_host(D.columns());
          for (size_t n=0; n<nnz; ++n)
            wgt += ovlps(other_spin,jdet(n)) * std::norm(val(n));
          wgt *= ovlps(spin,nd);
	} else {
          auto D = OpSpinDetCouplings[1][nd];     
	  auto val = nda::to_host(D.values()); 
	  //wgt = ovlps(spin,nd) * ComplexType(OpSpinDetCouplings[1].values(nd));
	  wgt = ovlps(spin,nd) * ComplexType(val(0));
        }
        nda::copy_select(true,0,confg,wgt,G2d,ComplexType(1.0),Gs);
      }
    }

    // off-diagonal contribution
    ComplexType dummy(0.0);
    confgB.resize((spin == 0) ? nup : ndown);
    if(walker_type == COLLINEAR)
    {
#if defined(ENABLE_DEVICE)
      if constexpr (MEM == DEVICE_MEMORY)
      {
        // GPU path: replace the O(sum nnz^2) host pair loop over excitation pairs with a
        // device kernel. Build sorted occupied-orbital configurations for `spin` on host
        // and copy once to device; feed the device-resident coupling CSR + dense orbitals.
        long nex_spin = nexcit[spin];
        int  nel_spin = nelec[spin];
        nda::array<int,2> confg_h(nex_spin, nel_spin);
        nda::array<int,1> cf(nel_spin);
        for (long c = 0; c < nex_spin; ++c) {
          abij.get_configuration(spin, c, cf);
          std::sort(cf.begin(), cf.end());
          for (int k = 0; k < nel_spin; ++k) confg_h(c,k) = cf(k);
        }
        memory::array<DEVICE_MEMORY,int,2> configs_d = confg_h;
        memory::array<DEVICE_MEMORY,ComplexType,2> Orbs_d = Orbs[spin%nspin_in_orbmat]();
        auto& csr = OpSpinDetCouplings[other_spin];
        long nrows = nexcit[other_spin];
        auto rbeg_h = csr.row_begin();
        auto rend_h = csr.row_end();
        // per-row upper-triangular pair count -> prefix sum (flattened pair index space)
        nda::array<long,1> pair_off_h(nrows+1);
        pair_off_h(0) = 0;
        for (long nd = 0; nd < nrows; ++nd) {
          long z = long(rend_h(nd)) - long(rbeg_h(nd));
          pair_off_h(nd+1) = pair_off_h(nd) + z*(z-1)/2;
        }
        long total_pairs = pair_off_h(nrows);
        memory::array<DEVICE_MEMORY,long,1> pair_off_d = pair_off_h;
        memory::array<DEVICE_MEMORY,int,1> rbeg_d = rbeg_h;
        memory::array<DEVICE_MEMORY,int,1> rend_d = rend_h;
        memory::array<DEVICE_MEMORY,ComplexType,2> Gs_d(nact[spin], npol*NMO);
        Gs_d() = ComplexType(0.0);
        kernels::device::vmf_offdiag(total_pairs, int(nrows), mpi->comm.rank(), mpi->comm.size(),
                                     nel_spin, pair_off_d.data(), rbeg_d.data(), rend_d.data(),
                                     csr.columns().data(), csr.values().data(),
                                     configs_d.data(), Orbs_d, Gs_d);
        nda::array<ComplexType,2> Gs_h = Gs_d;
        Gs += Gs_h;
      }
      else
#endif
      for (int nd = 0; nd < int(abij.number_of_unique_excitations()[other_spin]); ++nd)
      {
        if (nd % mpi->comm.size() == mpi->comm.rank())
        {
          auto D = OpSpinDetCouplings[other_spin][nd];     
          long nnz = D.nnz();
          auto val = nda::to_host(D.values());
          auto jdet = nda::to_host(D.columns());
          for (long n1=0; n1<nnz; ++n1)  
          {
            long cf1 = long(jdet(n1)); 
            abij.get_configuration(spin, cf1, confg);
            std::sort(confg.begin(), confg.end());
            for (long n2=n1+1; n2<nnz; ++n2)
            {
              long cf2 = long(jdet[n2]); 
              abij.get_configuration(spin, cf2, confgB);
              std::sort(confgB.begin(), confgB.end());
              exct.clear();
              int np = get_excitation_number(true, confg, confgB, exct, dummy, Iwork);
              if (np == 1)
              {
                ComplexType wgt = val(n1) * std::conj(val(n2)); 
                /*
                 * exct: [0]: location of orbital being excited, [1]: excited orbital 
                 * confg[exct[0]]: occupied orbital being excited
                 * WARNING!!! Assumes orthogonal states, needs overlap factor!!!
                 * Either calculate it (expensive) or demand orthogonality!!!   
                 */
                exct[0] = confg[exct[0]];
                Gs(exct[1],all) += wgt*nda::conj(Orbs[spin%nspin_in_orbmat]()(all,exct[0])); 
                Gs(exct[0],all) += std::conj(wgt)*nda::conj(Orbs[spin%nspin_in_orbmat]()(all,exct[1])); 
              }
            }
          }
        }
      }
    } else {
//MAM: needs checking!!!
      auto D = OpSpinDetCouplings[1][0];     
      long nunique = D.nnz();
      auto val = nda::to_host(D.values()); 
      for (size_t n1=0, n12=0; n1<nunique; ++n1)
      {
        abij.get_configuration(spin, n1, confg);
        std::sort(confg.begin(), confg.end());
        for (size_t n2=n1+1; n2<nunique; ++n2, ++n12)
        {
          if (n12 % mpi->comm.size() != mpi->comm.rank()) continue;
          abij.get_configuration(spin, n2, confgB);
          std::sort(confgB.begin(), confgB.end());
          exct.clear();
          int np = get_excitation_number(true, confg, confgB, exct, dummy, Iwork);
          if (np == 1)
          {
            ComplexType wgt = val(n1) * std::conj(val(n2));
            /*
             * exct: [0]: location of orbital being excited, [1]: excited orbital 
             * confg[exct[0]]: occupied orbital being excited
             * WARNING!!! Assumes orthogonal states, needs overlap factor!!!
             * Either calculate it (expensive) or demand orthogonality!!!   
             */
            exct[0] = confg[exct[0]];
            Gs(exct[1],all) += wgt*nda::conj(Orbs[spin%nspin_in_orbmat]()(all,exct[0])); 
            Gs(exct[0],all) += std::conj(wgt)*nda::conj(Orbs[spin%nspin_in_orbmat]()(all,exct[1])); 
          } // np==1
        } // n2
      } // n1
    } // walker_type == COLLINEAR
  }
  mpi->all_reduce(Gfull,std::plus<>{}); 
  Gfull() *= (1.0/ov);
  if constexpr (MEM==HOST_MEMORY) {
    auto G2d = nda::reshape(Gfull,std::array<long,2>{1,(nact[0]+nact[1])*npol*NMO});
    HamOp.vbias(G2d, v2d, dt);
  } else {
    memory::array<MEM,ComplexType,2> Gdev(Gfull);
    auto G2d = nda::reshape(Gdev,std::array<long,2>{1,(nact[0]+nact[1])*npol*NMO});
    HamOp.vbias(G2d, v2d, dt);
  }
  mpi->comm.barrier();
}

template class PHMSD<HOST_MEMORY>;
#if defined(ENABLE_DEVICE)
template class PHMSD<DEVICE_MEMORY>;
#endif

} // namespace afqmc
} // namespace sfqmc
