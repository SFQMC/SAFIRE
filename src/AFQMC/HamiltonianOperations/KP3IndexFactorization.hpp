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

#pragma once

#include "configuration.hpp"
#include "nda/nda.hpp"
#include "nda/nda.hpp"
#include "nda/tensor.hpp"
#include "utilities/check.hpp"
#include "utilities/check_shape.hpp"
#include "utilities/mpi_context.h"
#include "utilities/check_strides.hpp"
#include "utilities/memory_utils.hpp"
#include "numerics/shared_array/const_shared_array.hpp"
#include "numerics/nda_functions.hpp"
#include "numerics/operations/tensor.hpp"
#include "detail/one_body.hpp"
#include "AFQMC/Utilities/wfn_utils.hpp"

namespace sfqmc
{
namespace afqmc
{

template<MEMORY_SPACE MEM>
class KP3IndexFactorization
{
  template<class T>
  using csrMat = math::sparse::csr_matrix<T, MEM, int, int>;

public:
  HamiltonianType getHamType() const { return HamiltonianType::kp_factorized; }

  // since arrays can be in host, can't assume that types are consistent
  KP3IndexFactorization(std::shared_ptr<utils::mpi_context_t<mpi3::communicator>> _mpi,
          WALKER_TYPES type,
          int nbnd_,
          int q0,
          nda::array<nda::array<int,1>,2>&& nocc_,
          nda::array<int,1>&& minusq_,
          nda::array<int,2>&& qk_to_k2_,
          nda::array<int,1>&& qmap_,
          memory::const_shared_array<HOST_MEMORY,ComplexType,4>&& hij_,
          memory::const_shared_array<MEM,ComplexType,3>&& haj_,
          nda::array<memory::const_shared_array<MEM,ComplexType,6>,1>&& lq_,
          nda::array<memory::const_shared_array<MEM,ComplexType,6>,1>&& la_,
          nda::array<memory::const_shared_array<MEM,ComplexType,6>,1>&& lb_,
          memory::const_shared_array<HOST_MEMORY,ComplexType,4>&& vexx_,
          RealType e0_,
          int bf_size = 4096)
      : mpi(_mpi),
        walker_type(type),
        nkpts(nocc_.extent(1)),
        nbnd(nbnd_),
        nup(nelec_for_spin(nocc_,0)),
        ndown(walker_type==COLLINEAR ? nelec_for_spin(nocc_,1) : 0),
        nocc_max(max_nocc_per_kpoint(nocc_)),
        nchol_max(0),
        nocc(std::move(nocc_)),
        minusq(std::move(minusq_)),
        qk_to_k2(std::move(qk_to_k2_)),
        Qmap(std::move(qmap_)),
        ncv0(nkpts),
        Q0_index(q0),
        hij(std::move(hij_)),
        haj(std::move(haj_)),
        LQ(std::move(lq_)),
        Lank(std::move(la_)),
        Lakn(std::nullopt),
        Lbkn(std::nullopt),
        Lbnk(std::move(lb_)),
        vexx(std::move(vexx_)),
        default_buffer_size_in_MB(bf_size),
        E0(e0_)
  {
    // setup
    int nspin  = (walker_type == COLLINEAR) ? 2 : 1;
    int npol  = (walker_type == NONCOLLINEAR) ? 2 : 1;
    int NMO = nkpts*nbnd;  
    int nspin_in_H = hij.extent(0);
    int npol_in_H = hij.extent(2)/nbnd;
    int ndet = haj.extent(0);
    
    // check sizes
    utils::check_shape(hij, "hij", nspin_in_H, nkpts, npol_in_H*nbnd, npol_in_H*nbnd);
    utils::check_shape(haj, "haj", ndet, nup+ndown, npol*NMO);
    utils::check_shape(vexx, "vexx", nspin_in_H*npol_in_H, nkpts, nbnd, nbnd);
    utils::check_shape(LQ, "LQ", nkpts);
    utils::check_shape(Lank, "Lank", nkpts);
    for(int iq=0; iq<nkpts; ++iq) {
      if(iq<=minusq(iq)) {
        utils::check_shape(LQ(iq), std::format("LQ({})", iq),
                           nspin_in_H, npol_in_H, nkpts, nbnd, nbnd, LQ(iq).extent(5));
      }
      utils::check_shape(Lank(iq), std::format("Lank({})", iq),
                         ndet, nspin, nkpts, nocc_max, Lank(iq).extent(4), npol*nbnd);
      if(iq==minusq(iq)) {
        utils::check_shape(Lbnk(Qmap(iq)), std::format("Lbnk({})", Qmap(iq)),
                           ndet, nspin, nkpts, nocc_max, Lbnk(Qmap(iq)).extent(4), npol*nbnd);
      }
    } // iq

    ncvecs = 0;
    for(int iq=0; iq<nkpts; ++iq)        
    { 
      ncv0(iq) = ncvecs;
      ncvecs += LQ(std::min(iq,minusq(iq))).extent(5);
      nchol_max = std::max(nchol_max,int(LQ(std::min(iq,minusq(iq))).extent(5)));
    }

    auto [gkk_bytes, slot_bytes] = exchange_bytes_per_walker();
    app_log(2, "KP3IndexFactorization: exchange work arrays per walker: {}, buffer budget: {}",
            utils::format_bytes(gkk_bytes + slot_bytes),
            utils::format_bytes(long(default_buffer_size_in_MB) * 1024L * 1024L));

    // setup Lakn if needed
    if(ndet == 1) {
      Lakn = std::make_optional<nda::array<memory::const_shared_array<MEM,ComplexType,5>,1>>(nkpts);
      for(int iq=0; iq<nkpts; ++iq) {
        int nc = Lank(iq).extent(4);
        (*Lakn)(iq) = memory::share_from_ranks<MEM,ComplexType,5,2>(*mpi,
            {nspin,nkpts,nocc_max,npol*nbnd,nc},
            [&](std::array<long,2> idx, auto&& block) {
          auto [is,ik] = idx;
          nda::tensor::add(ComplexType(1.0),Lank(iq)()(0,is,ik,nda::ellipsis{}),"anj",
                           ComplexType(0.0),block,"ajn");
        });
      }
      int nsymQ = Lbnk.extent(0);
      Lbkn = std::make_optional<nda::array<memory::const_shared_array<MEM,ComplexType,5>,1>>(nsymQ);
      for(int i=0; i<nsymQ; ++i) {
        int nc = Lbnk(i).extent(4);
        (*Lbkn)(i) = memory::share_from_ranks<MEM,ComplexType,5,2>(*mpi,
            {nspin,nkpts,nocc_max,npol*nbnd,nc},
            [&](std::array<long,2> idx, auto&& block) {
          auto [is,ik] = idx;
          nda::tensor::add(ComplexType(1.0),Lbnk(i)()(0,is,ik,nda::ellipsis{}),"anj",
                           ComplexType(0.0),block,"ajn");
        });
      }
    }
  }

  nda::array<ComplexType,3> getOneBodyPropagatorMatrix(double dt,
                                                       nda::MemoryVector auto const& vMF)
  {
    using nda::range;
    auto all = range::all;
    int nspin  = (walker_type == COLLINEAR) ? 2 : 1;
    int npol  = (walker_type == NONCOLLINEAR) ? 2 : 1;
    int NMO = nkpts*nbnd;
    int nspin_in_H = hij.extent(0);
    int npol_in_H = hij.extent(2)/nbnd;

    nda::array<ComplexType, 3> H1(nspin, npol*NMO, npol*NMO);
    H1() = ComplexType(0.0);

    memory::buffered_array<MEM,ComplexType,2> vMF_2d(1,vMF.size());
    vMF_2d(0,all) = vMF();
    auto meanfield_shift{nda::to_host(vHS(vMF_2d, dt))};

    memory::buffered_array<HOST_MEMORY,ComplexType,7> hij_plus_v(nspin_in_H, npol_in_H, nkpts, nbnd, npol_in_H, nkpts, nbnd);
    kpoint_add_one_body_shifts(dt, hij(), meanfield_shift(), vexx(), hij_plus_v());
    broadcast_one_body(
        nda::reshape(hij_plus_v, nspin_in_H, npol_in_H, NMO, npol_in_H, NMO),
        nda::reshape(H1, nspin, npol, NMO, npol, NMO));

    return H1;
  }

  void runtime_optimization(nda::MemoryArrayOfRank<2> auto const& G)
  {  /*nothing to do right now*/ }

  nda::array<int,1> getFieldTypes() const {
    int nvc = number_of_cholesky_vectors();
    nda::array<int,1> v(nvc, int(ContinuousChargePropagator));
    return v;
  }

  // nothing to update 
  template<class... Args> void update_potentials([[maybe_unused]] Args&&... args) {}

  void energy(nda::MemoryArrayOfRank<2> auto && E,
              nda::MemoryArrayOfRank<2> auto const& G,
              int idet,
              bool addH1  = true,
              bool addEJ  = true,
              bool addEXX = true)
  {
    memory::check_memory_space<MEM>(E,G);
    using nda::range;
    auto all = range::all;
    ComplexType one(1.0), zero(0.0);
    int nwalk = G.extent(0);
    int nspin  = (walker_type == COLLINEAR) ? 2 : 1;
    int npol  = (walker_type == NONCOLLINEAR) ? 2 : 1;
    int nel  = (walker_type == COLLINEAR ? nup+ndown : nup); // NONCOLLINEAR has ndown=0
    utils::check_shape(E, "E", nwalk, 3);
    utils::check_shape(G, "G", nwalk, nel*npol*nkpts*nbnd);
    utils::check(idet>=0 and idet<haj.extent(0), "Invalid idet:{}",idet);
    utils::check_strides(E,G);
    ComplexType scl = (walker_type == CLOSED ? ComplexType(2.0) : ComplexType(1.0));

    // addH1
    E() = ComplexType(0.0);
    if (addH1)
    {
      E(all,0) = E0;
      auto haj_2d = nda::reshape(haj(),std::array<long,2>{haj.extent(0),haj.extent(1)*haj.extent(2)});
      nda::tensor::contract(scl, G, "wi", haj_2d(idet,all), "i", one, E(all,0), "w");
    }
    if (not(addEJ || addEXX))
      return;

    utils::check(G.is_contiguous(), "Layout mismatch");
    memory::array_view<MEM,const ComplexType,5> G5d(std::array<long,5>{nwalk,nel,npol,nkpts,nbnd},G.data());

    if(addEXX) {
      for(int is=0; is<nspin; ++is) {
        exchange_energy(is, is*nup, idet, G5d, E(all,1));
      }
    }
    if(addEJ) {
      memory::buffered_array<MEM,ComplexType,2> Kleft(nwalk,ncvecs);
      memory::buffered_array<MEM,ComplexType,2> Kright(nwalk,ncvecs);
      Kleft() = zero;
      Kright() = zero;
      for(int is=0; is<nspin; ++is) {
        coulomb_intermediates(is, is*nup, G5d, Kleft, Kright);
      }
      nda::tensor::contract(ComplexType(0.5*scl*scl),Kleft,"wn",Kright,"wn",zero,E(all,2),"w");
    }
  }

  void energy(SpinTypes spin_component,
              nda::MemoryArrayOfRank<2> auto && E,
              nda::MemoryArrayOfRank<2> auto const& G,
              int idet,
              nda::MemoryArrayOfRank<2> auto && EJn,
              bool addH1  = true,
              bool addEJ  = true,
              bool addEXX = true)
  {
    memory::check_memory_space<MEM>(E,G);
    using nda::range;
    auto all = range::all;
    ComplexType one(1.0), zero(0.0);
    int is = (spin_component == Alpha ? 0 : 1);
    int nwalk = G.extent(0);
    int npol  = (walker_type == NONCOLLINEAR) ? 2 : 1;
    int nel  = (is==0 ? nup : ndown);
    utils::check_shape(E, "E", nwalk, 3);
    utils::check_shape(G, "G", nwalk, nel*npol*nkpts*nbnd);
    utils::check(idet>=0 and idet<haj.extent(0), "Invalid idet:{}",idet);
    utils::check_strides(E,G);
    ComplexType scl = (walker_type == CLOSED ? ComplexType(2.0) : ComplexType(1.0));

    // addH1
    E() = ComplexType(0.0);
    if (addH1)
    {
      if(spin_component==Alpha)
        E(all,0) = E0;
      auto haj_2d = nda::reshape(haj(),std::array<long,2>{haj.extent(0),haj.extent(1)*haj.extent(2)});
      long n0 = (spin_component==Alpha ? 0l : long(nup));
      long nx = npol*nkpts*nbnd;
      nda::tensor::contract(scl, G, "wi", haj_2d(idet,range(n0*nx,(n0+nel)*nx)), "i", 
                            one, E(all,0), "w");
    }
    if (not(addEJ || addEXX))
      return;

    utils::check(G.is_contiguous(), "Layout mismatch");
    memory::array_view<MEM,const ComplexType,5> G5d(std::array<long,5>{nwalk,nel,npol,nkpts,nbnd},G.data());

    if(addEXX) {
      exchange_energy(is, 0, idet, G5d, E(all,1));
    }
    if(addEJ) {
      utils::check_shape(EJn, "EJn", nwalk, 2*ncvecs);
      EJn() = zero;
      // Alpha stores [Kleft, Kright], Beta [Kright, Kleft]
      auto first  = range(0, ncvecs);
      auto second = range(ncvecs, 2*ncvecs);
      auto Kleft  = EJn(all, spin_component == Alpha ? first : second);
      auto Kright = EJn(all, spin_component == Alpha ? second : first);
      coulomb_intermediates(is, 0, G5d, Kleft, Kright);
      nda::tensor::contract(ComplexType(0.5),Kleft,"wn",Kright,"wn",zero,E(all,2),"w");
      // gets multiplied by 2 in PHMSD, so undo it here!
      nda::tensor::scale(ComplexType(1.0/std::sqrt(2.0)),EJn);
    }
  }

  template<class... Args>
  void ph_reference_energy([[maybe_unused]] Args&&... args)
  {
    APP_ABORT(" Error: ph_reference_energy not implemented yet. ");
  }

  template<class... Args>
  void ph_excited_energy([[maybe_unused]] Args&&... args)
  {
    APP_ABORT(" Error: ph_excited_energy not implemented yet. ");
  }

  auto vHS_sparse(nda::MemoryArrayOfRank<2> auto && X, double dt)
  {
    utils::check(false, "vHS_sparse not implemented in THCOps.");
    nda::array<csrMat<ComplexType>,1> spvHS;
    return spvHS();
  }

  auto vHS(nda::MemoryMatrix auto&& X, double dt)
  {
    constexpr MEMORY_SPACE MEM_X = memory::get_memory_space<decltype(X)>();
    static_assert(MEM == MEM_X, "Memory space mismatch");
    using nda::range;
    auto all  = range::all;
    ComplexType one(1.0), zero(0.0);
    int nwalk = X.extent(0);
    int NMO   = nkpts*nbnd;
    int nspin_in_H = hij.extent(0);
    int npol_in_H = hij.extent(2)/nbnd;
    utils::check_shape(X, "X", nwalk, 2*ncvecs);

    // Note: Allocate first, to make better use of memory pool
    // vHS[nspin_in_vHS][nwalk][npol_in_vHS*NMO][NMO]
    memory::buffered_array<MEM_X,ComplexType,4> v(nspin_in_H,nwalk,npol_in_H*NMO,NMO);
    auto v7d = nda::reshape(v,std::array<long,7>{nspin_in_H,nwalk,npol_in_H,nkpts,nbnd,nkpts,nbnd});
    v() = ComplexType(0.0);

    auto X3d = nda::reshape(X,std::array<long,3>{nwalk,2,ncvecs});
    {
      // "rotate" X
      //  XIJ = 0.5*a*(Xn+ -i*Xn-), XJI = 0.5*a*(Xn+ +i*Xn-)
      RealType a = std::sqrt(dt)*0.5;
      memory::buffered_array<MEM,ComplexType,2> T(nwalk,ncvecs);
      T() = X3d(all,0,all);
      nda::tensor::add(ComplexType(0.0,-a),X3d(all,1,all),"wn",ComplexType(a),T,"wn");
      nda::tensor::add(ComplexType(a),X3d(all,0,all),"wn",ComplexType(0.0,a),X3d(all,1,all),"wn");
      X3d(all,0,all) = T();
      //  then combine Q/(-Q) pieces if Q != -Q
      //  X(Q)np = (X(Q)np + X(-Q)nm)
      for(int iq=0; iq<nkpts; iq++)  {
        int nchol = Lank(iq).extent(4);
        if( iq != minusq(iq) ) 
          math::accumulate(one,X3d(all,1,range(ncv0(minusq(iq)),ncv0(minusq(iq))+nchol)),
                                          X3d(all,0,range(ncv0(iq),ncv0(iq)+nchol)));
      }
    }

    if constexpr (MEM==HOST_MEMORY) {
      memory::buffered_array<MEM,ComplexType,3> vKK(nwalk,nbnd,nbnd);
      auto v2d = nda::reshape(vKK,std::array<long,2>{nwalk,nbnd*nbnd});
      for(int is=0; is<nspin_in_H; ++is) {
        for (int Q = 0; Q < nkpts; ++Q) {
          int nchol = Lank(Q).extent(4);
          for (int ip = 0; ip < npol_in_H; ++ip) {
            for (int ik = 0; ik < nkpts; ++ik) 
            { 
              // v[nw][i(in K)][k(in Q(K))] += sum_n LQK[i][k][n] X[Q][0][n][nw]
              if (Q <= minusq(Q))
              {
                auto Lq_ = nda::reshape(LQ(Q)()(is,ip,nda::ellipsis{}),std::array<long,3>{nkpts,nbnd*nbnd,nchol});
                // ci^dagger cj term
                // LQ(Q)(ispin, ipol, ik, i, j, nchol) 
                nda::blas::gemm(one,X3d(all,0,range(ncv0(Q),ncv0(Q)+nchol)),
                      nda::transpose(Lq_(ik,all,all)),zero,v2d);

                // accumulate on v7d(nwalk,nspin_in_H,npol_in_H,nkpts,nbnd,nkpts,nbnd)
                int k2 = qk_to_k2(Q,ik);
                nda::tensor::add(one,vKK,"wij",one,v7d(is,all,ip,ik,all,k2,all),"wij");
              } else {
                auto Lqm_ = nda::reshape(LQ(minusq(Q))()(is,ip,nda::ellipsis{}),std::array<long,3>{nkpts,nbnd*nbnd,nchol});
                // cj^dagger ci term
                // the kpoint index of LQ(Qm) refers to k2 
                int k2 = qk_to_k2(Q,ik);
                nda::blas::gemm(one,X3d(all,0,range(ncv0(Q),ncv0(Q)+nchol)),
                      nda::dagger(Lqm_(k2,all,all)),zero,v2d);

                // accumulate on v7d(nwalk,nspin_in_H,npol_in_H,nkpts,nbnd,nkpts,nbnd)
                nda::tensor::add(one,vKK,"wij",one,v7d(is,all,ip,ik,all,k2,all),"wji");
              }
              // v[nw][k(in Q(K))][i(in K)] += sum_n conj(LQK[i][k][n]) X[Q][n-][nw]
              if(Q == minusq(Q)) {
                auto Lq_ = nda::reshape(LQ(Q)()(is,ip,nda::ellipsis{}),std::array<long,3>{nkpts,nbnd*nbnd,nchol});
                // cj^dagger ci term for Q==minusq(Q)
                // the kpoint index refers to k2 
                nda::blas::gemm(one,X3d(all,1,range(ncv0(Q),ncv0(Q)+nchol)),
                      nda::dagger(Lq_(ik,all,all)),zero,v2d);

                // accumulate on v7d(nwalk,nspin_in_H,npol_in_H,nkpts,nbnd,nkpts,nbnd)
                int k2 = qk_to_k2(Q,ik);
                nda::tensor::add(one,vKK,"wij",one,v7d(is,all,ip,k2,all,ik,all),"wji");
              }
            } // ik
          } // ip 
        } // Q
      } // is
    } else {
      memory::buffered_array<MEM,ComplexType,5> vKK(nwalk,npol_in_H,nkpts,nbnd,nbnd);
      auto v2d = nda::reshape(vKK,std::array<long,2>{nwalk,npol_in_H*nkpts*nbnd*nbnd});
      for(int is=0; is<nspin_in_H; ++is) {
        for (int Q = 0; Q < nkpts; ++Q)
        { 
          int nchol = Lank(Q).extent(4);
          // v[nw][i(in K)][k(in Q(K))] += sum_n LQK[i][k][n] X[Q][0][n][nw]
          if (Q <= minusq(Q))
          {
            // ci^dagger cj term
            // LQ(Q)(ispin, ipol, ik, i, j, nchol) 
            auto Lq_ = nda::reshape(LQ(Q)()(is,nda::ellipsis{}),std::array<long,2>{npol_in_H*nkpts*nbnd*nbnd,nchol});
            nda::blas::gemm(one,X3d(all,0,range(ncv0(Q),ncv0(Q)+nchol)),
                  nda::transpose(Lq_),zero,v2d);

            // accumulate on v7d(nwalk,nspin_in_H,npol_in_H,nkpts,nbnd,nkpts,nbnd)
            for(int ik=0; ik<nkpts; ++ik) {
              int k2 = qk_to_k2(Q,ik);
              for(int ip=0; ip<npol_in_H; ++ip)
                math::accumulate(one,vKK(all,ip,ik,all,all),v7d(is,all,ip,ik,all,k2,all));
            }
          } else {
            // cj^dagger ci term
            // the kpoint index of LQ(Qm) refers to k2 
            auto Lq_ = nda::reshape(LQ(minusq(Q))()(is,nda::ellipsis{}),std::array<long,2>{npol_in_H*nkpts*nbnd*nbnd,nchol});
            nda::blas::gemm(one,X3d(all,0,range(ncv0(Q),ncv0(Q)+nchol)),
                  nda::dagger(Lq_),zero,v2d);

            // accumulate on v7d(nwalk,nspin_in_H,npol_in_H,nkpts,nbnd,nkpts,nbnd)
            for(int ik=0; ik<nkpts; ++ik) {
              int k2 = qk_to_k2(Q,ik);
              for(int ip=0; ip<npol_in_H; ++ip)
                nda::tensor::add(one,vKK(all,ip,k2,all,all),"wij",one,v7d(is,all,ip,ik,all,k2,all),"wji");
            }
          }
          // v[nw][k(in Q(K))][i(in K)] += sum_n conj(LQK[i][k][n]) X[Q][n-][nw]
          if(Q == minusq(Q)) {
            // cj^dagger ci term for Q==minusq(Q)
            // the kpoint index refers to k2 
            auto Lq_ = nda::reshape(LQ(Q)()(is,nda::ellipsis{}),std::array<long,2>{npol_in_H*nkpts*nbnd*nbnd,nchol});
            nda::blas::gemm(one,X3d(all,1,range(ncv0(Q),ncv0(Q)+nchol)),
                  nda::dagger(Lq_),zero,v2d);

            // accumulate on v7d(nwalk,nspin_in_H,npol_in_H,nkpts,nbnd,nkpts,nbnd)
            for(int ik=0; ik<nkpts; ++ik) {
              int k2 = qk_to_k2(Q,ik);
              for(int ip=0; ip<npol_in_H; ++ip)
                nda::tensor::add(one,vKK(all,ip,ik,all,all),"wij",one,v7d(is,all,ip,k2,all,ik,all),"wji");
            }
          }
        } // Q
      } // is
    }
    return v;
  }

  void vbias(nda::MemoryArrayOfRank<2> auto const& G, nda::MemoryArrayOfRank<2> auto& v, double dt)
  {
    memory::check_memory_space<MEM>(G,v);
    using nda::range;
    auto all = range::all;
    ComplexType one(1.0), zero(0.0);
    int nwalk = G.extent(0);
    int nspin = (walker_type == COLLINEAR) ? 2 : 1;
    int npol  = (walker_type == NONCOLLINEAR) ? 2 : 1;
    int nel  = (walker_type == COLLINEAR ? nup+ndown : nup); // NONCOLLINEAR has ndown=0 
    utils::check_strides(G,v);
    utils::check_shape(v, "v", nwalk, 2*ncvecs);
    if(haj.extent(0) == 1) { // ndet==1, G half rotated
      utils::check_shape(G, "G", nwalk, nel*npol*nkpts*nbnd);
    } else { // ndet>1, full G
      utils::check_shape(G, "G", nwalk, nspin*npol*nkpts*nbnd*npol*nkpts*nbnd);
    }
    // needed to build G3d easily, can build from strides if needed
    utils::check(G.is_contiguous(), "Layout mismatch");
    utils::check(v.is_contiguous(), "Layout mismatch");

    // scale a by sqrt(dt)
    RealType scl = (walker_type == CLOSED ? 2.0 : 1.0);
    ComplexType a = ComplexType(std::sqrt(dt)*0.5*scl);
    ComplexType ia = ComplexType(0.0,std::sqrt(dt)*0.5*scl);

    // v4d
    auto v3d = nda::reshape(v, std::array<long,3>{nwalk,2,ncvecs});
    v() = ComplexType(0.0);

    utils::check(G.is_contiguous(), "Layout mismatch");

    // G in new layout

    if (haj.extent(0) == 1)
    {
      memory::array_view<MEM,const ComplexType,5> G5d(std::array<long,5>{nwalk,nel,npol,nkpts,nbnd},G.data());
      for(int is=0; is<nspin; ++is)  {
        for(int iq=0; iq<nkpts; ++iq) {

          int nchol = Lank(iq).extent(4);
          int iqm = minusq(iq);
          memory::buffered_array<MEM,ComplexType,4> GQK(nwalk,nkpts,nocc_max,npol*nbnd);
          memory::buffered_array<MEM,ComplexType,2> vn(nwalk,nchol);
          auto G2d = nda::reshape(GQK,std::array<long,2>{nwalk,nkpts*nocc_max*npol*nbnd});

          Gc_to_GQKwaj(is,is*nup,iq,G5d,GQK);
          auto La = nda::reshape((*Lakn)(iq)()(is,nda::ellipsis{}),std::array<long,2>{nkpts*nocc_max*npol*nbnd,nchol});
          nda::blas::gemm(one,G2d,La,zero,vn);
         
          // accumulate on v3d
          // v+ = a*(v[Q]+v[-Q])
          nda::tensor::add(a,vn,"wn",one,v3d(all,0,range(ncv0(iq),ncv0(iq)+nchol)),"wn");
          // v- = -a*i*(v[Q]-v[-Q])
          nda::tensor::add(-ia,vn,"wn",one,v3d(all,1,range(ncv0(iq),ncv0(iq)+nchol)),"wn");

          if(iq == iqm) {
            auto Lb = nda::reshape((*Lbkn)(Qmap(iq))()(is,nda::ellipsis{}),std::array<long,2>{nkpts*nocc_max*npol*nbnd,nchol});
            nda::blas::gemm(one,G2d,Lb,zero,vn);

            // accumulate on v3d
            nda::tensor::add(a,vn,"wn",one,v3d(all,0,range(ncv0(iqm),ncv0(iqm)+nchol)),"wn");
            nda::tensor::add(ia,vn,"wn",one,v3d(all,1,range(ncv0(iqm),ncv0(iqm)+nchol)),"wn");
          } else {
            nda::tensor::add(a,vn,"wn",one,v3d(all,0,range(ncv0(iqm),ncv0(iqm)+nchol)),"wn");
            nda::tensor::add(ia,vn,"wn",one,v3d(all,1,range(ncv0(iqm),ncv0(iqm)+nchol)),"wn");
          }
        } // iq
      } // is  
    }
    else
    {
      utils::check(false," Error: KP3Index not yet implemented for multiple references.");
    }
  }

  /// Returns the number of spins and polarizations in the VHS potential.
  std::tuple<int,int> vHS_dims() const {
    int nspin_in_H = hij.extent(0);
    int npol_in_H = hij.extent(2)/nbnd;
    return std::make_tuple(nspin_in_H, npol_in_H);
  }
  int number_of_ke_vectors() const { return 2 * ncvecs; }
  int number_of_cholesky_vectors() const { return 2 * ncvecs; }
  RealType energy_offset() const { return E0; }

  nda::array<ComplexType, 2> getHSPotentials()
  { return nda::array<ComplexType, 2>{}; }

protected:
  // keeping communicators here seems unnecessary
  std::shared_ptr<utils::mpi_context_t<mpi3::communicator>> mpi;

  WALKER_TYPES walker_type;

  int nkpts=0;
  int nbnd=0;
  int nup=0;
  int ndown=0;
  int nocc_max=0;
  int nchol_max=0;
  int ncvecs=0;     // total number of cholesky vectors

  // For each (spin,kpoint): the list of occupied PsiT row indices at that kpoint.
  nda::array<nda::array<int,1>,2> nocc;

  // BZ information
  nda::array<int,1> minusq;
  nda::array<int,2> qk_to_k2;
  nda::array<int,1> Qmap;  // location of Q (Q==minusq(Q)) in Lbnk
  nda::array<int,1> ncv0;  // location of first cholesky vector for each Q
  int Q0_index = 0;

  // H1[nspin][nk][npol*nbnd][npol*nbnd]
  memory::const_shared_array<HOST_MEMORY,ComplexType,4> hij;

  // half rotated one body hamiltonian: [ndet][nup+ndn][npol*NMO]. Kept in full basis
  memory::const_shared_array<MEM,ComplexType,3> haj;

  // LQ(Q)(ispin, ipol, ik, i, j, nchol)
  nda::array<memory::const_shared_array<MEM,ComplexType,6>,1> LQ;

  // Lank(Q)(ndet, nspin, nkpts, nocc_max, nchol, npol*nbnd)
  nda::array<memory::const_shared_array<MEM,ComplexType,6>,1> Lank;

  // if ndet==1, this is used for faster evaluation of vbias
  std::optional<nda::array<memory::const_shared_array<MEM,ComplexType,5>,1>> Lakn;
  std::optional<nda::array<memory::const_shared_array<MEM,ComplexType,5>,1>> Lbkn;

  // Lbnk(Qmap(Q))(ndet, nspin, nkpts, nocc_max, nchol, npol*nbnd), only for q==minusq(q)
  nda::array<memory::const_shared_array<MEM,ComplexType,6>,1> Lbnk;

  // vexx(i,l) = -0.5 * sum_j <ij|jl> : [nspin][nk][npol*nbnd][npol*nbnd]
  memory::const_shared_array<HOST_MEMORY,ComplexType,4> vexx;

  int default_buffer_size_in_MB=2000;

  RealType E0 = 0.0;

  // maps Q (only for those with Qmap >=0) to the corresponding sector in vbias
//  stdIVector Q2vbias;

  // Bytes per walker of the exchange work arrays: {GKK, Tl+Tr of one (Ka,Kb) slot}.
  std::pair<long,long> exchange_bytes_per_walker() const {
    long npol = (walker_type == NONCOLLINEAR) ? 2 : 1;
    long gkk  = long(nkpts) * nkpts * nocc_max * npol * nbnd * sizeof(ComplexType);
    long slot = 2L * nocc_max * nocc_max * nchol_max * sizeof(ComplexType);
    return {gkk, slot};
  }

  // Coulomb intermediates of spin `is`, accumulated on Kleft/Kright(nwalk,ncvecs):
  //   Kleft(w, n in Q)  = Tr(L^{Q,n} G_w), the Q component of the density,
  //   Kright(w, n in Q) = Tr(L^{Q,n}^dagger G_w), the -Q component, i.e. the -Q block of Kleft if Q != -Q.
  // These are the contractions vbias combines as Kleft +- Kright.
  // `base` is the offset of spin `is`'s block in the occupied dimension of G.
  void coulomb_intermediates(int is, int base, nda::MemoryArrayOfRank<5> auto const& G5d,
                             nda::MemoryMatrix auto && Kleft, nda::MemoryMatrix auto && Kright) {
    using nda::range;
    auto all = range::all;
    ComplexType one(1.0), zero(0.0);
    int npol = (walker_type == NONCOLLINEAR) ? 2 : 1;
    long nwalk = G5d.extent(0);
    utils::check(Lakn.has_value(), "KP3IndexFactorization: the Coulomb energy requires a single determinant.");

    memory::buffered_array<MEM,ComplexType,4> GQK(nwalk,nkpts,nocc_max,npol*nbnd);
    auto G2d = nda::reshape(GQK,std::array<long,2>{nwalk,nkpts*nocc_max*npol*nbnd});
    for(int Q = 0; Q < nkpts; ++Q) {
      int Qm = minusq(Q);
      int nchol = Lank(Q).extent(4);
      memory::buffered_array<MEM,ComplexType,2> vn(nwalk,nchol);
      Gc_to_GQKwaj(is,base,Q,G5d,GQK);

      auto La = nda::reshape((*Lakn)(Q)()(is,nda::ellipsis{}),std::array<long,2>{nkpts*nocc_max*npol*nbnd,nchol});
      nda::blas::gemm(one,G2d,La,zero,vn);
      math::accumulate(one,vn,Kleft(all,range(ncv0(Q),ncv0(Q)+nchol)));
      if(Q == Qm) {
        auto Lb = nda::reshape((*Lbkn)(Qmap(Q))()(is,nda::ellipsis{}),std::array<long,2>{nkpts*nocc_max*npol*nbnd,nchol});
        nda::blas::gemm(one,G2d,Lb,zero,vn);
        math::accumulate(one,vn,Kright(all,range(ncv0(Q),ncv0(Q)+nchol)));
      } else {
        math::accumulate(one,vn,Kright(all,range(ncv0(Qm),ncv0(Qm)+nchol)));
      }
    }
  }

  // Exchange energy of spin `is`, accumulated on Exx(nwalk). Walkers are processed in chunks so
  // that GKK and the Tl/Tr work arrays together fit in default_buffer_size_in_MB; on device, what
  // is left of the budget batches (Ka,Kb) pairs.
  // `base` is the offset of spin `is`'s block in the occupied dimension of G.
  void exchange_energy(int is, int base, int idet,
                       nda::MemoryArrayOfRank<5> auto const& G5d,
                       nda::MemoryVector auto && Exx) {
    using nda::range;
    auto all = range::all;
    ComplexType one(1.0), zero(0.0);
    ComplexType scl = (walker_type == CLOSED ? ComplexType(2.0) : ComplexType(1.0));
    int npol  = (walker_type == NONCOLLINEAR) ? 2 : 1;
    int nqpts = LQ.extent(0);
    long nwalk = G5d.extent(0);
    if(nwalk == 0) {
      return;
    }

    auto [gkk_bytes, slot_bytes] = exchange_bytes_per_walker();
    long const budget = long(default_buffer_size_in_MB) * 1024L * 1024L;
    long const nw_chunk = std::clamp(budget / (gkk_bytes + slot_bytes), 1L, nwalk);
    // MAM: For cpu threading, let batch_size >> 1
    long batch_size = 1;
    if constexpr(MEM == DEVICE_MEMORY) {
      batch_size = std::clamp((budget / nw_chunk - gkk_bytes) / slot_bytes, 1L, long(nkpts) * nkpts);
    }

    // sized for a full chunk; Tl and Tr are the two halves of Tbuff
    long const t_size = batch_size * nw_chunk * nocc_max * nocc_max * nchol_max;
    memory::buffered_array<MEM,ComplexType,1> GKKbuff(long(nkpts) * nkpts * nw_chunk * nocc_max * npol * nbnd);
    memory::buffered_array<MEM,ComplexType,1> Tbuff(2 * t_size);

    for(long w0 = 0; w0 < nwalk; w0 += nw_chunk) {
      long nw = std::min(nw_chunk, nwalk - w0);
      auto wr = range(w0, w0 + nw);

      // reorder G
      memory::array_view<MEM,ComplexType,4,nda::C_layout> GKK(std::array<long,4>{nkpts,nkpts,nw*nocc_max,npol*nbnd},
                                                              GKKbuff.data());
      Gc_to_GKKwaj(is, base, G5d(wr,nda::ellipsis{}), GKK);

      // Lank(Q)(idet,ispin,nkpts, nocc_max, nchol, npol*nbnd)
      using A_t = decltype(nda::transpose(Lank(0)()(0,0,0,0,all,all)));
      using B_t = decltype(GKK(0,0,all,all));
      std::vector<A_t> Av;
      std::vector<B_t> Bv;
      std::vector<B_t> Cv;
      Av.reserve(2*batch_size);
      Bv.reserve(2*batch_size);
      Cv.reserve(2*batch_size);

      for(int Q = 0; Q < nqpts; ++Q) {
        int Qm    = minusq(Q);
        int nchol = Lank(Q).extent(4);
        // Lank(Q)(ndet, nspin, nkpts, nocc_max, nchol, npol*nbnd)
        auto La = nda::reshape(Lank(Q)()(idet,is,nda::ellipsis{}),std::array<long,3>{nkpts,nocc_max*nchol,npol*nbnd});
        auto LB = (Q==Qm ? Lbnk(Qmap(Q))() : Lank(Qm)());
        auto Lb = nda::reshape(LB(idet,is,nda::ellipsis{}),std::array<long,3>{nkpts,nocc_max*nchol,npol*nbnd});

        memory::array_view<MEM,ComplexType,3,nda::C_layout> Tl3d(std::array<long,3>{batch_size,nw*nocc_max,nocc_max*nchol},
                                                                  Tbuff.data());
        memory::array_view<MEM,ComplexType,3,nda::C_layout> Tr3d(std::array<long,3>{batch_size,nw*nocc_max,nocc_max*nchol},
                                                                  Tbuff.data() + t_size);
        auto Tl5d = nda::reshape(Tl3d,std::array<long,5>{batch_size,nw,nocc_max,nocc_max,nchol});
        auto Tr5d = nda::reshape(Tr3d,std::array<long,5>{batch_size,nw,nocc_max,nocc_max,nchol});

        int batch_cnt = 0;
        ComplexType batch_pref = zero;
        auto flush = [&]() {
          nda::blas::gemm_batch<false>(one,Bv,Av,zero,Cv);
          nda::tensor::contract(batch_pref,Tl5d(range(batch_cnt),nda::ellipsis{}),"lwabn",
                                Tr5d(range(batch_cnt),nda::ellipsis{}),"lwban",one,Exx(wr),"w");
          Av.clear();
          Bv.clear();
          Cv.clear();
          batch_cnt = 0;
        };
        // queues the (Ka,Kb) term; a batch is contracted with a single prefactor
        auto push = [&](int Ka, int Kb, ComplexType pref) {
          if(batch_cnt > 0 && pref != batch_pref) {
            flush();
          }
          batch_pref = pref;
          int Kl = qk_to_k2(Qm,Kb);
          int Kk = qk_to_k2(Q,Ka);

          // L[Kb][Kl] * G [Ka][Kl]
          // L(nkpts, nocc_max*nchol, nbnd)
          Av.emplace_back(nda::transpose(Lb(Kb,all,all)));
          // GKK(nkpts,nkpts,nw*nocc_max,npol*nbnd)
          Bv.emplace_back(GKK(Ka,Kl,all,all));
          Cv.emplace_back(Tr3d(batch_cnt,all,all));

          Av.emplace_back(nda::transpose(La(Ka,all,all)));
          Bv.emplace_back(GKK(Kb,Kk,all,all));
          Cv.emplace_back(Tl3d(batch_cnt++,all,all));

          if(batch_cnt >= batch_size) {
            flush();
          }
        };

        // Q==-Q sums all (Ka,Kb) with a factor of 0.5; Q!=-Q sums Kb>=Ka, with the factor of 0.5
        // on Ka==Kb only. The Ka==Kb terms go first so the prefactor changes at most once.
        for(int Ka = 0; Ka < nkpts; ++Ka) {
          push(Ka, Ka, ComplexType(-0.5) * scl);
        }
        ComplexType pref = ComplexType(Q == Qm ? -0.5 : -1.0) * scl;
        for(int Ka = 0; Ka < nkpts; ++Ka) {
          for(int Kb = (Q == Qm ? 0 : Ka + 1); Kb < nkpts; ++Kb) {
            if(Kb != Ka) {
              push(Ka, Kb, pref);
            }
          }
        }
        if(batch_cnt > 0) {
          flush();
        }
      } // Q
    } // walker chunk
  }

  // Changes the layout of G:
  //    From: G(nwalk,nel_tot,npol,nkpts,nbnd)
  //    To:   GKK(nkpts,nkpts,nwalk*nocc_max,npol*nbnd)
  // `base` is the offset of spin `is`'s block in the occupied dimension of G; the
  // occupied orbital `a` of kpoint ik1 lives at G-row base+nocc(is,ik1)(a).
  void Gc_to_GKKwaj(int is, int base, nda::MemoryArrayOfRank<5> auto const& G,
                        nda::MemoryArrayOfRank<4> auto && GKK)
  {
    using nda::range;
    auto all  = range::all;
    int npol  = (walker_type == NONCOLLINEAR) ? 2 : 1;
    int nwalk = G.extent(0);

    auto G6d = nda::reshape(GKK,std::array<long,6>{nkpts,nkpts,nwalk,nocc_max,npol,nbnd});
    GKK() = ComplexType(0.0);
    for(int ik1=0; ik1<nkpts; ++ik1)
    {
      auto const& rows = nocc(is,ik1);
      int nk1 = int(rows.size());
      if(nk1==0) continue;
      bool contig = contiguous_rows(rows);
      if constexpr (MEM==HOST_MEMORY) {
        for(int ik2=0; ik2<nkpts; ++ik2)
        {
          for(int ip=0; ip<npol; ++ip)
            for(int iw=0; iw<nwalk; ++iw) {
              if(contig) {
                int r0 = base + rows(0);
                G6d(ik1,ik2,iw,range(nk1),ip,all) = G(iw,range(r0,r0+nk1),ip,ik2,all);
              } else {
                for(int a=0; a<nk1; ++a)
                  G6d(ik1,ik2,iw,a,ip,all) = G(iw,base+rows(a),ip,ik2,all);
              }
            }
        }
      } else {
        if(contig) {
          int r0 = base + rows(0);
          nda::tensor::add(ComplexType(1.0),G(all,range(r0,r0+nk1),all,all,all),"wapkj",
                           ComplexType(0.0),G6d(ik1,all,all,range(nk1),all,all),"kwapj");
        } else {
          for(int a=0; a<nk1; ++a)
            nda::tensor::add(ComplexType(1.0),G(all,range(base+rows(a),base+rows(a)+1),all,all,all),"wapkj",
                             ComplexType(0.0),G6d(ik1,all,all,range(a,a+1),all,all),"kwapj");
        }
      }
    }
  }

  // Changes the layout of G:
  //    From: G(nwalk,nel_tot,npol,nkpts,nbnd)
  //    To:   GQK(nkpts,nwalk,nkpts,nocc_max,npol*nbnd)
  // `base` is the offset of spin `is`'s block in the occupied dimension of G.
  void Gc_to_GQKwaj(int is, int base, int iq, nda::MemoryArrayOfRank<5> auto const& G,
                    nda::MemoryArrayOfRank<4> auto && GQK)
  {
    using nda::range;
    auto all  = range::all;
    int npol  = (walker_type == NONCOLLINEAR) ? 2 : 1;
    int nwalk = G.extent(0);

    auto G5d = nda::reshape(GQK,std::array<long,5>{nwalk,nkpts,nocc_max,npol,nbnd});
    GQK() = ComplexType(0.0);
    for(int ik=0; ik<nkpts; ++ik)
    {
      int k2 = qk_to_k2(iq,ik);
      auto const& rows = nocc(is,ik);
      int nk = int(rows.size());
      if(nk==0) continue;
      bool contig = contiguous_rows(rows);
      for(int ip=0; ip<npol; ++ip) {
        if(contig) {
          int r0 = base + rows(0);
          math::accumulate(ComplexType(1.0),G(all,range(r0,r0+nk),ip,k2,all),G5d(all,ik,range(nk),ip,all));
        } else {
          for(int a=0; a<nk; ++a)
            math::accumulate(ComplexType(1.0),G(all,range(base+rows(a),base+rows(a)+1),ip,k2,all),G5d(all,ik,range(a,a+1),ip,all));
        }
      }
    }
  }

};

} // namespace afqmc

} // namespace sfqmc

