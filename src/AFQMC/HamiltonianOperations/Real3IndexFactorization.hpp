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

#include <vector>
#include <type_traits>

#include "AFQMC/config.h"
#include "nda/nda.hpp"
#include "nda/tensor.hpp"
#include "utilities/check.hpp"
#include "utilities/freemem.h"
#include "utilities/mpi_context.h"
#include "utilities/check_strides.hpp"
#include "numerics/shared_array/const_shared_array.hpp"
#include "numerics/nda_functions.hpp"
#include "detail/one_body.hpp"

#include "AFQMC/Wavefunctions/detail/phmsd_impl.hpp"
#include "AFQMC/HamiltonianOperations/full_g_estimators.hpp"

namespace sfqmc
{
namespace afqmc
{

template<MEMORY_SPACE MEM>
class Real3IndexFactorization
{
public:
  static const HamiltonianTypes HamOpType = RealDenseFactorized;
  HamiltonianTypes getHamType() const { return HamOpType; }

  // Does vbias() accept a FULL [nwalk, nspin*npol*NMO*npol*NMO] density matrix, in addition to the
  // half-rotated compact one? NOMSD needs it for ndet>1 trials, and StochasticWfn::vMF needs it for
  // ANY trial once inner_nwalkers > 1, because the stochastic mean field is a reduction of the inner
  // ensemble against itself and has no half-rotated form. Callers that can hand over a full G must
  // gate on this: the operators that lack it must reject such a G, never reinterpret it.
  constexpr bool has_fullG_vbias() const { return true; }

  Real3IndexFactorization(std::shared_ptr<utils::mpi_context_t<mpi3::communicator>> ctxt,
        WALKER_TYPES type,
        int NMO_,
        int nup_,
        int ndown_,
        memory::const_shared_array<HOST_MEMORY,ComplexType,3>&& hij_,
        memory::const_shared_array<MEM,ComplexType,3>&& haj_,
        memory::const_shared_array<MEM,RealType,4>&& vik,
        nda::array<memory::const_shared_array<MEM,ComplexType,5>,1>&& vnak_,
        memory::const_shared_array<HOST_MEMORY,RealType,3>&& v0_,
        ComplexType e0_,
        long maxMem = 2000)
      : mpi(ctxt), 
        walker_type(type),
        NMO(NMO_),
        nup(nup_),
        ndown(ndown_),
        nCV(vik.extent(3)),
        max_memory_MB(maxMem),
        hij(std::move(hij_)),
        haj(std::move(haj_)),
        Likn(std::move(vik)),
        Lnak(std::move(vnak_)),
        vexx(std::move(v0_)),
        E0(e0_)
  {
    nCV = Likn.extent(3);
    int npol  = (walker_type == NONCOLLINEAR) ? 2 : 1;
    int nspin  = (walker_type == COLLINEAR) ? 2 : 1;
    int nstot = hij.extent(0);
    int nptot = hij.extent(1)/NMO;
    int npol_H2  = (walker_type == NONCOLLINEAR) ? Likn.extent(0) : 1;
    int nspin_H2  = (walker_type == COLLINEAR) ? Likn.extent(0) : 1;
    int ndet = haj.extent(0);
    int nel[] = {nup, (type == COLLINEAR ? ndown : 0) };
    // checking dimensions!!!
    utils::check(hij.shape() == std::array<long,3>{nstot,nptot*NMO,nptot*NMO},
                 "Real3IndexFactorization: Size mismatch");
    utils::check(haj.shape() == std::array<long,3>{ndet,nel[0]+nel[1],npol*NMO},
                 "Real3IndexFactorization: Size mismatch");
    utils::check(Likn.shape() == std::array<long,4>{nspin_H2*npol_H2,NMO,NMO,nCV},
                 "Real3IndexFactorization: Size mismatch");
    utils::check(Lnak(0).shape() == std::array<long,5>{ndet,npol,nCV,nup,NMO}, 
                 "Real3IndexFactorization: Size mismatch");
    if(nspin==2)
      utils::check(Lnak(1).shape() == std::array<long,5>{ndet,npol,nCV,ndown,NMO}, 
                   "Real3IndexFactorization: Size mismatch");
    utils::check(vexx.shape() == std::array<long,3>{nspin_H2*npol_H2,NMO,NMO},
                 "Real3IndexFactorization: Size mismatch");
    app_log(1,"****************************************************************** ");
    app_log(1,"  Static memory usage by Real3IndexFactorization (node 0 in MB) ");
    app_log(1,"  Likn: {}", double(Likn.size() * sizeof(RealType)) / 1024.0 / 1024.0);
    app_log(1,"  Lnak: {}", double((Lnak(0).size() + (nspin==2?Lnak(1).size():0.0)) 
                * sizeof(ComplexType)) / 1024.0 / 1024.0);
    app_log(1,"  Buffer memory limited to (not yet allocated) : {} MB", max_memory_MB);
    utils::memory_report();
  }

  ~Real3IndexFactorization() = default; 

  Real3IndexFactorization(const Real3IndexFactorization& other) = default;
  Real3IndexFactorization& operator=(const Real3IndexFactorization& other) = default;
  Real3IndexFactorization(Real3IndexFactorization&& other)                 = default;
  Real3IndexFactorization& operator=(Real3IndexFactorization&& other) = default;

  nda::array<ComplexType,3> getOneBodyPropagatorMatrix(double dt,
                                                       nda::MemoryVector auto const& vMF)
  {
    auto all = nda::range::all;
    int npol  = (walker_type == NONCOLLINEAR) ? 2 : 1;
    int nspin  = (walker_type == COLLINEAR) ? 2 : 1;
    int nspin_in_H = hij().extent(0);
    int npol_in_H = hij().extent(1)/NMO;
    utils::check(vMF.extent(0) == number_of_cholesky_vectors(), "Size mismatch");

    memory::buffered_array<MEM,ComplexType,2> vMF_2d(1,vMF.size());
    vMF_2d(0,all) = vMF();

    auto meanfield_shift{nda::to_host(vHS(vMF_2d, dt))};

    nda::array<ComplexType, 3> H1(nspin, npol*NMO, npol*NMO);
    H1() = ComplexType(0.0);


    memory::buffered_array<HOST_MEMORY,ComplexType,5> hij_plus_v(nspin_in_H, npol_in_H, NMO, npol_in_H, NMO);
    add_one_body_shifts(dt, hij(), meanfield_shift(), vexx(), hij_plus_v());
    broadcast_one_body(hij_plus_v,
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
    int nspin = (walker_type == COLLINEAR ? 2 : 1);
    int npol = (walker_type == NONCOLLINEAR ? 2 : 1);
    int nwalk = G.extent(0);
    int nel  = (walker_type == COLLINEAR ? nup+ndown : nup); // NONCOLLINEAR has ndown=0 
    utils::check(E.shape() == std::array<long,2>{nwalk,3}, "Size mismatch.");
    utils::check(G.extent(1) == nel*npol*NMO, "Size mismatch.");
    utils::check_strides(E,G);

    E() = ComplexType(0.0);
    if(addH1) E(all,0) = E0;

    memory::buffered_array<MEM,ComplexType,2> Kl((addEJ?nwalk:0),(addEJ?nCV:0));
    if(addEJ) Kl() = ComplexType(0.0);

    utils::check(G.is_contiguous(), "Layout mismatch");
    memory::array_view<MEM,const ComplexType,3> G3d(std::array<long,3>{nwalk,nel,npol*NMO},G.data());

    for(int is=0; is<nspin; ++is) 
      energy_impl(is, E, G3d(all,range(is*nup,nup+is*ndown),all), idet, Kl,
                addH1, addEJ, addEXX);

    if (addEJ)
    {
      RealType scl = (walker_type == CLOSED ? 4.0 : 1.0);
      nda::tensor::contract(ComplexType(0.5*scl),Kl,"wn",Kl,"wn",ComplexType(1.0),E(all,2),"w");
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
    memory::check_memory_space<MEM>(E,G,EJn);
    using nda::range;
    auto all = range::all;
    int ispin = ( spin_component == Alpha ? 0 : 1);
    int npol = (walker_type == NONCOLLINEAR ? 2 : 1);
    int nwalk = G.extent(0);
    int nel  = (ispin==0 ? nup : ndown);
    utils::check(E.shape() == std::array<long,2>{nwalk,3}, "Size mismatch.");
    if(addEJ)
      utils::check(EJn.shape() == std::array<long,2>{nwalk,nCV}, "Size mismatch.");
    utils::check(G.extent(1) == nel*npol*NMO, "Size mismatch.");
    utils::check_strides(E,G);

    E() = ComplexType(0.0);

    // by convention, add E0 only to Alpha
    if(addH1 and (spin_component == Alpha)) E(all,0) = E0; 
    if(addEJ) EJn() = ComplexType(0.0);

    utils::check(G.is_contiguous(), "Layout mismatch");
    memory::array_view<MEM,const ComplexType,3> G3d(std::array<long,3>{nwalk,nel,npol*NMO},G.data());

    energy_impl(ispin, E, G3d, idet, EJn, addH1, addEJ, addEXX);
    if (addEJ)
      nda::tensor::contract(ComplexType(0.5),EJn,"wn",EJn,"wn",ComplexType(1.0),E(all,2),"w");
  }

  void ph_reference_energy(SpinTypes spin_component,
              nda::MemoryArrayOfRank<2> auto && E,
              nda::MemoryArrayOfRank<2> auto const& G,
              nda::MemoryArrayOfRank<2> auto && EJn,
              bool addH1 = true)
  {
    memory::check_memory_space<MEM>(E,G,EJn);
    using nda::range;
    auto all = range::all;
    int ispin = ( spin_component == Alpha ? 0 : 1);
    int nwalk = G.extent(0);
    utils::check(E.shape() == std::array<long,2>{nwalk,3}, "Size mismatch.");
    utils::check(EJn.shape() == std::array<long,2>{nwalk,nCV}, "Size mismatch.");
    utils::check_strides(E,G);
    utils::check(haj.extent(0) == 1, "ph_reference_energy requires ndet=1");

    E() = ComplexType(0.0);
    // by convention, add E0 only to Alpha
    if(addH1 and (spin_component == Alpha)) E(all,0) = E0;
    ph_ref_energy_impl(ispin, E, G, EJn, addH1);
    nda::tensor::contract(ComplexType(0.5),EJn,"wn",EJn,"wn",ComplexType(1.0),E(all,2),"w"); 
  }

  void ph_excited_energy(SpinTypes spin_component,
              int nelec,
              nda::MemoryVector auto const& iexcit,
              nda::MemoryVector auto const& refc,
              nda::MemoryArrayOfRank<2> auto && E,
              nda::MemoryArrayOfRank<2> auto && wgt,
              nda::MemoryArrayOfRank<4> auto const& R,
              nda::MemoryArrayOfRank<3> auto && K,
              bool addH1 = true)
  {
    memory::check_memory_space<MEM>(E,wgt,R,K,iexcit,refc);
    utils::check_strides(E,wgt,R,K);
    using nda::range;
    auto all = range::all; 
    // R[nwalk, ndet, nex, nact] 
    // E[nwalk, 3]
    // wgt[ndet, nwalk]
    // K[ndet, nwalk, nkev]
    auto [nwalk, ndet, nex, nact] = R.shape();
  
    utils::check(E.shape() == std::array<long,2>{nwalk,3}, "Size mismatch");
    utils::check(wgt.shape() == std::array<long,2>{ndet,nwalk}, "Size mismatch");
    utils::check(K.shape() == std::array<long,3>{ndet,nwalk,nCV}, "Size mismatch");

    // by convention, add E0 only to Alpha
    /* One body terms */
    if(addH1) {
      utils::check(Swia_ph.extent(0) >= nwalk * nelec * nact, "Error in ph_excited_energy: Unexpected size in Swia_ph.");
      memory::array_view<MEM,ComplexType,3> Swia({nwalk,nelec,nact},Swia_ph.data());
      ph_excited_1body_energy(iexcit, refc, Swia, R, wgt, E(all,0));
    }

    /* Two body terms */
    // Make sure Twina_ph has appropriate dimensions
    utils::check(Twina_ph.extent(0) >= nwalk * nelec * nCV * nact, "Error in ph_excited_energy: Unexpected size in Twina_ph.");
    memory::array_view<MEM,ComplexType,4> Twina({nwalk,nelec,nCV,nact},Twina_ph.data());

    ph_excited_2body_energy_dense_cholesky(iexcit, refc, Twina, R, wgt, E(all,1), E(all,2), K);
  }

  auto vHS_sparse(nda::MemoryArrayOfRank<2> auto && X, double dt)
  {
    utils::check(false, "vHS_sparse not implemented in Real3IndexFactorization.");
    nda::array<math::sparse::csr_matrix<ComplexType,MEM,int,int>,1> spvHS;
    return spvHS();
  }

  auto vHS(nda::MemoryMatrix auto&& X, double dt)
  {
    constexpr MEMORY_SPACE MEM_X = memory::get_memory_space<decltype(X)>();
    static_assert(MEM == MEM_X, "Memory space mismatch");
    using nda::range;
    auto all = range::all;
    int nwalk = X.extent(0);
    int nstot = (walker_type == COLLINEAR) ? Likn.extent(0) : 1;
    int nptot = (walker_type == NONCOLLINEAR) ? Likn.extent(0) : 1;
    utils::check_strides(X);
    utils::check(X.shape() == std::array<long,2>{nwalk,nCV}, "Size mismatch.");

    // Note: Allocate first, to make better use of memory pool
    // vHS[nspin_in_vHS][nwalk][npol_in_vHS*NMO][NMO]
    memory::buffered_array<MEM_X,ComplexType,4> v(nstot,nwalk,nptot*NMO,NMO);
    v() = ComplexType(0.0);

    // scale by sqrt(dt)
    RealType a(std::sqrt(dt));

    if constexpr (MEM==HOST_MEMORY) {
      auto v4d = nda::reshape(v,std::array<long,4>{nstot,nwalk,nptot,NMO*NMO});;
      memory::buffered_array<MEM,ComplexType,2> Xt(nCV,nwalk);
      memory::buffered_array<MEM,ComplexType,2> vt(NMO*NMO,nwalk);
      Xt() = nda::transpose(X());
      for (int is = 0, isp=0; is < nstot; is++) {
        for (int ip = 0; ip < nptot; ip++, ++isp) {
          auto Ln = nda::reshape(Likn()(isp%(nstot*nptot),nda::ellipsis{}),std::array<long,2>{NMO*NMO,nCV});;
          nda::blas::gemm(a,Ln,Xt,RealType(0.0),vt);
          v4d(is,all,ip,all) = nda::transpose(vt()); 
        }
      }
    } else {
      auto Xr = memory::to_real_view(X);
      auto vr = memory::to_real_view(v);
      for (int is = 0, isp=0; is < nstot; is++) {
        for (int ip = 0; ip < nptot; ip++, ++isp) {
          auto Ln = Likn()(isp%(nstot*nptot),all,all,all);
          auto v_ = vr(is,all,range(ip*NMO,(ip+1)*NMO),all,all);
          nda::tensor::contract(a, Ln, "ijn", Xr, "wnc", RealType(1.0), v_, "wijc");
        }
      }
    }
    return v;
  }

  void vbias(nda::MemoryArrayOfRank<2> auto const& G, nda::MemoryArrayOfRank<2> auto& v, double dt)
  {
    memory::check_memory_space<MEM>(G,v);
    using nda::range;
    auto all = range::all;
    int nwalk = G.extent(0);
    int nspin = (walker_type == COLLINEAR) ? 2 : 1;
    int npol  = (walker_type == NONCOLLINEAR) ? 2 : 1;
    int nstot = (walker_type == COLLINEAR) ? Likn.extent(0) : 1;
    int nptot = (walker_type == NONCOLLINEAR) ? Likn.extent(0) : 1;
    int nCV   = Lnak(0).extent(2);
    int nel   = (walker_type == COLLINEAR ? nup+ndown : nup); // NONCOLLINEAR has ndown=0 
    utils::check_strides(G,v);
    // limiting G to contiguous arrays for simplicity now, reconsider if necessary
    utils::check(v.shape() == std::array<long,2>{nwalk,nCV}, "Real3IndexFactorization::vbias: Size mismatch.");
    // Dispatch on the G layout, not the trial determinant count. The half-rotated path takes a
    // compact G [nwalk, nel*npol*NMO]; the full path takes a full G [nwalk, nspin*npol*NMO*npol*NMO]
    // and contracts the un-rotated Likn. NOMSD passes a compact G for ndet==1 and a full G for
    // ndet>1, so existing behavior is unchanged; the stochastic trial reaches the full path with a
    // single-determinant trial (this is what the former vbias_fullG provided).
    long const half_size = long(nel)*npol*NMO;
    long const full_size = long(nspin)*npol*NMO*npol*NMO;
    utils::check(G.extent(1) == half_size || G.extent(1) == full_size,
                 "Real3IndexFactorization::vbias: Size mismatch.");
    bool const half_rotated = (G.extent(1) == half_size);
    utils::check(G.is_contiguous(), "Layout mismatch");

    // scale a by sqrt(dt)
    RealType a(std::sqrt(dt) * (walker_type == CLOSED ? 2.0 : 1.0));
    
    v() = ComplexType(0.0);

    if (half_rotated)
    {
      memory::array_view<MEM,const ComplexType,3> G3d(std::array<long,3>{nwalk,nel,npol*NMO},G.data());
      //Lnak(idet,ispin,ipol,n,a,k) * G(w,a,k)
      if constexpr (MEM==HOST_MEMORY) {
        if(npol==1 and nspin==1) {
          auto Ln = nda::reshape(Lnak(0)()(0,0,nda::ellipsis{}),std::array<long,2>{nCV,nup*NMO});
          nda::blas::gemm(ComplexType(a), G, nda::transpose(Ln), ComplexType(0.0), v);
        } else {
          for (int is = 0; is < nspin; is++) {
            for (int ip = 0; ip < npol; ip++) {
              for(int ia=0; ia<(is==0?nup:ndown); ++ia) {
                auto Ln = Lnak(is)()(0,ip,all,ia,all);
                auto G_ = G3d(all,is*nup+ia,range(ip*NMO,(ip+1)*NMO));
                nda::blas::gemm(ComplexType(a), G_(), nda::transpose(Ln()), ComplexType(1.0), v);
              }
            }
          }
        }
      } else {
        for (int is = 0; is < nspin; is++) {
          int nel_is = (is == 0 ? nup : ndown);
          // Empty spin sector, nothing to do.
          if(nel_is == 0) {
            continue;
          }
          for (int ip = 0; ip < npol; ip++) {
            auto Ln = Lnak(is)()(0,ip,all,range(nel_is),all);
            auto G_ = G3d(all,range(is*nup,nup+is*ndown),range(ip*NMO,(ip+1)*NMO));
            // KE: Likely need to transpose Ln, check on GPU build first!
            nda::tensor::contract(ComplexType(a), G_, "wak",  Ln, "nak", ComplexType(1.0), v, "wn");
          }
        }
      }
    }
    else
    {
      memory::array_view<MEM,const ComplexType,6> G6d(std::array<long,6>{nwalk,nspin,npol,NMO,npol,NMO},G.data());
      //Likn(nstot*nptot,NMO,NMO,nCV) * G(w,s,i,k)
      if constexpr (MEM==HOST_MEMORY) {
        memory::buffered_array<MEM,ComplexType,3> Gt(NMO,NMO,nwalk);
        auto Gt2d = nda::reshape(Gt,std::array<long,2>{NMO*NMO,nwalk});
        memory::buffered_array<MEM,ComplexType,2> vt(nCV,nwalk);
        if(npol==1) {
          memory::array_view<MEM,const ComplexType,3> Gwsij(std::array<long,3>{nwalk,nspin,NMO*NMO},G.data());
          for (int is = 0; is < nspin; is++) {
            auto Ln = nda::reshape(Likn()(is%nstot,nda::ellipsis{}), std::array<long,2>{NMO*NMO,nCV});
            Gt2d() = nda::transpose(Gwsij(all,is,all));
            nda::blas::gemm(a, nda::transpose(Ln), Gt2d, RealType(0.0), vt);
            v() += nda::transpose(vt());
          }
        } else {
          for (int is = 0, isp=0; is < nspin; is++) {
            for (int ip = 0; ip < npol; ip++, ++isp) {
              auto Ln = nda::reshape(Likn()(isp%(nstot*nptot),nda::ellipsis{}), std::array<long,2>{NMO*NMO,nCV});
              for(int iw=0; iw<nwalk; ++iw) 
                Gt(all,all,iw) = G6d(iw,is,ip,all,ip,all);
              nda::blas::gemm(a, nda::transpose(Ln), Gt2d, RealType(0.0), vt);
              v() += nda::transpose(vt());
            }  
          }  
        }
      } else {
        auto Gr = memory::to_real_view(G6d);
        auto vr = memory::to_real_view(v);
        for (int is = 0, isp=0; is < nspin; is++) {
          for (int ip = 0; ip < npol; ip++, ++isp) {
            auto Ln = Likn()(isp%(nstot*nptot),all,all,all);
            auto G_ = Gr(all,is,ip,all,ip,all,all);
            nda::tensor::contract(a, Ln, "ijn", G_, "wijc", RealType(1.0), vr, "wnc");
          }
        }
      }
    }
  }

  void generalizedFockMatrix(nda::MemoryMatrix auto && G, 
                             nda::MemoryMatrix auto&& Fp, 
                             nda::MemoryMatrix auto&& Fm)
  {
/*
    int nwalk = G.size(0);
    int nspin = (walker_type == COLLINEAR ? 2 : 1);
    int NMO   = hij.size(1);
    utils::check(Fp.size(0) == nwalk, "");
    utils::check(Fm.size(0) == nwalk, "");
    utils::check(G[0].num_elements() == nspin * NMO * NMO, "");
    utils::check(Fp[0].num_elements() == nspin * NMO * NMO, "");
    utils::check(Fm[0].num_elements() == nspin * NMO * NMO, "");

    if(hij.size(0) == 2*NMO)
      APP_ABORT(" Error: generalizedFockMatrix not implemented with spin dependent H1.");

    // Rwn[nwalk][nCV]: 1+nspin copies
    // Twpqn[nwalk][NMO][NMO][nCV]: 1+nspin copies
    // extra copies

    // can you find out how much memory is available on the buffer?
    long LBytes = max_memory_MB * 1024L * 1024L / long(sizeof(SPComplexType));
    if constexpr (MP) {
      LBytes -= long((3 * nspin + 1) * nwalk * NMO * NMO); // G, Fp, Fm and Gt
    } else {
      LBytes -= long((1 + nspin) * nwalk * NMO * NMO); //  G and Gt
    }
    LBytes *= long(sizeof(SPComplexType));
    int Bytes = int(LBytes / long(2 * (NMO * NMO + 1) * local_nCV * sizeof(SPComplexType)));
    int nwmax = std::min(std::max(1, Bytes), nwalk);
    utils::check(nwmax >= 1 && nwmax <= nwalk, "");

    sp_pointer ptr_Fp(nullptr);
    sp_pointer ptr_Fm(nullptr);
    auto buffer_alloc=buffer_manager.get_generator().template get_allocator<SPComplexType>();
    if constexpr (MP) {
      buffer_alloc.deallocate(ptr_Fp, nwalk * nspin * NMO * NMO);
      buffer_alloc.deallocate(ptr_Fp, nwalk * nspin * NMO * NMO);
    } else {
      ptr_Fm = make_device_ptr(Fm.origin());
      ptr_Fp = make_device_ptr(Fp.origin());
    }
    SpCMatrix_ref Fp_(ptr_Fp, {nwalk, nspin * NMO * NMO});
    SpCMatrix_ref Fm_(ptr_Fm, {nwalk, nspin * NMO * NMO});
    fill_n(Fp_.origin(), Fp_.num_elements(), SPComplexType(0.0));
    fill_n(Fm_.origin(), Fm_.num_elements(), SPComplexType(0.0));

    SPComplexType scl = (walker_type == CLOSED ? 2.0 : 1.0);
    std::vector<sp_pointer> Aarray;
    std::vector<sp_pointer> Barray;
    std::vector<sp_pointer> Carray;
    Aarray.reserve(nwalk);
    Barray.reserve(nwalk);
    Carray.reserve(nwalk);

    long gsz(0);
    if constexpr (MP) {
      gsz = nspin * nwmax * NMO * NMO;
    } else {
      if (nspin > 1)
        gsz = nspin * nwmax * NMO * NMO;  
    }
    StaticSpVector GBuff(iextensions<1u>{gsz}, buffer_manager.get_generator().template get_allocator<SPComplexType>());

    int nw0(0);
    while (nw0 < nwalk)
    {
      int nw = std::min(nwalk - nw0, nwmax);

      // transpose/cast G
      sp_pointer ptr(nullptr);
      if constexpr (MP) {
        ptr = GBuff.origin();
        for (int ispin = 0, is0 = 0, ip = 0; ispin < nspin; ispin++, is0 += NMO * NMO)
          for (int n = 0; n < nw; ++n, ip += NMO * NMO)
            copy_n_cast(make_device_ptr(G[nw0 + n].origin()) + is0, NMO * NMO, ptr + ip);
      } else {
        if (nspin == 1)
        {
          ptr = make_device_ptr(G[nw0].origin());
        }
        else
        {
          ptr = GBuff.origin();
          using std::copy_n;
          for (int ispin = 0, is0 = 0, ip = 0; ispin < nspin; ispin++, is0 += NMO * NMO)
            for (int n = 0; n < nw; ++n, ip += NMO * NMO)
              copy_n(make_device_ptr(G[nw0 + n].origin()) + is0, NMO * NMO, ptr + ip);
        }
      }
      SpCTensor_ref GF(ptr, {nspin, nw * NMO, NMO}); // now contains G in the correct structure [spin][w][i][j]
      StaticSpMatrix Gt({NMO * NMO, nw}, buffer_manager.get_generator().template get_allocator<SPComplexType>());
      fill_n(Gt.origin(), Gt.num_elements(), SPComplexType(0.0));

      StaticSpMatrix Rnw({local_nCV, nw}, buffer_manager.get_generator().template get_allocator<SPComplexType>());
      // calculate Rwn
      for (int ispin = 0; ispin < nspin; ispin++)
      {
        SpCMatrix_ref G_(GF[ispin].origin(), {nw, NMO * NMO});
        ma::add(SPComplexType(1.0), Gt, SPComplexType(1.0), ma::T(G_), Gt);
      }
      // R[n,w] = \sum_ik L[n,ik] G[ik,w]
      ma::product(SPRealType(1.0), ma::T(Likn), Gt, SPRealType(0.0), Rnw);
      StaticSpMatrix Rwn({nw, local_nCV}, buffer_manager.get_generator().template get_allocator<SPComplexType>());
      ma::transpose(Rnw, Rwn);

      // add coulomb contribution of <pr||qs>Grs term to Fp, reuse Gt for temporary storage
      // Fp[p,t] = \sum_{jl} L[p,t,n] L[j,l,n] P[j,l]
      // Fp[pt,w] = \sum_n L[pt,n] R[n,w]
      ma::product(SPRealType(1.0), Likn, Rnw, SPRealType(0.0), Gt);
      for (int ispin = 0; ispin < nspin; ispin++)
      {
        ma::add(SPComplexType(1.0), Fp_({nw0, nw0 + nw}, {ispin * NMO * NMO, (ispin + 1) * NMO * NMO}),
                SPComplexType(scl), ma::T(Gt), Fp_({nw0, nw0 + nw}, {ispin * NMO * NMO, (ispin + 1) * NMO * NMO}));
      }

      // L[i,kn]
      SpRMatrix_ref Ln(make_device_ptr(Likn.origin()), {NMO, NMO * local_nCV});
      // T[w,p,t,n] = \sum_{l} L[l,t,n] P[w,l,p]
      StaticSpMatrix Twptn({nw * NMO, NMO * local_nCV},
                         buffer_manager.get_generator().template get_allocator<SPComplexType>());
      // transpose for faster contraction
      StaticSpMatrix Taux({nw * NMO, NMO * local_nCV},
                        buffer_manager.get_generator().template get_allocator<SPComplexType>());
      SpCMatrix_ref Ttnwp(Taux.origin(), {NMO * local_nCV, nw * NMO});
      // Twptn3D: {nw, NMO * NMO, local_nCV} 
      auto&& Twptn3D= Twptn.partitioned(nw).rotated().flatted().unrotated(); 	
      // Twptn4D: {nw, NMO, NMO, local_nCV} 
      auto&& Twptn4D= Twptn.rotated().partitioned(NMO).unrotated().partitioned(nw); 	
      // Taux4D: {nw, NMO, NMO, local_nCV}	
      auto&& Taux4D= Taux.rotated().partitioned(NMO).unrotated().partitioned(nw); 	
      SpCMatrix_ref Gt_(Gt.origin(), {NMO, nw * NMO});

      for (int ispin = 0, is0 = 0; ispin < nspin; ispin++, is0 += NMO * NMO)
      {
        SpCMatrix_ref G_(GF[ispin].origin(), {nw * NMO, NMO});
        ma::transpose(G_, Gt_);

        // J = \sum_{iklr} L[i,k,n] L[q,l,n] P[s,p,l] P[r,i,k]
        // R[n] = \sum_{ik} L[i,k,n] P[r,i,k]
        // Here T[tn,wp] = \sum_{l} L[tn,l] P[l,wp]
        // T[ln,p] = T[npl] = L[nkl] P[p,l]
        ma::product(SPRealType(1.0), ma::T(Ln), Gt_, SPRealType(0.0), Ttnwp);
        // T[wp,tn]
        ma::transpose(Ttnwp, Twptn);

        // transpose Twptn -> Twtpn=Taux
        // T[wt,pn]
	ma::transpose_wabn_to_wban(Twptn4D, Taux4D);

        // add exchange component to Fm_
        Aarray.clear();
        Barray.clear();
        Carray.clear();
        for (int w = 0; w < nw; w++)
        {
          Aarray.push_back(Taux.partitioned(nw)[w].origin());
          Barray.push_back(Twptn.partitioned(nw)[w].origin());
          Carray.push_back(Fm_[w].origin() + is0);
        }
        using ma::gemmBatched;
        // careful with expected Fortran ordering here!!!
        // K[p,q] = \sum_{ln} T[n,l,p] T[n,q,l]
        //          \sum_{ln} T[nl,p] T[nl,q]
        gemmBatched('T', 'N', NMO, NMO, NMO * local_nCV, SPComplexType(1.0), Aarray.data(), NMO * local_nCV,
                    Barray.data(), NMO * local_nCV, SPComplexType(1.0), Carray.data(), NMO, nw);

        // add coulomb component to Fm_
        Aarray.clear();
        Barray.clear();
        Carray.clear();
        for (int w = 0; w < nw; w++)
        {
          Aarray.push_back(Twptn3D[w].origin());
          Barray.push_back(Rwn[w].origin());
          Carray.push_back(Fm_[w].origin() + is0);
        }
        using ma::gemmBatched;
        // careful with expected Fortran ordering here!!!
        // J[w][pq] = \sum_{n} T[w][pq,n] R[w][n]
        gemmBatched('T', 'N', NMO * NMO, 1, local_nCV, SPComplexType(-1.0) * scl, Aarray.data(), local_nCV,
                    Barray.data(), local_nCV, SPComplexType(1.0), Carray.data(), NMO * NMO, nw);

        // Fp
        // Need Gt_[i][wj]
	ma::transpose_wabn_to_wban( G_.rotated().partitioned(NMO).unrotated().partitioned(1), 
				    Gt_.rotated().partitioned(nw).unrotated().partitioned(1));
        ma::product(SPRealType(1.0), ma::T(Ln), Gt_, SPRealType(0.0), Ttnwp);
        ma::transpose(Ttnwp, Twptn);
        ma::transpose_wabn_to_wban(Twptn4D, Taux4D); 
        // add coulomb component to Fp_, same as Fm_ above
        Aarray.clear();
        Barray.clear();
        Carray.clear();
        for (int w = 0; w < nw; w++)
        {
          Aarray.push_back(Twptn3D[w].origin());
          Barray.push_back(Rwn[w].origin());
          Carray.push_back(Fp_[nw0+w].origin() + is0);
        }
        using ma::gemmBatched;
        // careful with expected Fortran ordering here!!!
        // Coulomb component
        gemmBatched('T', 'N', NMO * NMO, 1, local_nCV, SPComplexType(-1.0) * scl, Aarray.data(), local_nCV,
                    Barray.data(), local_nCV, SPComplexType(1.0), Carray.data(), NMO * NMO, nw);

        // add exchange component of Fp_
        Aarray.clear();
        Barray.clear();
        Carray.clear();
        for (int w = 0; w < nw; w++)
        {
          Aarray.push_back(Taux.partitioned(nw)[w].origin());
          Barray.push_back(Twptn.partitioned(nw)[w].origin());
          Carray.push_back(Fp_[nw0+w].origin() + is0);

          // add exchange contribution of <pr||qs>Grs term by adding Lptn to Twptn
          // dispatch directly from here to be able to add to the real part only
          // K1B[p,q] = -\sum_{jl} L[jt,n] L[pl,n] P[jl]
          ma::axpy(Likn.num_elements(), SPRealType(-1.0), ma::pointer_dispatch(Likn.origin()), 1,
               reinterpret_pointer_cast<SPRealType>(ma::pointer_dispatch(Twptn3D[w].origin())), 2,
	       ma::select_backend<shmSpRMatrix>());
        }
        using ma::gemmBatched;
        // careful with expected Fortran ordering here!!!
        gemmBatched('T', 'N', NMO, NMO, NMO * local_nCV, SPComplexType(1.0), Aarray.data(), NMO * local_nCV,
                    Barray.data(), NMO * local_nCV, SPComplexType(1.0), Carray.data(), NMO, nw);

      } // ispin

      nw0 += nw;
    }

    if constexpr (MP) {
      copy_n_cast(Fp_.origin(), Fp_.num_elements(), make_device_ptr(Fp.origin()));
      copy_n_cast(Fm_.origin(), Fm_.num_elements(), make_device_ptr(Fm.origin()));
      buffer_alloc.deallocate(ptr_Fp, nwalk * nspin * NMO * NMO);	
      buffer_alloc.deallocate(ptr_Fm, nwalk * nspin * NMO * NMO);	
    }

    //fill_n(Fp.origin(),Fp.num_elements(),SPComplexType(0.0));
    //fill_n(Fm.origin(),Fm.num_elements(),SPComplexType(0.0));
    // add one body terms now
    {
      std::vector<pointer> Aarr;
      std::vector<pointer> Barr;
      std::vector<pointer> Carr;
      Aarr.reserve(nspin * nwalk);
      Barr.reserve(nspin * nwalk);
      Carr.reserve(nspin * nwalk);
      // Fm -= G[w][p][r] * h[q][r]
      Aarr.clear();
      Barr.clear();
      Carr.clear();
      for (int ispin = 0, is0 = 0; ispin < nspin; ispin++, is0 += NMO * NMO)
      {
        for (int w = 0; w < nwalk; w++)
        {
          Aarr.push_back(make_device_ptr(hij_dev.origin()));
          Barr.push_back(make_device_ptr(G[w].origin()) + is0);
          Carr.push_back(make_device_ptr(Fm[w].origin()) + is0);
        }
      }
      using ma::gemmBatched;
      // careful with expected Fortran ordering here!!!
      gemmBatched('T', 'N', NMO, NMO, NMO, ComplexType(-1.0), Aarr.data(), NMO, Barr.data(), NMO, ComplexType(1.0),
                  Carr.data(), NMO, Aarr.size());


      // Fp -= G[w][r][p] * h[q][r]
      Aarr.clear();
      Barr.clear();
      Carr.clear();
      C4Tensor_ref Fp4D(make_device_ptr(Fp.origin()), {nwalk, nspin, NMO, NMO});
      for (int ispin = 0, is0 = 0; ispin < nspin; ispin++, is0 += NMO * NMO)
      {
        for (int w = 0; w < nwalk; w++)
        {
          Aarr.push_back(make_device_ptr(hij_dev.origin()));
          Barr.push_back(make_device_ptr(G[w].origin()) + is0);
          Carr.push_back(make_device_ptr(Fp[w].origin()) + is0);

          // add diagonal contribution to Fp
          ma::add(ComplexType(1.0), Fp4D[w][ispin], ComplexType(1.0), ma::T(hij_dev), Fp4D[w][ispin]);
        }
      }
      using ma::gemmBatched;
      // careful with expected Fortran ordering here!!!
      gemmBatched('T', 'T', NMO, NMO, NMO, ComplexType(-1.0), Aarr.data(), NMO, Barr.data(), NMO, ComplexType(1.0),
                  Carr.data(), NMO, Aarr.size());
    }
*/
  }

  std::tuple<int,int> vHS_dims() const {
    int nstot = (walker_type == COLLINEAR) ? Likn.extent(0) : 1;
    int nptot = (walker_type == NONCOLLINEAR) ? Likn.extent(0) : 1;
    return std::make_tuple(nstot,nptot);
  }
  int number_of_ke_vectors() const { return nCV; }
  int number_of_cholesky_vectors() const { return nCV; }

  nda::array<ComplexType, 2> getHSPotentials()
  { return nda::array<ComplexType, 2>{}; }

  void energy_fullG(nda::MemoryArrayOfRank<2> auto && E,
                    nda::MemoryArrayOfRank<2> auto const& Gfull,
                    bool addH1 = true,
                    bool addEJ = true,
                    bool addEXX = true)
  {
    if (walker_type == NONCOLLINEAR)
      APP_ABORT("Real3IndexFactorization::energy_fullG: NONCOLLINEAR full-G is not implemented.");
    // The un-rotated full-G energy kernels contract the bare Cholesky built by ensure_full_cholesky()
    // from Likn(0). For COLLINEAR both spins reuse that single block, so the Cholesky must be
    // spin-independent (Likn.extent(0)==1 -- the case for a standard molecular UHF Hamiltonian, and the
    // only case that reaches this stochastic single-determinant full-G path in practice). vbias's full-G
    // branch DOES handle a spin-dependent Cholesky (Likn(is%nstot) per spin) because it is shared with
    // the multi-determinant NOMSD trial; the stochastic energy kernel does not yet, so a spin-dependent
    // COLLINEAR Hamiltonian (Likn.extent(0)==2, e.g. spin-resolved integrals) is rejected here rather
    // than silently scoring both spins with the alpha block. Fail-fast: this fires on the first energy
    // evaluation, before any measurement, so no incorrect physics is reported. Lifting it needs per-spin
    // Cholesky densification in energy_collinear (mirroring energy_impl / vbias) -- deferred as it is
    // untestable without a spin-dependent DenseFactorized fixture (none exists) and unneeded for
    // molecular UHF.
    if (walker_type == COLLINEAR)
    {
      utils::check(Likn.extent(0) == 1,
                   "Real3IndexFactorization::energy_fullG: COLLINEAR full-G currently requires a "
                   "spin-independent Cholesky (Likn.extent(0)==1); spin-dependent (per-spin) Cholesky "
                   "in the stochastic energy kernel is a follow-up.");
      // The SAME restriction applies to the ONE-BODY and was missing: ensure_full_cholesky() flattens
      // hij(0, i, k) -- spin index 0, unconditionally -- and energy_collinear then contracts that single
      // h against BOTH spin blocks of G. With a genuinely spin-dependent H1 that silently scores the beta
      // spin with h_alpha, and nothing downstream would flag it. Fail fast for the same reason the
      // Cholesky check does: it fires on the first energy evaluation, before any measurement.
      //
      // Test the VALUES, not the storage shape. `hij.extent(0) == 1` would be the wrong condition:
      // Hamiltonians legitimately store two IDENTICAL spin blocks (the BH test fixture ships hcore as
      // (2*NMO, NMO) with max|h_alpha - h_beta| == 0 exactly), and flattening block 0 there is exact.
      // Rejecting those would refuse correct input -- and would mask any real defect behind an abort.
      auto h = hij();
      if (h.extent(0) > 1)
      {
        double dmax = 0.0;
        for (long i = 0; i < h.extent(1); ++i)
          for (long k = 0; k < h.extent(2); ++k)
            dmax = std::max(dmax, std::abs(h(0, i, k) - h(1, i, k)));
        utils::check(dmax <= 1e-12,
                     "Real3IndexFactorization::energy_fullG: COLLINEAR full-G requires a spin-independent "
                     "one-body, but max|h_alpha - h_beta| exceeds 1e-12. ensure_full_cholesky() flattens "
                     "spin 0 only, so the beta spin would be scored with h_alpha. Per-spin H1 in the "
                     "stochastic full-G energy kernel is a follow-up.");
      }
    }
    ensure_full_cholesky();
    if (walker_type == COLLINEAR)
      full_g::energy_collinear<MEM>(mpi, std::forward<decltype(E)>(E), Gfull, Lank_full_flat_,
                                    hij_full_flat_, int(nCV), E0, addH1, addEJ, addEXX);
    else
      full_g::energy_closed<MEM>(mpi, std::forward<decltype(E)>(E), Gfull, Lank_full_flat_,
                                 hij_full_flat_, int(nCV), E0, addH1, addEJ, addEXX);
  }

private:
  std::shared_ptr<utils::mpi_context_t<mpi3::communicator>> mpi;

  WALKER_TYPES walker_type;

  int NMO, nup, ndown;
  int nCV;
  long max_memory_MB = 2000;

  // bare one body hamiltonian
  memory::const_shared_array<HOST_MEMORY,ComplexType,3> hij;

  // (potentially half rotated) one body hamiltonian
  memory::const_shared_array<MEM,ComplexType,3> haj;

  //Cholesky Tensor Lik(nstot_H2*nptot_H2,NMO,NMO,nCV)
  memory::const_shared_array<MEM,RealType,4> Likn;

  // half-tranformed Cholesky tensor
  // Lnak(ispin)(idet,ipol,n,a,k)
  nda::array<memory::const_shared_array<MEM,ComplexType,5>,1> Lnak;

  // one-body piece of Hamiltonian factorization
  memory::const_shared_array<HOST_MEMORY,RealType,3> vexx;

  // used in PH reference/excited energy evaluation
  // Twian = sum_k G_ref[w][i][k] L[a][n][k]
  memory::array<MEM,ComplexType,1> Twina_ph;
  // Swia = sum_k G_ref[w][i][k] h[a][k]
  memory::array<MEM,ComplexType,1> Swia_ph;

  bool full_cholesky_ready_{false};
  memory::array<MEM,ComplexType,2> Lank_full_flat_;
  memory::array<MEM,ComplexType,1> hij_full_flat_;

  void ensure_full_cholesky()
  {
    if (full_cholesky_ready_)
      return;
    // Build both flat arrays on HOST (element assembly), then move to the MEM space in one shot. Writing
    // element-by-element into a device array from host is a segfault, so never assemble into the MEM member
    // directly (this path is first exercised on device by the PR-3 dynamic full-G port).
    auto Lank_host = memory::array<HOST_MEMORY, ComplexType, 2>(NMO * nCV, NMO);
    auto Lhost = nda::to_host(Likn()(0, nda::range::all, nda::range::all, nda::range::all));
    for (int i = 0; i < NMO; ++i)
      for (int k = 0; k < NMO; ++k)
        for (int nc = 0; nc < nCV; ++nc)
          Lank_host(i * nCV + nc, k) = ComplexType(Lhost(i, k, nc));
    auto hij_host = memory::array<HOST_MEMORY, ComplexType, 1>(NMO * NMO);
    auto hij_h = hij();
    for (int i = 0; i < NMO; ++i)
      for (int k = 0; k < NMO; ++k)
        hij_host(i * NMO + k) = hij_h(0, i, k);
    if constexpr (MEM != HOST_MEMORY)
    {
#if defined(ENABLE_DEVICE)
      Lank_full_flat_ = nda::to_device(Lank_host);
      hij_full_flat_  = nda::to_device(hij_host);
#else
      static_assert(MEM == HOST_MEMORY, "Device memory requires ENABLE_DEVICE");
#endif
    }
    else
    {
      Lank_full_flat_ = Lank_host;
      hij_full_flat_  = hij_host;
    }
    full_cholesky_ready_ = true;
  } 

  // zero of energy 
  ComplexType E0;

  void energy_impl(int ispin,
              nda::MemoryMatrix auto&& E,
              nda::MemoryArrayOfRank<3> auto const& G,
              int idet,
              nda::MemoryMatrix auto && Kl,
              bool addH1  = true,
              bool addEJ  = true,
              bool addEXX = true)
  {
    using nda::range;
    auto all = range::all;
    utils::check(ispin==0 or ispin==1, "Invalid argument");
    utils::check(idet >= 0 and idet < haj.extent(0), "Invalid idet");

    int nwalk = G.extent(0);
    int npol = (walker_type == NONCOLLINEAR ? 2 : 1);
    int nel[2] = {nup,ndown};
    RealType scl = (walker_type == CLOSED ? 2.0 : 1.0);
    utils::check(G.shape() == std::array<long,3>{nwalk,nel[ispin],npol*NMO}, "Size mismatch");
    utils::check(E.extent(0)==nwalk and E.extent(1)==3, "Size mismatch");
    if (addEJ)
      utils::check(Kl.extent(0) == nwalk and Kl.extent(1) == nCV, "Size mismatch");

    // Empty spin sector, nothing to do.
    if (nel[ispin] == 0)
      return;

    // one-body contribution
    // haj(ndet,nel,npol*nmo)
    if (addH1)
      nda::tensor::contract(ComplexType(scl),G,"waj",
                            haj()(idet,range(ispin*nup,nup+ispin*ndown),all),"aj",
                            ComplexType(1.0),E(all,0),"w");
    if (addEXX)
    {
      // MAM: if the memory available in the MR is larger than max_memory_MB, use it!
      int max_nCV = 0;
      long LBytes = max_memory_MB * 1024L * 1024L;
      int Bytes   = int(LBytes / long(nwalk * nel[ispin] * nel[ispin] * sizeof(ComplexType)));
      max_nCV     = std::min(std::max(1, Bytes), nCV);
      utils::check(max_nCV > 0 and max_nCV <= nCV, "Logic error!!!");

      // [ndet,npol,nCV,nup,NMO]
      auto L3d = nda::reshape(Lnak(ispin)()(idet,nda::ellipsis{}), 
                              std::array<long,3>{npol,nCV*nel[ispin],NMO});
 
      // contiguous G to allow use of gemm
      bool needs_Gc = (MEM==HOST_MEMORY and not (npol==1 and G.is_contiguous()));
      memory::buffered_array<MEM,ComplexType,3> Gc((needs_Gc?nwalk:0),nel[ispin],npol*NMO); 
      if(needs_Gc)
        Gc() = G();

      int nv = 0;
      while (nv < nCV)
      {
        int nvecs = std::min(nCV - nv, max_nCV);
        // L(ndet,nspin,npol,nCV,nel,NMO)

        if constexpr (MEM==HOST_MEMORY) {

          memory::buffered_array<MEM,ComplexType,4> Twbna(nwalk,nel[ispin],nvecs,nel[ispin]); 
          auto T2d = nda::reshape(Twbna, std::array<long,2>{nwalk*nel[ispin],nvecs*nel[ispin]});

          if(npol==1 and G.is_contiguous()) {
            auto G2d = nda::reshape(G, std::array<long,2>{nwalk*nel[ispin],NMO});
            nda::blas::gemm(G2d,nda::transpose(L3d(0,range(nv*nel[ispin],(nv+nvecs)*nel[ispin]),all)),T2d);
          } else {
            auto G3d = nda::reshape(Gc, std::array<long,3>{nwalk*nel[ispin],npol,NMO});
            T2d() = ComplexType(0.0);
            for(int ip=0; ip<npol; ++ip) 
              nda::blas::gemm(ComplexType(1.0),G3d(all,ip,all),nda::transpose(L3d(ip,range(nv*nel[ispin],(nv+nvecs)*nel[ispin]),all)),ComplexType(1.0),T2d);
          }
          for(int iw=0; iw<nwalk; ++iw)
            for(int ia=0; ia<nel[ispin]; ++ia) {
              E(iw,1) += ComplexType(-0.5*scl)*nda::blas::dot(Twbna(iw,ia,all,ia),Twbna(iw,ia,all,ia));
              for(int ib=ia+1; ib<nel[ispin]; ++ib)
                E(iw,1) += ComplexType(-scl)*nda::blas::dot(Twbna(iw,ia,all,ib),Twbna(iw,ib,all,ia));
            }

          if (addEJ) {
            for(int iw=0; iw<nwalk; ++iw)
              for(int ia=0; ia<nel[ispin]; ++ia)
                Kl(iw,range(nv,nv+nvecs)) += Twbna(iw,ia,all,ia);
          }

        } else {

          memory::buffered_array<MEM,ComplexType,4> Twbna(nwalk,nel[ispin],nvecs,nel[ispin]); 
          auto Lna = Lnak(ispin)()(idet,all,range(nv,nv+nvecs),range(nel[ispin]),all);
          // G[nwalk,nel[ispin],npol*NMO]
          std::array<long,4> shape = {nwalk,nel[ispin],npol,NMO};
          auto str = G.strides();
          std::array<long,4> strides = {str[0],str[1],NMO,1};
          nda::idx_map<4, 0, nda::C_stride_order<4>, nda::layout_prop_e::none> idxm(shape,strides);
          auto G4d = memory::array_view<MEM,const ComplexType,4>(idxm, G.data());
          nda::tensor::contract(ComplexType(1.0),G4d,"wbpk",Lna,"pnak",ComplexType(0.0),Twbna,"wbna");

          // E[w] = -0.5*scl* sum_abn Twanb * Twbna
          nda::tensor::contract(ComplexType(-0.5*scl),Twbna,"wanb",Twbna,"wbna",
                                ComplexType(1.0),E(all,1),"w");

          if (addEJ) {
            // kernel!!!
            for(int ia=0; ia<nel[ispin]; ++ia)
              nda::tensor::add(ComplexType(1.0),Twbna(all,ia,all,ia),"wn",
                               ComplexType(1.0),Kl(all,range(nv,nv+nvecs)),"wn");
          }

        }

        nv += max_nCV;
      }
    }
    utils::check(not addEJ or addEXX," Error: addEJ and not addEXX not yet implemented. \n\n");
  }  // energy_impl

  // similar to energy_impl, but generates Twina_ph to be used subsequently in
  // ph_excited_energy.  
  // Due to the nature of the implementation, looping over blocks of CV is not possible.
  // There has to be enough memory to do everything at once
  // Careful here! G is the Green's function of the reference!
  // It is size {nwalk, #electrons, NMO} and not! {nwalk, nactive, NMO} like Lnak!!! 
  void ph_ref_energy_impl(int is,
              nda::MemoryArrayOfRank<2> auto && E,
              nda::MemoryArrayOfRank<2> auto const& G2d,
              nda::MemoryArrayOfRank<2> auto && Kl,
              bool addH1  = true)
  {
    using nda::range;
    auto all = range::all;
    utils::check(is==0 or is==1, "Invalid argument");
    utils::check(haj.extent(0) == 1, "Calling ph_ref_energy_impl with ndet>1.");

    int nwalk = G2d.extent(0);
    int npol = (walker_type == NONCOLLINEAR ? 2 : 1);
    int nel = G2d.extent(1)/(npol*NMO);
    int nact[2] = {nup,ndown};
    RealType scl = (walker_type == CLOSED ? 2.0 : 1.0);

    utils::check(G2d.shape() == std::array<long,2>{nwalk,nel*npol*NMO}, "Size mismatch");
    utils::check(E.extent(0)==nwalk and E.extent(1)==3, "Size mismatch");
    utils::check(Kl.extent(0) == nwalk and Kl.extent(1) == nCV, "Size mismatch");

    // MAM: write routine to partition dimension, which is always possible, this removes
    //      the requirement to be contiguous. See energy_impl for example 
    utils::check(G2d.is_contiguous(), "Layout mismatch");
    memory::array_view<MEM,const ComplexType,3> G(std::array<long,3>{nwalk,nel,npol*NMO},G2d.data());

    // one-body contribution
    // haj(ndet,nact,npol*NMO)
    if (addH1)
    {
      if(Swia_ph.extent(0) < nwalk*nel*std::max(nup,ndown)) 
        Swia_ph = memory::array<MEM,ComplexType,1>(nwalk*nel*std::max(nup,ndown)); 
                           
      memory::array_view<MEM,ComplexType,3> Swia(std::array<long,3>{nwalk,nel,nact[is]},Swia_ph.data());
      auto G_ = nda::reshape(G2d, std::array<long,2>{nwalk*nel,npol*NMO});
      auto S2d = nda::reshape(Swia, std::array<long,2>{nwalk*nel,nact[is]});
      nda::blas::gemm(G_,nda::transpose(haj()(0,range(is*nup,nup+is*ndown),all)),S2d);

      // right now this only works if the reference configuration is refc[i] = i!!!!
      // need refc array otherwise
      //E[iw][0] += Swia[iw][i][refc[i]];
      if constexpr (MEM==HOST_MEMORY) {
        for(int iw=0; iw<nwalk; iw++)
          for(int i=0; i<nel; i++)
            E(iw,0) += Swia(iw,i,i);
      } else {
// build 2d array with diagonal on second dimension, then call reduce
        for(int i=0; i<nel; i++)
          nda::tensor::add(ComplexType(1.0),Swia(all,i,i),"w",ComplexType(1.0),E(all,0),"w");
      }
    }

    // try to catch exception???
    if(Twina_ph.extent(0) < nwalk*nel*nCV*std::max(nup,ndown)) 
      Twina_ph = memory::array<MEM,ComplexType,1>(nwalk*nel*nCV*std::max(nup,ndown)); 

    memory::array_view<MEM,ComplexType,4> Twina({nwalk,nel,nCV,nact[is]},Twina_ph.data());

    if constexpr (MEM==HOST_MEMORY) {

      // fix polarization!!!
      auto G3d = nda::reshape(G2d, std::array<long,3>{nwalk*nel,npol,NMO});
      auto L3d = nda::reshape(Lnak(is)()(0,nda::ellipsis{}),
                              std::array<long,3>{npol,nCV*nact[is],NMO});
      auto T2d = nda::reshape(Twina, std::array<long,2>{nwalk*nel,nCV*nact[is]});
      T2d() = ComplexType(0.0);
      for(int ip=0; ip<npol; ip++)
        nda::blas::gemm(ComplexType(1.0),G3d(all,ip,all),nda::transpose(L3d(ip,all,all)),
                        ComplexType(1.0),T2d);

      for(int iw=0; iw<nwalk; ++iw)
        for(int ia=0; ia<nel; ++ia) {
          E(iw,1) += ComplexType(-0.5*scl)*nda::blas::dot(Twina(iw,ia,all,ia),Twina(iw,ia,all,ia));
          for(int ib=ia+1; ib<nel; ++ib)
            E(iw,1) += ComplexType(-scl)*nda::blas::dot(Twina(iw,ia,all,ib),Twina(iw,ib,all,ia));
        }

      for(int iw=0; iw<nwalk; ++iw)
        for(int ia=0; ia<nel; ++ia)
          Kl(iw,all) += Twina(iw,ia,all,ia);

    } else {

      auto Lna = Lnak(is)()(0,nda::ellipsis{});
      auto G4d = nda::reshape(G2d, std::array<long,4>{nwalk,nel,npol,NMO});
      nda::tensor::contract(ComplexType(1.0),G4d,"wipk",Lna,"pnak",ComplexType(0.0),Twina,"wina");

      // E[w] = -0.5*scl* sum_abn Twanb * Twina
      nda::tensor::contract(ComplexType(-0.5*scl),Twina(all,all,all,range(nel)),"winj",
                                                  Twina(all,all,all,range(nel)),"wjni",
                            ComplexType(1.0),E(all,1),"w");

      // need kernel!!! 
      for(int ia=0; ia<nel; ++ia)
        nda::tensor::add(ComplexType(1.0),Twina(all,ia,all,ia),"wn",
                         ComplexType(1.0),Kl,"wn");
    }

  }  // ph_ref_energy_impl 

};

} // namespace afqmc

} // namespace sfqmc

