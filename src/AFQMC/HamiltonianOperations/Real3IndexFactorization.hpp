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
#include "IO/banner.hpp"
#include "nda/nda.hpp"
#include "nda/tensor.hpp"
#include "utilities/check.hpp"
#include "utilities/freemem.h"
#include "utilities/memory_utils.hpp"
#include "utilities/mpi_context.h"
#include "utilities/check_strides.hpp"
#include "numerics/shared_array/const_shared_array.hpp"
#include "numerics/nda_functions.hpp"
#include "detail/one_body.hpp"

#include "AFQMC/Wavefunctions/detail/phmsd_impl.hpp"

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
    app_log(1, section("Static memory usage by Real3IndexFactorization (node 0)"));
    app_log(1,"Likn: {}", utils::format_bytes(Likn.size() * sizeof(RealType)));
    app_log(1,"Lnak: {}", utils::format_bytes((Lnak(0).size() + (nspin==2?Lnak(1).size():0.0)) * sizeof(ComplexType)));
    app_log(1,"Buffer memory limited to (not yet allocated) : {} MB", max_memory_MB);
    utils::memory_report();
  }

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
    if(haj.extent(0) == 1) // ndet==1, G half rotated
      utils::check(G.extent(1) == nel*npol*NMO, "Real3IndexFactorization::vbias: Size mismatch.");
    else // ndet>1, full G 
      utils::check(G.extent(1) == nspin*npol*NMO*npol*NMO, "Real3IndexFactorization::vbias: Size mismatch.");
    utils::check(G.is_contiguous(), "Layout mismatch");

    // scale a by sqrt(dt)
    RealType a(std::sqrt(dt) * (walker_type == CLOSED ? 2.0 : 1.0));
    
    v() = ComplexType(0.0);

    if (Lnak(0).extent(0) == 1)
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

  std::tuple<int,int> vHS_dims() const {
    int nstot = (walker_type == COLLINEAR) ? Likn.extent(0) : 1;
    int nptot = (walker_type == NONCOLLINEAR) ? Likn.extent(0) : 1;
    return std::make_tuple(nstot,nptot);
  }
  int number_of_ke_vectors() const { return nCV; }
  int number_of_cholesky_vectors() const { return nCV; }

  nda::array<ComplexType, 2> getHSPotentials()
  { return nda::array<ComplexType, 2>{}; }

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

