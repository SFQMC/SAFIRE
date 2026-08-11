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
#include "nda/tensor.hpp"
#include "utilities/check.hpp"
#include "utilities/mpi_context.h"
#include "utilities/check_strides.hpp"
#include "numerics/shared_array/const_shared_array.hpp"
#include "numerics/nda_functions.hpp"
#include "detail/one_body.hpp"

namespace sfqmc
{
namespace afqmc
{

template<MEMORY_SPACE _MEM, bool REAL>
class THCOps
{
  static constexpr MEMORY_SPACE MEM = _MEM;

  using ValueType     = typename std::conditional_t<REAL, RealType, ComplexType>;
  template<class T>
  using csrMat = math::sparse::csr_matrix<T, MEM, int, int>;

public:
  static constexpr HamiltonianTypes HamOpType = THC;
  constexpr HamiltonianTypes getHamType() const { return THC; }

  THCOps() = delete;

  /*
   * nup/ndown stands for number of alpha/beta electrons
   */
  THCOps(std::shared_ptr<utils::mpi_context_t<mpi3::communicator>> ctxt,
         WALKER_TYPES type,
         long nmo_,
         long nup_, 
         long ndn_, 
         memory::const_shared_array<HOST_MEMORY,ComplexType,3>&& hij_,
         memory::const_shared_array<MEM,ComplexType,3>&& haj_,
         memory::const_shared_array<MEM,ValueType,3>&& x_,
         memory::const_shared_array<MEM,ComplexType,5>&& y_,
         memory::const_shared_array<MEM,ValueType,2>&& l_,
         std::optional<memory::const_shared_array<MEM,ValueType,2>>&& z_,
         std::optional<memory::const_shared_array<MEM,ValueType,3>>&& x_rot_,
         std::optional<memory::const_shared_array<MEM,ComplexType,5>>&& y_rot_,
         std::optional<memory::const_shared_array<MEM,ValueType,2>>&& z_rot_,
         memory::const_shared_array<HOST_MEMORY,ComplexType,3>&& v0_,
         ComplexType e0_)
      : mpi(ctxt), 
        walker_type(type),
        NMO(nmo_),
        nup(nup_),
        ndown(ndn_),
        nelec{nup, ndown},
        hij(std::move(hij_)),
        haj(std::move(haj_)),
        _Xsiu_(std::move(x_)),
        _Ydsau_(std::move(y_)),
        _Luv_(std::move(l_)),
        _Zuv_(std::move(z_)),
        _Xsiu_rot_(std::move(x_rot_)),
        _Ydsau_rot_(std::move(y_rot_)),
        _Zuv_rot_(std::move(z_rot_)),
        vexx(std::move(v0_)),
        E0(e0_)
  {
    // make sure dimensions are compatible
    int nspin  = (walker_type == COLLINEAR) ? 2 : 1;
    int npol  = (walker_type == NONCOLLINEAR) ? 2 : 1;
    int nu = _Luv_.extent(0);
    int nv = _Luv_.extent(1);
    int nu_rot = (_Xsiu_rot_.has_value() ? _Xsiu_rot_->shape()[2] : nu);
    int nel  = (walker_type == COLLINEAR ? nup+ndown : nup); // NONCOLLINEAR has ndown=0 
    int nspin_in_H = hij.extent(0);
    int npol_in_H = hij.extent(1)/NMO;
    int ndet = haj.extent(0);
    utils::check(vexx.shape() == std::array<long,3>{nspin_in_H*npol_in_H,NMO,NMO},"THCOps: Size mismatch"); 
    utils::check(hij.shape() == std::array<long,3>{nspin_in_H,npol_in_H*NMO,npol_in_H*NMO},"THCOps: Size mismatch"); 
    utils::check(haj.shape() == std::array<long,3>{ndet,nel,npol*NMO},"THCOps: Size mismatch"); 
    utils::check(_Xsiu_.shape() == std::array<long,3>{nspin_in_H,npol_in_H*NMO,nu},"THCOps: Size mismatch"); 
    utils::check(_Ydsau_.shape() == std::array<long,5>{ndet,nspin,npol,nup,nu},"THCOps: Size mismatch"); 
    utils::check(_Luv_.shape() == std::array<long,2>{nu,nv},"THCOps: Size mismatch"); 
    if(_Zuv_.has_value())
      utils::check(_Zuv_->shape() == std::array<long,2>{nu,nu},"THCOps: Size mismatch"); 
    if(_Xsiu_rot_.has_value())
      utils::check(_Xsiu_rot_->shape() == std::array<long,3>{nspin_in_H,npol*NMO,nu_rot},"THCOps: Size mismatch"); 
    if(_Ydsau_rot_.has_value())
      utils::check(_Ydsau_rot_->shape() == std::array<long,5>{ndet,nspin,npol,nup,nu_rot},"THCOps: Size mismatch"); 
    if(_Zuv_rot_.has_value())
      utils::check(_Zuv_rot_->shape() == std::array<long,2>{nu_rot,nu_rot},"THCOps: Size mismatch"); 
  }

  nda::array<ComplexType,3> getOneBodyPropagatorMatrix(double dt,
                                                       nda::MemoryVector auto const& vMF)
  {
    using nda::range;
    auto all = range::all;
    int nspin  = (walker_type == COLLINEAR) ? 2 : 1;
    int npol  = (walker_type == NONCOLLINEAR) ? 2 : 1;
    int nspin_in_H = hij.extent(0);
    int npol_in_H = hij.extent(1)/NMO;
    utils::check(vMF.size() == number_of_cholesky_vectors(), "Size mismatch");

    memory::buffered_array<MEM,ComplexType,2> vMF_2d(1,vMF.size());
    vMF_2d(0,all) = vMF();
    
    auto meanfield_shift{nda::to_host(vHS(vMF_2d, dt))};

    nda::array<ComplexType, 3> H1(nspin, npol*NMO, npol*NMO);
    
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
    int nwalk = G.extent(0);
    int nspin  = (walker_type == COLLINEAR) ? 2 : 1;
    int npol  = (walker_type == NONCOLLINEAR) ? 2 : 1;
    int nel  = (walker_type == COLLINEAR ? nup+ndown : nup); // NONCOLLINEAR has ndown=0 
    utils::check(E.shape() == std::array<long,2>{nwalk,3}, "THC::energy: Size mismatch.");
    utils::check(G.extent(1) == nel*npol*NMO, "THC::energy: Size mismatch.");
    utils::check_strides(E,G);
    // limiting G to contiguous arrays for simplicity now, reconsider if necessary
    RealType scl = (walker_type == CLOSED ? 2.0 : 1.0);

    // addH1
    E() = ComplexType(0.0);
    if (addH1)
    {
      E(all,0) = E0; 
      auto haj_2d = nda::reshape(haj(),std::array<long,2>{haj.extent(0),haj.extent(1)*haj.extent(2)});
      nda::tensor::contract(ComplexType(scl), G, "wi", haj_2d(idet,all), "i", 
                            ComplexType(1.0), E(all,0), "w");
    }
    if (not(addEJ || addEXX))
      return;

    // get array_views to the correct data and correct determinant
    bool has_rot = _Xsiu_rot_.has_value();
    const auto Xsiu = ( has_rot ? (*_Xsiu_rot_)() : _Xsiu_() );
    const auto Ysau = ( has_rot ? (*_Ydsau_rot_)()(idet,nda::ellipsis{}) : _Ydsau_()(idet,nda::ellipsis{}) );
    const auto Zuv = ( has_rot ? (*_Zuv_rot_)() : (*_Zuv_)() );

    int nu = Zuv.extent(0);
    long nspin_in_H = _Xsiu_().shape()[0];
    long npol_in_H = _Xsiu_().shape()[1]/NMO; 

    // calculate how many walkers can be done concurrently
    long Bytes = default_buffer_size_in_MB * 1024L * 1024L;
    Bytes /= long((nu * nu + nu + nu * nup) * sizeof(ComplexType));
    int nwmax = ( MEM==HOST_MEMORY ? 1 : std::min(nwalk, std::max(1, int(Bytes))));

    utils::check(G.is_contiguous(), "Layout mismatch");
    memory::array_view<MEM,const ComplexType,3> G3d(std::array<long,3>{nwalk,nel,npol*NMO},G.data());

    int iw(0);
    while (iw < nwalk)
    {
      int nw = std::min(nwmax, nwalk - iw);
      // Guv[nspin][nu][nv]
      memory::buffered_array<MEM,ComplexType,3> Guv(nw,nu,nu);
//memory::buffered_array<MEM,ComplexType,3> Gvu(nw,nu,nu);
      // Guu[u]: summed over spin
      memory::buffered_array<MEM,ComplexType,2> Guu(nu,nw);
      Guu() = ComplexType(0.0);
      for (int ispin = 0; ispin < nspin; ++ispin)
      {
        // An empty spin sector contributes nothing to either EXX or Guu
        if(nelec[ispin] == 0) {
          continue;
        }
        for (int p1 = 0; p1 < npol; ++p1)
        {
          auto [is_,ip1_] = interaction_block(ispin,p1,npol,nspin_in_H,npol_in_H);
          for (int p2 = 0; p2 < npol; ++p2)
          {
            // Buffer space
            memory::buffered_array<MEM,ComplexType,3> Tva(nw,nu,nelec[ispin]);
            Guv_Guu(ispin, p1, p2, G3d(range(iw, iw + nw), range(ispin*nup,nup+ispin*ndown), all),
                    Guv, Guu, nda::flatten(Tva), idet);

/*
// need fast transposition, right now it is a bit slower and uses 2x more memory
            if constexpr (MEM==HOST_MEMORY) {
              for(int i=0; i<nw; ++i) {
                Gvu(i,all,all) = nda::transpose(Guv(i,all,all)); 
                Guv(i,nda::ellipsis{}) *= Zuv();
                E(iw+i,1) += ComplexType(-0.5*scl) * 
                    nda::dot(nda::flatten(Guv(i,nda::ellipsis{})),nda::flatten(Gvu(i,all,all))); 
              }
            } else {
              utils::check(false,"finish");
            }
*/
            
            if constexpr (MEM==HOST_MEMORY) {
              for(int i=0; i<nw; ++i)
                Guv(i,nda::ellipsis{}) *= Zuv();
            } else {
	      if constexpr(REAL) {
                auto Guv4d = memory::to_real_view(Guv);
                nda::tensor::elementwise(RealType(1.0),Zuv,"uv",
                                         RealType(1.0),Guv4d,"wuvc",nda::tensor::op::MUL);
              } else {
                nda::tensor::elementwise(ComplexType(1.0),Zuv,"uv",
                                         ComplexType(1.0),Guv,"wuv",nda::tensor::op::MUL);
              }
            }

            // reuse Guv memory
            memory::array_view<MEM,ComplexType,3> Twib(std::array<long,3>{nw,NMO,nelec[ispin]},Guv.data());

            // R[w,u][b] = sum_v Guv[w,u][v] * rotcXau[b][v]
            auto Yau = Ysau(ispin,p2,range(nelec[ispin]),all);
            auto Guv2 = nda::reshape(Guv(), nw*nu, nu);
            auto Tva2 = nda::reshape(Tva(), nw*nu, nelec[ispin]);
            nda::blas::gemm(Guv2, nda::transpose(Yau), Tva2);

            //T[w][b][k] = sum_u R[w][u][b] * Piu[k][u]
            auto Xiu = Xsiu(is_,range(ip1_*NMO,(ip1_+1)*NMO),all);
            if constexpr (MEM==HOST_MEMORY) { 
              for(int iw=0; iw<nw; iw++) 
                nda::blas::gemm(Xiu,Tva(iw,all,all),Twib(iw,all,all));
            } else { 
	      if constexpr(REAL) {
                auto Ta4d = memory::to_real_view(Tva);
                auto Tb4d = memory::to_real_view(Twib);
                nda::tensor::contract(Ta4d,"wuac",Xiu,"iu",Tb4d,"wiac"); 
	      } else {
                nda::tensor::contract(Xiu,"iu",Tva,"wua",Twib,"wia"); 
              }
            }

            // E[w] = sum_ai T[w][a][i] * G[w][a][i] 
            auto Gwai = G3d(range(iw, iw + nw),range(ispin*nup,nup+ispin*ndown),range(p1*NMO,(p1+1)*NMO)); 
            nda::tensor::contract(ComplexType(-0.5*scl),Twib,"wia",Gwai,"wai",
                                  ComplexType(1.0),E(range(iw, iw + nw),1),"w"); 
          }
        }
      }
      if (addEJ)
      {
        memory::buffered_array<MEM,ComplexType,2> Tuw(nu,nw);
        nda::blas::gemm(nda::transpose(Zuv),Guu,Tuw);
        nda::tensor::contract(ComplexType(RealType(0.5 * scl * scl)),Guu,"uw",Tuw,"uw",
                              ComplexType(1.0),E(range(iw, iw + nw), 2),"w"); 
      }
      iw += nw;
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
    auto all  = range::all;
    int nwalk = G.extent(0);
    int npol  = (walker_type == NONCOLLINEAR) ? 2 : 1;
    int nel   = G.extent(1)/(npol*NMO); 
    int ispin = (spin_component == Alpha ? 0 : 1);
    utils::check(E.shape() == std::array<long,2>{nwalk,3}, "THC::energy: Size mismatch.");
    utils::check(G.extent(1) == nel*npol*NMO, "THC::energy: Size mismatch.");
    utils::check(nel == nelec[ispin], "G.extent(1) != nelec[ispin].");
    if(addEJ)
      utils::check(EJn.shape() == std::array<long,2>{nwalk,number_of_ke_vectors()}, "Size mismatch.");
    utils::check_strides(E,G);
    // limiting G to contiguous arrays for simplicity now, reconsider if necessary

    utils::check(G.is_contiguous(), "Layout mismatch");
    memory::array_view<MEM,const ComplexType,3> G3d(std::array<long,3>{nwalk,nel,npol*NMO},G.data());

    // addH1
    E() = ComplexType(0.0);
    if (addH1)
    {
      if(spin_component==Alpha) E(all,0) = E0;
      if(nel > 0) {
        nda::tensor::contract(ComplexType(1.0), G3d, "wai",
                              haj()(idet,range(ispin*nup,nup+ispin*ndown),all), "ai",
                              ComplexType(1.0), E(all,0), "w");
      }
    }
    if (not(addEJ || addEXX))
      return;
    // An empty spin sector (e.g. collinear with ndown==0) contributes nothing
    // beyond E0; everything below would be zero-sized.
    if (nel == 0) {
      if (addEJ) {
        EJn() = 0.0;
      }
      return;
    }

    // get array_views to the correct data and correct determinant
    bool has_rot = _Xsiu_rot_.has_value();
    const auto Xsiu = ( has_rot ? (*_Xsiu_rot_)() : _Xsiu_() );
    const auto Ysau = ( has_rot ? (*_Ydsau_rot_)()(idet,nda::ellipsis{}) : _Ydsau_()(idet,nda::ellipsis{}) );
    const auto Zuv = ( has_rot ? (*_Zuv_rot_)() : (*_Zuv_)() );

    int nu = Zuv.extent(0);
    long nspin_in_H = _Xsiu_().shape()[0];
    long npol_in_H = _Xsiu_().shape()[1]/NMO; 

    // calculate how many walkers can be done concurrently
    long Bytes = default_buffer_size_in_MB * 1024L * 1024L;
    Bytes /= long((nu * nu + nu + nu * nup) * sizeof(ComplexType));
    int nwmax = ( MEM==HOST_MEMORY ? 1 : std::min(nwalk, std::max(1, int(Bytes))));
    
    int iw(0);
    while (iw < nwalk)
    {
      int nw = std::min(nwmax, nwalk - iw);
      // Guv[nspin][nu][nv]
      memory::buffered_array<MEM,ComplexType,3> Guv(nw,nu,nu);
      // Guu[u]: summed over spin
      memory::buffered_array<MEM,ComplexType,2> Guu(nu,nw);
      Guu() = ComplexType(0.0);

      for (int p1 = 0; p1 < npol; ++p1)
      {
        auto [is_,ip1_] = interaction_block(ispin,p1,npol,nspin_in_H,npol_in_H);
        for (int p2 = 0; p2 < npol; ++p2)
        {
          // Buffer space
          memory::buffered_array<MEM,ComplexType,3> Tva(nw,nu,nelec[ispin]);
          Guv_Guu(ispin, p1, p2, G3d(range(iw, iw+nw),all,all), Guv, Guu, nda::flatten(Tva), idet);

          if constexpr (MEM==HOST_MEMORY) {
            for(int i=0; i<nw; ++i)
              Guv(i,nda::ellipsis{}) *= Zuv();
          } else {
            if constexpr(REAL) {
              auto Guv4d = memory::to_real_view(Guv);
              nda::tensor::elementwise(RealType(1.0),Zuv,"uv",
                                       RealType(1.0),Guv4d,"wuvc",nda::tensor::op::MUL);
            } else {
              nda::tensor::elementwise(ComplexType(1.0),Zuv,"uv",
                                       ComplexType(1.0),Guv,"wuv",nda::tensor::op::MUL);
            }
          }

          // reuse Guv memory
          memory::array_view<MEM,ComplexType,3> Twib(std::array<long,3>{nw,NMO,nelec[ispin]},Guv.data());

          // R[w,u][b] = sum_v Guv[w,u][v] * rotcXau[b][v]
          auto Yau = Ysau(ispin,p2,range(nelec[ispin]),all);
          if constexpr (MEM==HOST_MEMORY)
            for(int iw=0; iw<nw; iw++)
              nda::blas::gemm(Guv(iw,all,all),nda::transpose(Yau),Tva(iw,all,all));
          else
            nda::tensor::contract(Yau,"av",Guv,"wuv",Tva,"wua");

          //T[w][b][k] = sum_u R[w][u][b] * Piu[k][u]
          auto Xiu = Xsiu(is_,range(ip1_*NMO,(ip1_+1)*NMO),all);
          if constexpr (MEM==HOST_MEMORY) {
            for(int iw=0; iw<nw; iw++)
              nda::blas::gemm(Xiu,Tva(iw,all,all),Twib(iw,all,all));
          } else {
            if constexpr(REAL) {
              auto Ta4d = memory::to_real_view(Tva);
              auto Tb4d = memory::to_real_view(Twib);
              nda::tensor::contract(Ta4d,"wuac",Xiu,"iu",Tb4d,"wiac");
            } else {
              nda::tensor::contract(Xiu,"iu",Tva,"wua",Twib,"wia");
            }
          }

          // E[w] = sum_ai T[w][a][i] * G[w][a][i] 
          auto Gwai = G3d(range(iw, iw + nw),range(ispin*nup,nup+ispin*ndown),range(p1*NMO,(p1+1)*NMO)); 
          nda::tensor::contract(ComplexType(-0.5),Twib,"wia",Gwai,"wai",
                                ComplexType(1.0),E(range(iw, iw + nw), 1),"w"); 
        }
      }

      if (addEJ)
      {
        EJn() = ComplexType(0.0);
        memory::buffered_array<MEM,ComplexType,2> Tuw(nu,nw);
        nda::blas::gemm(nda::transpose(Zuv),Guu,Tuw);
        if constexpr (MEM==HOST_MEMORY)
          EJn() = nda::transpose(Tuw());
        else
          nda::tensor::assign(Tuw,"uw",EJn,"wu");
        nda::tensor::contract(ComplexType(RealType(0.5)),Guu,"uw",EJn,"wu",
                              ComplexType(1.0),E(range(iw, iw + nw), 2),"w"); 
      }
      iw += nw;
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

  // returns v[nwalk, nspin_in_basis*npol_in_basis, NMO, NMO]
  // no spin-orbit vHS yet
  auto vHS(nda::MemoryMatrix auto&& X, double dt)
  {
    constexpr MEMORY_SPACE MEM_X = memory::get_memory_space<decltype(X)>();
    static_assert(MEM == MEM_X, "Memory space mismatch");
    using nda::range;
    auto all = range::all;
    int nchol = ( REAL ? _Luv_().extent(1) : 2 * _Luv_().extent(1) );
    int nwalk = X.extent(0);
    long nspin_in_H = _Xsiu_().shape()[0];
    long npol_in_H = _Xsiu_().shape()[1]/NMO;
    utils::check_strides(X);
    // limiting X/v to contiguous arrays for simplicity now, reconsider if necessary
    utils::check(X.shape() == std::array<long,2>{nwalk,nchol}, "THC::vHS: Size mismatch.");

    // Note: Allocate first, to make better use of memory pool
    // vHS[nspin_in_vHS][nwalk][npol_in_vHS*NMO][NMO]
    memory::buffered_array<MEM_X,ComplexType,4> v(nspin_in_H,nwalk,npol_in_H*NMO,NMO);
    auto v5d = nda::reshape(v, std::array<long,5>{nspin_in_H,nwalk,npol_in_H,NMO,NMO});
    v() = ComplexType(0.0);

    // scale by sqrt(dt)
    RealType a(std::sqrt(dt));

    // get array_views to the correct data and correct determinant
    bool has_rot = _Xsiu_rot_.has_value();
    const auto Xsiu = ( has_rot ? (*_Xsiu_rot_)() : _Xsiu_() );
    const auto Luv = _Luv_(); 
    int nu = Luv.extent(0);

    // calculate how many walkers can be done concurrently
    long Bytes = default_buffer_size_in_MB * 1024L * 1024L;
    Bytes /= size_t(NMO * (nu+NMO) * sizeof(ComplexType));
    int nwmax = std::min(nwalk, std::max(1, int(Bytes)));

    // work array
    memory::buffered_array<MEM,ComplexType,2> Tuw(nu,nwalk);
    auto Tuw_r = memory::to_real_view(Tuw);

    // T[u][w] = sum_v L[u][v] * X[v][w] 
    {
      memory::array_view<MEM,const RealType,2> Luv2(std::array<long,2>{nu,nchol},
                                                    reinterpret_cast<RealType const*>(Luv.data()));
      memory::buffered_array<MEM,ComplexType,2> Xt(nchol,nwalk);
      if constexpr (MEM==HOST_MEMORY)
        Xt() = nda::transpose(X);
      else
        nda::tensor::assign(X,"wn",Xt,"nw");
      nda::blas::gemm(Luv2,Xt,Tuw);
    }

    // v[w][is*npol+ip][i][j] = sum_u conj(X[is][ip*NMO+i][u]) * X[is][ip*NMO+j][u] * T[u][w] 
    int iw(0);
    while (iw < nwalk)
    {
      int nw = std::min(nwmax, nwalk - iw);
      memory::buffered_array<MEM,ComplexType,3> Qwiu(nw,NMO,nu);
      memory::buffered_array<MEM_X,ComplexType,2> vt_2d(nw*NMO,NMO);
      auto vt_3d = nda::reshape(vt_2d, std::array<long,3>{nw,NMO,NMO});
      for( int is=0; is<nspin_in_H; ++is) {
        for( int ip=0; ip<npol_in_H; ++ip) {
       
          auto Xiu = Xsiu(is,range(ip*NMO,(ip+1)*NMO),all); 
          if constexpr (REAL) {

            memory::array_view<MEM,ComplexType,3> Quwi(std::array<long,3>{nu,nw,NMO},Qwiu.data()); 
            // Qwiu[w][i][u] = T[w][u] * conj(Piu[i][u])
            if constexpr (MEM==HOST_MEMORY) {
              for(int w=0; w<nw; ++w)
                for(int i=0; i<NMO; ++i)
                  Quwi(all,w,i) = Tuw(all,iw+w) * Xiu(i,all);
            } else {
              auto Quwi_r = memory::to_real_view(Quwi);
              nda::tensor::elementwise_trinary(1.0,Tuw_r(all,range(iw,iw+nw),all),"uwc",1.0,Xiu,"iu",0.0,Quwi_r,"uwic",nda::tensor::op::MUL,nda::tensor::op::SUM);
            }
            
            auto Q2d = nda::reshape(Quwi, std::array<long,2>{nu,nw*NMO});
            memory::array_view<MEM,ComplexType,2> vij(std::array<long,2>{NMO,nw*NMO},vt_2d.data()); 
            auto vij_3d = nda::reshape(vij, std::array<long,3>{NMO,nw,NMO});
            nda::blas::gemm(a,Xiu,Q2d,0.0,vij);
            if constexpr (MEM==HOST_MEMORY)
              for(int w=0; w<nw; w++)
                v(is,iw+w,range(ip*NMO,(ip+1)*NMO),all) = nda::transpose(vij_3d(all,w,all));
            else
              nda::tensor::add(ComplexType(1.0),vij_3d,"jwi",
                               ComplexType(0.0),v5d(is,range(iw,iw+nw),ip,all,all),"wij");

          } else {
 
            // Qwiu[w][i][u] = T[w][u] * conj(Piu[i][u])
            if constexpr (MEM==HOST_MEMORY) {
              for(int w=0; w<nw; ++w)
                for(int i=0; i<NMO; ++i)
                  Qwiu(w,i,all) = Tuw(all,iw+w) * nda::conj(Xiu(i,all));
            } else {
              nda::tensor::elementwise_trinary(ComplexType(1.0),Tuw(all,range(iw,iw+nw)),"uw",ComplexType(1.0),nda::conj(Xiu),"iu",ComplexType(0.0),Qwiu,"wiu",nda::tensor::op::MUL,nda::tensor::op::SUM); 
            }

            auto Q2d = nda::reshape(Qwiu, std::array<long,2>{nw*NMO,nu});
            nda::blas::gemm(ComplexType(a),Q2d,nda::transpose(Xiu),ComplexType(0.0),vt_2d);
            v(is,range(iw,iw+nw),range(ip*NMO,(ip+1)*NMO),all) = vt_3d();

          }
        }
      }
      iw += nw;
    }
    return v;
  }

  void vbias(nda::MemoryArrayOfRank<2> auto const& G, nda::MemoryArrayOfRank<2> auto& v, double dt)
  {
    memory::check_memory_space<MEM>(G,v);
    using nda::range;
    int nchol = ( REAL ? _Luv_().extent(1) : 2 * _Luv_().extent(1) );
    int nwalk = G.extent(0);
    int nspin  = (walker_type == COLLINEAR) ? 2 : 1;
    int npol  = (walker_type == NONCOLLINEAR) ? 2 : 1;
    int nel  = (walker_type == COLLINEAR ? nup+ndown : nup); // NONCOLLINEAR has ndown=0 
    utils::check_strides(G,v);
    // limiting G to contiguous arrays for simplicity now, reconsider if necessary
    utils::check(v.shape() == std::array<long,2>{nwalk,nchol}, "THC::vbias: Size mismatch.");
    if(haj.extent(0) == 1) // ndet==1, G half rotated
      utils::check(G.extent(1) == nel*npol*NMO, "THC::vbias: Size mismatch.");
    else // ndet>1, full G 
      utils::check(G.extent(1) == nspin*npol*NMO*npol*NMO, "THC::vbias: Size mismatch.");
    utils::check(G.is_contiguous(), "Layout mismatch");

    // scale a by sqrt(dt)
    RealType a(std::sqrt(dt));

    // get array_views to the correct data and correct determinant
    const auto Luv = _Luv_(); 
    int nu = Luv.extent(0);

    if (haj.extent(0) == 1)
    {
      memory::array_view<MEM,const ComplexType,3> G3d(std::array<long,3>{nwalk,nel,npol*NMO},G.data());
      memory::buffered_array<MEM,ComplexType,2> Guu(nu,nwalk);
      memory::buffered_array<MEM,ComplexType,2> vt(nchol,nwalk);
      Guu_from_compact(G3d, Guu, 0);
      memory::array_view<MEM,const RealType,2> Luv2(std::array<long,2>{nu,nchol},reinterpret_cast<RealType const*>(Luv.data()));
      nda::blas::gemm(a,nda::transpose(Luv2),Guu,0.0,vt);
      if constexpr (MEM==HOST_MEMORY)
        v() = nda::transpose(vt());
      else
        nda::tensor::assign(vt,"nw",v,"wn");
    }
    else
    {
      utils::check(false," Error: THC not yet implemented for multiple references.");
      // multideterminant is not half-rotated, so use Likn
      // which spin???
      memory::array_view<MEM,const ComplexType,4> G3d(std::array<long,4>{nwalk,nspin,npol*NMO,npol*NMO},G.data());
    }
  }

  /// Returns the number of spins and polarizations in the VHS potential.
  std::tuple<int,int> vHS_dims() const {
    return std::make_tuple(_Xsiu_().shape()[0],_Xsiu_().shape()[1]/NMO);
  }
  int number_of_ke_vectors() const { 
    utils::check(_Zuv_.has_value() or _Zuv_rot_.has_value(), "Missing Zuv/Zuv_rot.");
    if(_Zuv_.has_value()) return _Zuv_->extent(0);
    else return _Zuv_rot_->extent(0); 
  }
  int number_of_cholesky_vectors() const { return ( REAL ? _Luv_().extent(1) : 2 * _Luv_().extent(1) ); }

  // add nspin_in_basis to allow for a spin independent basis too
  nda::array<ComplexType, 2> getHSPotentials() 
  { return nda::array<ComplexType, 2>{}; }

protected:
  // Guu[nu][nwalk]
  void Guu_from_compact(nda::MemoryArrayOfRank<3> auto const& G,
               nda::MemoryArrayOfRank<2> auto && Guu,
               int idet)
  {
    using nda::range;
    auto all = range::all;
    long nspin_in_H = _Xsiu_().shape()[0];
    long npol_in_H = _Xsiu_().shape()[1]/NMO;
    int nspin  = (walker_type == COLLINEAR) ? 2 : 1;
    int npol  = (walker_type == NONCOLLINEAR) ? 2 : 1;
    int nel  = (walker_type == COLLINEAR ? nup+ndown : nup); // NONCOLLINEAR has ndown=0 
    bool has_rot = _Xsiu_rot_.has_value();
    const auto Xsiu = ( has_rot ? (*_Xsiu_rot_)() : _Xsiu_());
    const auto Ysau = ( has_rot ? (*_Ydsau_rot_)()(idet,nda::ellipsis{}) :
                                 _Ydsau_()(idet,nda::ellipsis{}) );
    long nw   = G.extent(0);
    long nu   = Xsiu.extent(2);

    // G3d[w][a][j]
    utils::check(G.shape() == std::array<long,3>{nw,nel,npol*NMO}, "THC::Guu_from_compact: Shape mismatch");
    utils::check(Guu.shape() == std::array<long,2>{nu,nw}, "THC::Guu_from_compact: Shape mismatch");
    Guu() = ComplexType(0.0);
    ComplexType a = (walker_type == CLOSED) ? ComplexType(2.0) : ComplexType(1.0);
    for( int is=0; is<nspin; is++ ) {
      for( int ip=0; ip<npol; ip++ ) {
        
        auto [is_,ip_] = interaction_block(is,ip,npol,nspin_in_H,npol_in_H);
        auto Xiu = Xsiu(is_,range(ip_*NMO,(ip_+1)*NMO),all);
        auto Yau = Ysau(is,ip,range(nelec[is]),all);

        if constexpr (MEM==HOST_MEMORY) {
          memory::buffered_array<MEM,ComplexType,2> Tau(nelec[is],nu);    
          for(int iw=0; iw<nw; iw++) {
            if constexpr (REAL) {
              auto G4d = memory::to_real_view(G);
              auto T3d = memory::to_real_view(Tau);
              auto Gaic = G4d(iw,range(is*nup,nup+is*ndown),range(ip*NMO,(ip+1)*NMO),all);
              // MAM: Not ideal, contract is not optimal in cpu
              nda::tensor::contract(Gaic,"aic",Xiu,"iu",T3d,"auc");
            } else {
              auto Gai = G(iw,range(is*nup,nup+is*ndown),range(ip*NMO,(ip+1)*NMO));
              nda::blas::gemm(Gai,Xiu,Tau);
            }
            // Gwu[w][u] = a * sum_a T1[w][a][u] * cXau[a][u]
            for(int ia=0; ia<nelec[is]; ++ia) 
              Guu(all,iw) += ComplexType(a)*Tau(ia,all)*Yau(ia,all);
          }
        } else {
          if(nelec[is] > 0) {
            memory::buffered_array<MEM,ComplexType,3> Twau(nw,nelec[is],nu);    
            if constexpr (REAL) {
              auto G4d = memory::to_real_view(G);
              auto Gwaic = G4d(all,range(is*nup,nup+is*ndown),range(ip*NMO,(ip+1)*NMO),all); 
              auto T4d = memory::to_real_view(Twau);
              nda::tensor::contract(Gwaic,"waic",Xiu,"iu",T4d,"wauc");
            } else {
              auto Gwai = G(all,range(is*nup,nup+is*ndown),range(ip*NMO,(ip+1)*NMO)); 
              nda::tensor::contract(Gwai,"wai",Xiu,"iu",Twau,"wau");
            }
            // Gwu[w][u] = a * sum_a T1[w][a][u] * cXau[a][u]
            nda::tensor::contract(ComplexType(a),Twau,"wau",Yau,"au",ComplexType(1.0),Guu,"uw");
          }
        }
 
      } // npol 
    } // nspin 
  }

/*
  // Guu[nu][nwalk]
  template<class MatA, class MatB>
  void Guu_from_full(MatA const& G, MatB&& Guu)
  {
    int nmo_ = int(Piu.size(0));
    int nu   = int(Piu.size(1));
    int u0, uN;
    std::tie(u0, uN) = FairDivideBoundary(comm->rank(), nu, comm->size());
    int nwalk        = G.size(0);

    RUNTIME_CHECK(G.size(0) == Guu.size(1), "");
    RUNTIME_CHECK(Guu.size(0) == nu, "");
    RUNTIME_CHECK(G.size(1) == nmo_ * nmo_, "");

    // calculate how many walkers can be done concurrently
    long Bytes = default_buffer_size_in_MB * 1024L * 1024L;
    Bytes /= size_t(nmo_ * nu * sizeof(SPComplexType));
    int nwmax = std::min(nwalk, std::max(1, int(Bytes)));

    ComplexType a = (walker_type == CLOSED) ? ComplexType(2.0) : ComplexType(1.0);
    Array<SPComplexType, 2> T1({nwmax * nmo_, (uN - u0)},
                               device_buffer_manager.get_generator().template get_allocator<SPComplexType>());
    comm->barrier();
    ma::fill(Guu.sliced(u0,uN), SPComplexType(0.0));

    APP_ABORT(" Error: Finish Guu_from_full \n\n");
    int iw(0);
    while (iw < nwalk)
    {
      int nw = std::min(nwmax, nwalk - iw);
      Array_cref<SPComplexType, 2> Giw(make_device_ptr(G[iw].origin()), {nw * nmo_, nmo_});
      //        ma::product(Giw,Piu({0,nmo_},{u0,uN}),T1);
      // Guu[u+u0][w] = alpha * sum_i T[w][i][u] * P[i][u]
      // this looks wrong!!!
      //Awiu_Biu_Cuw(uN - u0, nw, nmo_, SPComplexType(a), T1.origin(), make_device_ptr(Piu.origin()) + u0, nu,
      //             make_device_ptr(Guu[u0].origin()) + iw, nwalk);
      //ma::Awiu_Biu_Cuw(SPComplexType(a), T1.partitioned(nwmax).sliced(0, nw),
      //		       Piu.range({u0,uN},1), Guu.sliced(u0,uN).range({iw,iw+nw},1));
      iw += nw;
    }
    comm->barrier();
  }
*/

  // Computes Guv and Guu for a set of walkers
  // rotMuv is partitioned along 'u'
  // G[w][nel*nmo]
  // Guv[w][nu][nv]
  // Guu[w][v], accumulated on this routine, sum over spin is outside
  // Tbuff: work space
  void Guv_Guu(int ispin, int p1, int p2, nda::MemoryArrayOfRank<3> auto const& G, 
         nda::MemoryArrayOfRank<3> auto && Guv, nda::MemoryArrayOfRank<2> auto && Guu, 
         nda::MemoryVector auto && Tbuff, int idet)
  {
    using nda::range;
    auto all = range::all;
    long nspin_in_H = _Xsiu_().shape()[0];
    long npol_in_H = _Xsiu_().shape()[1]/NMO;
    int npol  = (walker_type == NONCOLLINEAR) ? 2 : 1;
    auto [is_,ip_] = interaction_block(ispin,p2,npol,nspin_in_H,npol_in_H);
    range M_rng(ip_*NMO,(ip_+1)*NMO);
    int nel  = G.extent(1);
    bool has_rot = _Xsiu_rot_.has_value();
    const auto Xiu = ( has_rot ? (*_Xsiu_rot_)()(is_,M_rng,all) :
                                  _Xsiu_()(is_,M_rng,all) );
    const auto Yau = ( has_rot ? (*_Ydsau_rot_)()(idet,ispin,p1,range(nelec[ispin]),all) : 
                                 _Ydsau_()(idet,ispin,p1,range(nelec[ispin]),all) );
    int nw = int(G.extent(0));
    int nu = Xiu.extent(1);

    utils::check(nel == nelec[ispin], "THC::Guv_Guu: G.extent(1) != nelec[ispin]"); 
    utils::check(Tbuff.size() >= nw*nel*nu, "THC::Guv_Guu: Size mismatch");
    // G3d[w][a][j]
    utils::check(G.shape() == std::array<long,3>{nw,nel,npol*NMO}, "THC::Guv_Guu: Shape mismatch");

    static_assert(std::decay_t<decltype(G)>::is_stride_order_C(), "Stride mismatch");
    if constexpr (MEM==HOST_MEMORY) {
      // Twav[w][a][v] = sum_j G[w][a][j] X[j][v]
      // G[w][u][v] = sum_a X[a][u] Twav[w][a][v]
      if constexpr (REAL) {
        memory::buffered_array<MEM,ComplexType,2> G_(NMO,nel); 
        memory::array_view<MEM,ComplexType,2> Tua(std::array<long,2>{nu,nel},Tbuff.data());
        for(int iw=0; iw<nw; ++iw) { 
          G_() = nda::transpose(G(iw,all,range(p2*NMO,(p2+1)*NMO)));
          nda::blas::gemm(nda::transpose(Xiu),G_,Tua);
          nda::blas::gemm(nda::transpose(Yau),nda::transpose(Tua),Guv(iw,all,all));
        }
      } else {
        memory::array_view<MEM,ComplexType,2> Tau(std::array<long,2>{nel,nu},Tbuff.data());
        for(int iw=0; iw<nw; ++iw) { 
          nda::blas::gemm(G(iw,all,range(p2*NMO,(p2+1)*NMO)),Xiu,Tau);
          nda::blas::gemm(nda::transpose(Yau),Tau,Guv(iw,all,all));
        }
      }
    } else {
      memory::array_view<MEM,ComplexType,3> Twav(std::array<long,3>{nw,nel,nu},Tbuff.data());
      if(nel > 0) {
        if constexpr (REAL) {
          auto G4d = memory::to_real_view(G);
          auto T4d = memory::to_real_view(Twav);
          // choose electron range compatible with ispin
          auto Gwaic = G4d(all,all,range(p2*NMO,(p2+1)*NMO),all);
          // Twav[w][a][v] = sum_j G[w][a][j] X[j][v]
          nda::tensor::contract(Gwaic,"wajc",Xiu,"jv",T4d,"wavc");
        } else {
          auto Gwai = G(all,all,range(p2*NMO,(p2+1)*NMO));
          // Twav[w][a][v] = sum_j G[w][a][j] X[j][v]
          nda::tensor::contract(Gwai,"wai",Xiu,"iv",Twav,"wav");
        }
        // G[w][u][v] = sum_a X[a][u] Twav[w][a][v]
        nda::tensor::contract(Yau,"au",Twav,"wav",Guv,"wuv");
      } else {
        nda::tensor::set(0, Guv);
      }
    }

    // Gwv = Gwvv, 
    if(p1==p2) {
      if constexpr (MEM==HOST_MEMORY) {
        for(int i=0; i<nw; i++)
          Guu(all,i) += nda::diagonal(Guv(i,all,all));
      } else {
        std::array<long,2> str = {Guv.strides()[0],Guv.strides()[1]+1};
        std::array<long,2> shape = {Guv.extent(0),Guv.extent(1)};
        nda::idx_map<2, 0, nda::C_stride_order<2>, nda::layout_prop_e::none> idxm(shape,str);
        memory::array_view<MEM,ComplexType,2> Guv_diag(idxm, Guv.data());
        nda::tensor::add(ComplexType(1.0),Guv_diag,"wu",ComplexType(1.0),Guu,"uw");   
      }
    }
  }
/*
  / *
    // since this is for energy, only compact is accepted
    // Computes Guv and Guu for a single walker
    // As opposed to the other Guu routines,
    //  this routine expects G for the walker in matrix form
    // rotMuv is partitioned along 'u'
    // G[nel][nmo]
    // Guv[nu][nu]
    // Guu[u]: summed over spin
    // T1[nel_][nu]
    template<class MatA, class MatB, class MatC, class MatD>
    void Guv_Guu2(MatA const& G, MatB&& Guv, MatC&& Guu, MatD&& T1, int k) {

      static_assert(std::decay<MatA>::type::dimensionality == 2, "Wrong dimensionality");
      static_assert(std::decay<MatB>::type::dimensionality == 2, "Wrong dimensionality");
      static_assert(std::decay<MatC>::type::dimensionality == 1, "Wrong dimensionality");
      static_assert(std::decay<MatD>::type::dimensionality == 2, "Wrong dimensionality");
      int nmo_ = int(rotPiu.size(0));
      int nu = int(rotMuv.size(0));  // potentially distributed over nodes
      int nv = int(rotMuv.size(1));  // not distributed over nodes
      RUNTIME_CHECK(rotPiu.size(1) == nv, "");
      int v0,vN;
      std::tie(v0,vN) = FairDivideBoundary(comm->rank(),nv,comm->size());
      int nu0 = rotnmu0; 
      ComplexType zero(0.0,0.0);

      RUNTIME_CHECK(Guu.size(0) == nv, "");
      RUNTIME_CHECK(Guv.size(0) == nu, "");
      RUNTIME_CHECK(Guv.size(1) == nv, "");

      // sync first
      comm->barrier();
      int nel_ = (walker_type==CLOSED)?nup:(nup+ndown);
      RUNTIME_CHECK(G.size(0) == size_t(nel_), "");
      RUNTIME_CHECK(G.size(1) == size_t(nmo_), "");
      RUNTIME_CHECK(T1.size(0) == size_t(nel_), "");
      RUNTIME_CHECK(T1.size(1) == size_t(nv), "");

      ma::product(G,rotPiu({0,nmo_},{v0,vN}),
                  T1(T1.extension(0),{v0,vN}));
      // This operation might benefit from a 2-D work distribution
      ma::product(rotcXau[k].sliced(nu0,nu0+nu),
                  T1(T1.extension(0),{v0,vN}),
                  Guv(Guv.extension(0),{v0,vN}));
      for(int v=v0; v<vN; ++v)
        if( v < nu0 || v >= nu0+nu ) {
          Guu[v] = ma::dot(rotcXau[k][v],T1(T1.extension(0),v)); 
        } else
         Guu[v] = Guv[v-nu0][v];
      comm->barrier();
    }
*/
protected:
  std::shared_ptr<utils::mpi_context_t<mpi3::communicator>> mpi;

  WALKER_TYPES walker_type;

  int NMO, nup, ndown;
  int nelec[2];

  // H1[nspin][npol*NMO][npol*NMO]
  memory::const_shared_array<HOST_MEMORY,ComplexType,3> hij;

  // half rotated one body hamiltonian: [ndet][nup+ndn][npol*NMO]
  memory::const_shared_array<MEM,ComplexType,3> haj;

  // Xsiu[nspin][npol*NMO][nu]
  memory::const_shared_array<MEM,ValueType,3> _Xsiu_;

  // Ydsau[ndet][nspin][ipol][nup][nu]
  memory::const_shared_array<MEM,ComplexType,5> _Ydsau_;

  // Luv[nu][nv]
  memory::const_shared_array<MEM,ValueType,2> _Luv_;

  // Zuv[nu][nv]
  std::optional<decltype(_Luv_)> _Zuv_;

  // Xsiu_rot[nspin][npol*NMO][nu_rot]
  std::optional<decltype(_Xsiu_)> _Xsiu_rot_;

  // Ydsau_rot[ndet][nspin][ipol][nup][nu_rot]
  std::optional<decltype(_Ydsau_)> _Ydsau_rot_; 

  // Zuv_rot[nu_rot][nu_rot]
  std::optional<decltype(_Luv_)> _Zuv_rot_; 

  // vexx(i,l) = -0.5 * sum_j <ij|jl>
  memory::const_shared_array<HOST_MEMORY,ComplexType,3> vexx;

  ComplexType E0;

  long default_buffer_size_in_MB = 4L * 1024L;
};

} // namespace afqmc

} // namespace sfqmc

