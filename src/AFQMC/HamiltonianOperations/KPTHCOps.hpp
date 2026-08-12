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
#include "utilities/mpi_context.h"
#include "utilities/check_strides.hpp"
#include "numerics/shared_array/const_shared_array.hpp"
#include "numerics/nda_functions.hpp"
#include "detail/one_body.hpp"
#include "AFQMC/Utilities/wfn_utils.hpp"

namespace sfqmc
{
namespace afqmc
{

/*
 * v+(q,n) = 0.5 * ( L(q,n) + dagger(L(q,n)) )
 * v-(q,n) = i*0.5 * ( L(q,n) - dagger(L(q,n)) )
 * L(q,n) = sum_k sum_ab L^{k,q}_{ab,n} dagger( c^{k}_a ) c^{k-q}_b
 */


template<MEMORY_SPACE _MEM>
class KPTHCOps
{
  static constexpr MEMORY_SPACE MEM = _MEM;
  template<class T>
  using csrMat = math::sparse::csr_matrix<T, MEM, int, int>;

public:
  static constexpr HamiltonianTypes HamOpType = KPTHC;
  constexpr HamiltonianTypes getHamType() const { return KPTHC; }

  /*
   * nup/ndown stands for number of alpha/beta electrons
   * Completely broken for half_rotated integrals, FIX FIX FIX
   */
  KPTHCOps(std::shared_ptr<utils::mpi_context_t<mpi3::communicator>> ctxt,
         WALKER_TYPES type,
         int nmo_,
         int nup_, 
         int ndn_, 
         int nkpts_,
         int q0_,
         nda::array<nda::array<int,1>,2>&& nocc_per_kp_,
         nda::array<int,1>&& minusq_,
         nda::array<int,2>&& qk_to_k2_,
         memory::const_shared_array<HOST_MEMORY,ComplexType,4>&& hij_,
         memory::const_shared_array<MEM,ComplexType,3>&& haj_,
         memory::const_shared_array<MEM,ComplexType,4>&& x_,
         memory::const_shared_array<MEM,ComplexType,6>&& y_,
         memory::const_shared_array<MEM,ComplexType,3>&& l_,
         std::optional<memory::const_shared_array<MEM,ComplexType,3>>&& z_,
         std::optional<memory::const_shared_array<MEM,ComplexType,4>>&& x_rot_,
         std::optional<memory::const_shared_array<MEM,ComplexType,6>>&& y_rot_,
         std::optional<memory::const_shared_array<MEM,ComplexType,3>>&& z_rot_,
         memory::const_shared_array<HOST_MEMORY,ComplexType,4>&& v0_,
         ComplexType e0_)
      : mpi(ctxt), 
        walker_type(type),
        NMO(nmo_),
        nup(nup_),
        ndown(ndn_),
        nelec{nup, ndown},
        nkpts(nkpts_),
        nbnd(NMO/nkpts),
        nocc_per_kp(nocc_per_kp_),
        minusq(std::move(minusq_)),
        qk_to_k2(std::move(qk_to_k2_)),
        Q0_index(q0_),
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
    int nu = _Luv_.extent(1);
    int nv = _Luv_.extent(2);
    int nu_rot = (_Xsiu_rot_.has_value() ? _Xsiu_rot_->shape()[3] : nu);
    int nel  = (walker_type == COLLINEAR ? nup+ndown : nup); // NONCOLLINEAR has ndown=0 
    int nstot = hij.extent(0);
    int nptot = hij.extent(2)/nbnd;
    int ndet = haj.extent(0);
    int nqpts_ibz = _Luv_.extent(0);
    auto nocc_max = max_nocc_per_kpoint(nocc_per_kp);
    utils::check(vexx.shape() == std::array<long,4>{nstot*nptot,nkpts,nbnd,nbnd},"KPTHCOps: Size mismatch"); 
    utils::check(hij.shape() == std::array<long,4>{nstot,nkpts,nptot*nbnd,nptot*nbnd},"KPTHCOps: Size mismatch"); 
    utils::check(haj.shape() == std::array<long,3>{ndet,nel,npol*NMO},"KPTHCOps: Size mismatch"); 
    utils::check(_Xsiu_.shape() == std::array<long,4>{nstot,nkpts,npol*nbnd,nu},"KPTHCOps: Size mismatch"); 
    utils::check(_Ydsau_.shape() == std::array<long,6>{ndet,nspin,npol,nkpts,nocc_max,nu},"KPTHCOps: Size mismatch"); 
    utils::check(_Luv_.shape() == std::array<long,3>{nqpts_ibz,nu,nv},"KPTHCOps: Size mismatch"); 
    if(_Zuv_.has_value())
      utils::check(_Zuv_->shape() == std::array<long,3>{nqpts_ibz,nu,nu},"KPTHCOps: Size mismatch"); 
    if(_Xsiu_rot_.has_value())
      utils::check(_Xsiu_rot_->shape() == std::array<long,4>{nstot,nkpts,npol*nbnd,nu_rot},"KPTHCOps: Size mismatch"); 
    if(_Ydsau_rot_.has_value())
      utils::check(_Ydsau_rot_->shape() == std::array<long,6>{ndet,nspin,npol,nkpts,nocc_max,nu_rot},"KPTHCOps: Size mismatch"); 
    if(_Zuv_rot_.has_value())
      utils::check(_Zuv_rot_->shape() == std::array<long,3>{nqpts_ibz,nu_rot,nu_rot},"KPTHCOps: Size mismatch"); 
  }

  nda::array<ComplexType,3> getOneBodyPropagatorMatrix(double dt,
                                                       nda::MemoryVector auto const& vMF)
  {
    using nda::range;
    auto all = range::all;
    int nspin  = (walker_type == COLLINEAR) ? 2 : 1;
    int npol  = (walker_type == NONCOLLINEAR) ? 2 : 1;
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
    int nwalk = G.extent(0);
    int nspin  = (walker_type == COLLINEAR) ? 2 : 1;
    int npol  = (walker_type == NONCOLLINEAR) ? 2 : 1;
    int nel  = (walker_type == COLLINEAR ? nup+ndown : nup); // NONCOLLINEAR has ndown=0
    utils::check(E.shape() == std::array<long,2>{nwalk,3}, "THC::energy: Size mismatch.");
    utils::check(G.extent(1) == nel*npol*NMO, "THC::energy: Size mismatch.");
    utils::check_strides(E,G);
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
    const auto Zuv = ( has_rot ? (*_Zuv_rot_)() : (*_Zuv_)() );

    int nu = Zuv.extent(1);

    utils::check(G.is_contiguous(), "Layout mismatch");
    memory::array_view<MEM,const ComplexType,5> G5d(std::array<long,5>{nwalk,nel,npol,nkpts,nbnd},G.data());

    if constexpr (MEM==HOST_MEMORY)
    {
      memory::buffered_array<MEM,ComplexType,2> Wvu(nu,nu);
      memory::buffered_array<MEM,ComplexType,2> Wuv(nu,nu);
      memory::buffered_array<MEM,ComplexType,3> Guu(2,nkpts,nu);
      for(int iw=0; iw<nwalk; ++iw) 
      {
        Guu() = ComplexType(0.0);
        for (int ispin = 0; ispin < nspin; ++ispin)
        {
          for (int p1 = 0; p1 < npol; ++p1)
          {
            for (int p2 = 0; p2 < npol; ++p2)
            {

              // Guv(k1,k2,0,u,v) = sum_a,j Y(d,s,k1,a,u) * G(w,k1,a,k2,j) * X(s,k2,j,v)
              auto GKK = Guv_from_compact(G5d(range(iw,iw+1), nda::ellipsis{}),ispin,p1,p2,idet);
              
 
              // MAM: Is this always the case???
              for (int k1 = 0, k12=0; k1 < nkpts; ++k1) {
              for (int k2 = k1; k2 < nkpts; ++k2, ++k12) {

                ComplexType k_scl = (k1==k2 ? 1.0 : 2.0);
                if(p1==p2) {
                //accumulate Guu
                  // find q such that qk_to_k2(q,k1) = k2
                  int iq=-1;
                  for(int i=0; i<nkpts; i++)
                    if(qk_to_k2(i,k1)==k2) {
                      iq=i;
                      break;
                    }
                  utils::check(iq>=0, "Error: Problems mapping {k1,k2} to q.");
                  Guu(0,iq,all) += k_scl*nda::diagonal(GKK(k1,k2,0,all,all));
                  // now right hand side, where need to find q such that qk_to_k2(q,k2) = k1
                  iq=-1;
                  for(int i=0; i<nkpts; i++)
                    if(qk_to_k2(i,k2)==k1) {
                      iq=i;
                      break;
                    }
                  utils::check(iq>=0, "Error: Problems mapping {k2,k1} to q.");
                  Guu(1,iq,all) += k_scl*nda::diagonal(GKK(k1,k2,0,all,all));
                }

                // sum_k1_k2_u_v G(k1,k2,u,v) * sum_q Z(q,u,v) * G(k2-q,k1-q,v,u)
                // sum_k1_k2_u_v G(k1,k2,u,v) * sum_q conj(Z(q,v,u)) * G(k2-q,k1-q,v,u)
                // sum_k1_k2_u_v G(k1,k2,u,v) * T(k1,k2,v,u) 
                Wvu() = ComplexType(0.0);
                auto W1d = nda::flatten(Wvu);
                for (int iq = 0; iq < nkpts; ++iq) {
                  int k1_ = qk_to_k2(iq,k1); // k1-q
                  int k2_ = qk_to_k2(iq,k2); // k2-q
                  W1d += nda::conj(nda::flatten(Zuv(iq,all,all))) * 
                                       nda::flatten(GKK(k2_,k1_,0,all,all));
                }

                Wuv() = nda::transpose(Wvu());
                E(iw,1) += ComplexType(-0.5*k_scl*scl/double(nkpts)) * 
                           nda::blas::dot(nda::flatten(GKK(k1,k2,0,all,all)),nda::flatten(Wuv));  

/*
                for (int iq = 0; iq < nkpts; ++iq) {
                  int qk1 = qk_to_k2(iq,k1);
                  int qk2 = qk_to_k2(iq,k2);
                  int n0 = ( qk2==0 ? 0 : nda::sum(nocc_per_kp(ispin,range(qk2))) );
                  int nel_qk2 = nocc_per_kp(ispin,qk2);
                  memory::array_view<MEM,ComplexType,3> Tau(std::array<long,3>{nw,nel_qk2,nu},Tbuff.data());

                  Wuv() = Guv();
                  if constexpr (MEM==HOST_MEMORY) {
                    for(int i=0; i<nw; ++i)
                      Wuv(i,all,all) *= Zuv(iq,all,all);
                  } else {
                    nda::tensor::elementwise(1.0,Zuv(iq,all,all),"uv",
                                             1.0,Wuv,"wuv",nda::tensor::binary_op::PROD);
                  }

                  // R[w,u][b] = sum_v Guv[w,u][v] * Yau[b][v]
                  auto Yau = Ysau(ispin,p2,qk2,range(nel_qk2),all);
                  nda::tensor::contract(Yau,"av",Wuv,"wuv",Tau,"wau"); 
 
                  // reuse Guv memory
                  memory::array_view<MEM,ComplexType,3> Twbi(std::array<long,3>{nw,nel_qk2,nbnd},Wuv.data());
                  //T[w][b][k] = sum_u R[w][u][b] * Xiu[k][u]
                  auto Xiu = Xsiu(is_,qk1,range(ip1_*nbnd,(ip1_+1)*nbnd),all);
                  nda::tensor::contract(Tau,"wau",Xiu,"iu",Twbi,"wai"); 
  
                  // E[w] = sum_ai T[w][a][i] * G[w][a][i] 
                  auto Gwai = G5d(range(iw, iw + nw),range(ispin*nup+n0,ispin*nup+n0+nel_qk2),p1,qk1,all); 
                  memory::buffered_array<MEM,ComplexType,1> Ew(nw);
                  nda::tensor::contract(ComplexType(-0.5*scl/double(nkpts)),Twbi,"wai",Gwai,"wai",ComplexType(0.0),Ew,"w"); 
                  nda::tensor::add(ComplexType(1.0),Ew,ComplexType(1.0),E(range(iw, iw + nw), 1));

                } // iq
*/
              } // k2
              } //k1
            } //p2
          } // p1
        } //is
        if (addEJ)
        {
          memory::buffered_array<MEM,ComplexType,2> Tqu(nkpts,nu);
          for (int iq = 0; iq < nkpts; ++iq) 
            nda::blas::gemv(nda::transpose(Zuv(iq,all,all)),Guu(0,iq,all),Tqu(iq,all));
          E(iw,2) += ComplexType(RealType(0.5*scl*scl/nkpts)) * 
                       nda::blas::dot(nda::flatten(Tqu),nda::flatten(Guu(1,nda::ellipsis{})));
        }
      } // iw
    } else {
      utils::check(false,"finish");
    }
  }

  void energy([[maybe_unused]] SpinTypes spin_component,
              [[maybe_unused]] nda::MemoryArrayOfRank<2> auto && E,
              [[maybe_unused]] nda::MemoryArrayOfRank<2> auto const& Gc,
              [[maybe_unused]] int nd,
              [[maybe_unused]] nda::MemoryArrayOfRank<2> auto && EJn,
              [[maybe_unused]] bool addH1  = true,
              [[maybe_unused]] bool addEJ  = true,
              [[maybe_unused]] bool addEXX = true)
  {
    utils::check(false," Error: spin-dependent energy not implemented ");
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
  auto vHS(nda::MemoryMatrix auto&& X, double dt)
  {
    constexpr MEMORY_SPACE MEM_X = memory::get_memory_space<decltype(X)>();
    static_assert(MEM == MEM_X, "Memory space mismatch");
    using nda::range;
    auto all = range::all;
    int nchol = 2 * nkpts * _Luv_().extent(2);
    int nwalk = X.extent(0);
    int nstot = hij.extent(0);
    int nptot = hij.extent(2)/nbnd;

    utils::check(X.is_contiguous(), "Layout mismatch");
    utils::check_strides(X);
    utils::check(X.shape() == std::array<long,2>{nwalk,nchol}, "THC::vbias: Size mismatch.");

    // Note: Allocate first, to make better use of memory pool
    // vHS[nspin_in_vHS][nwalk][npol_in_vHS*NMO][NMO]
    memory::buffered_array<MEM_X,ComplexType,4> v(nstot,nwalk,nptot*NMO,NMO);
    auto v7d = nda::reshape(v,std::array<long,7>{nstot,nwalk,nptot,nkpts,nbnd,nkpts,nbnd});
    v() = ComplexType(0.0);

    // scale by sqrt(dt)
    RealType a(std::sqrt(dt/double(nkpts))*0.5);

    // get array_views to the correct data and correct determinant
    bool has_rot = _Xsiu_rot_.has_value();
    const auto Xsiu = ( has_rot ? (*_Xsiu_rot_)() : _Xsiu_() );
    const auto Luv = _Luv_(); 
    int nu = Luv.extent(1);

    auto X4d = nda::reshape(X,std::array<long,4>{nwalk,2,nkpts,nu});

    {
      // "rotate" X
      //  XIJ = 0.5*a*(Xn+ -i*Xn-), XJI = 0.5*a*(Xn+ +i*Xn-)
      memory::buffered_array<MEM,ComplexType,3> T(nwalk,nkpts,nu);
      T() = X4d(all,0,all,all);
      nda::tensor::add(ComplexType(0.0,-a),X4d(all,1,all,all),"wqn",ComplexType(a),T,"wqn");
      nda::tensor::add(ComplexType(a),X4d(all,0,all,all),"wqn",ComplexType(0.0,a),X4d(all,1,all,all),"wqn");
      X4d(all,0,all,all) = T();
      //  then combine Q/(-Q) pieces if Q != -Q
      //  X(Q)np = (X(Q)np + X(-Q)nm)
      for(int iq=0; iq<nkpts; iq++)  {
        if( iq != minusq(iq) ) {
          nda::tensor::add(ComplexType(1.0),X4d(all,1,minusq(iq),all),"wn",ComplexType(1.0),X4d(all,0,iq,all),"wn");
        }
      } 
    }

    // work array
    memory::buffered_array<MEM,ComplexType,3> Twqu(nwalk,nkpts,nu);
    // T(w,q,u) = sum_v L(q,u,n) * X(w,0,q,n) 

    if constexpr (MEM==HOST_MEMORY) {
      for(int iw=0; iw<nwalk; ++iw)
        for(int iq=0; iq<nkpts; ++iq)
          nda::blas::gemv(ComplexType(1.0),Luv(iq,all,all),X4d(iw,0,iq,all),
                          ComplexType(0.0),Twqu(iw,iq,all));
    } else {
      nda::tensor::contract(ComplexType(1.0),X4d(all,0,all,all),"wqn",Luv,"qun",
                            ComplexType(0.0),Twqu,"wqu");
    }  

    // v[w][is*npol+ip][i][j] = sum_u conj(X[is][ip*NMO+i][u]) * X[is][ip*NMO+j][u] * T[u][w] 
    if constexpr (MEM==HOST_MEMORY) {

      memory::buffered_array<MEM,ComplexType,3> v_(nwalk,nbnd,nbnd);
      auto v2d = nda::reshape(v_,std::array<long,2>{nwalk*nbnd,nbnd});
      memory::buffered_array<MEM,ComplexType,3> Qwiu(nwalk,nbnd,nu);
      auto Q2d = nda::reshape(Qwiu,std::array<long,2>{nwalk*nbnd,nu});
      for(int ik=0; ik<nkpts; ++ik) {
        for(int iq=0; iq<nkpts; ++iq) { 
          int k2 = qk_to_k2(iq,ik);
          for( int is=0; is<nstot; ++is) {
            for( int ip=0; ip<nptot; ++ip) {
        
              auto Xiu = Xsiu(is,ik,range(ip*nbnd,(ip+1)*nbnd),all); 
              auto Xju = Xsiu(is,k2,range(ip*nbnd,(ip+1)*nbnd),all); 
 
              // Qwiu[w][i][u] = T[w][u] * conj(Piu[i][u])
              for(int w=0; w<nwalk; ++w)
                for(int i=0; i<nbnd; ++i)
                  Qwiu(w,i,all) = Twqu(w,iq,all) * nda::conj(Xiu(i,all));

              // v(nstot,nwalk,nptot,nkpts,nbnd,nkpts,nbnd)
              nda::blas::gemm(ComplexType(1.0),Q2d,nda::transpose(Xju),ComplexType(0.0),v2d);
              v7d(is,all,ip,ik,all,k2,all) += v_;

            }  // ip
          } // is
        } // iq
      } // ik

      memory::array_view<MEM,ComplexType,2> Twu(std::array<long,2>{nwalk,nu},Twqu.data());
      for(int iq=0; iq<nkpts; ++iq) {
        if(iq != minusq(iq)) continue;

        // T(w,q,u) = sum_v L(q,u,n) * X(w,0,q,n) 
        // can submit all q's with group batched gemm
        nda::tensor::contract(ComplexType(1.0),X4d(all,1,iq,all),"wn",
                                               nda::conj(Luv(iq,all,all)),"un",
                              ComplexType(0.0),Twu,"wu");

        for(int ik=0; ik<nkpts; ++ik) {
          int k2 = qk_to_k2(iq,ik);
          for( int is=0; is<nstot; ++is) {
            for( int ip=0; ip<nptot; ++ip) {
        
              auto Xiu = Xsiu(is,k2,range(ip*nbnd,(ip+1)*nbnd),all);
              auto Xju = Xsiu(is,ik,range(ip*nbnd,(ip+1)*nbnd),all);
 
              // Qwiu[w][i][u] = T[w][u] * Piu[i][u]
              for(int w=0; w<nwalk; ++w)
                for(int i=0; i<nbnd; ++i)
                  Qwiu(w,i,all) = Twu(w,all) * Xiu(i,all);

              nda::blas::gemm(ComplexType(1.0),Q2d,nda::dagger(Xju),ComplexType(0.0),v2d);
              v7d(is,all,ip,k2,all,ik,all) += v_; 

            }  // ip
          } // is
        } // ik
      } // iq

      // now missing contributions from q==-q terms
    } else {

      memory::buffered_array<MEM,ComplexType,4> Qwiu(nkpts,nwalk,nbnd,nu);
      memory::buffered_array<MEM,ComplexType,4> v_(nkpts,nwalk,nbnd,nbnd);
      auto Qk = nda::reshape(Qwiu,std::array<long,3>{nkpts,nwalk*nbnd,nu});
      auto vk = nda::reshape(v_,std::array<long,3>{nkpts,nwalk*nbnd,nbnd});
      using A_t = decltype(Qk(0,all,all));
      using B_t = decltype(nda::transpose(Xsiu(0,0,range(1),all)));
      using B2_t = decltype(nda::dagger(Xsiu(0,0,range(1),all)));
      using C_t = decltype(vk(0,all,all));
      std::vector<A_t> Av; 
      std::vector<B_t> Bv;
      std::vector<B2_t> B2v;
      std::vector<C_t> Cv;
      Av.reserve(nkpts);
      Bv.reserve(nkpts);
      B2v.reserve(nkpts);
      Cv.reserve(nkpts);
      for(int ik=0; ik<nkpts; ++ik) Cv.emplace_back(vk(ik,all,all)); 

      for( int is=0; is<nstot; ++is) {
        for( int ip=0; ip<nptot; ++ip) {
          for(int iq=0; iq<nkpts; ++iq) { 

            // Qwiu[w][i][u] = T[w][u] * conj(Piu[i][u])
            auto Xkiu = Xsiu(is,all,range(ip*nbnd,(ip+1)*nbnd),all); 
            nda::tensor::elementwise_trinary(1.0,Twqu(all,iq,all),"wu",
              1.0,nda::conj(Xkiu),"kiu",
              0.0,Qwiu,"kwiu",
              nda::tensor::binary_op::PROD,nda::tensor::binary_op::SUM);

            // v(nstot,nwalk,nptot,nkpts,nbnd,nkpts,nbnd)
            Av.clear(); 
            Bv.clear(); 
            for(int ik=0; ik<nkpts; ++ik) {
              int k2 = qk_to_k2(iq,ik);
              Av.emplace_back(Qk(ik,all,all));
              Bv.emplace_back(nda::transpose(Xsiu(is,k2,range(ip*nbnd,(ip+1)*nbnd),all)));
            } 
            nda::blas::gemm_batch<false>(ComplexType(1.0),Av,Bv,ComplexType(0.0),Cv);

            // use non-blocking copy
            for(int ik=0; ik<nkpts; ++ik) { 
              int k2 = qk_to_k2(iq,ik);
              nda::tensor::add(ComplexType(1.0),v_(ik,all,all,all),"wij",
                               ComplexType(1.0),v7d(is,all,ip,ik,all,k2,all),"wij");
            }

          } // iq
        }  // ip
      } // is

      memory::array_view<MEM,ComplexType,2> Twu(std::array<long,2>{nwalk,nu},Twqu.data());
      for( int is=0; is<nstot; ++is) {
        for( int ip=0; ip<nptot; ++ip) {
          for(int iq=0; iq<nkpts; ++iq) {
            if(iq != minusq(iq)) continue;

            // T(w,q,u) = sum_v L(q,u,n) * X(w,0,q,n) 
// evaluate all with batched gemm
            nda::tensor::contract(ComplexType(1.0),X4d(all,1,iq,all),"wn",
                                                   nda::conj(Luv(iq,all,all)),"un",
                                  ComplexType(0.0),Twu,"wu");

            auto Xkiu = Xsiu(is,all,range(ip*nbnd,(ip+1)*nbnd),all);
            // Qwiu[w][i][u] = T[w][u] * Piu[i][u]
            nda::tensor::elementwise_trinary(1.0,Twu,"wu",
                      1.0,Xkiu,"kiu",
                      0.0,Qwiu,"kwiu",
                      nda::tensor::binary_op::PROD,nda::tensor::binary_op::SUM);

            // v(nstot,nwalk,nptot,nkpts,nbnd,nkpts,nbnd)
            Av.clear();
            B2v.clear();
            for(int ik=0; ik<nkpts; ++ik) {
              int k2 = qk_to_k2(iq,ik);
              Av.emplace_back(Qk(k2,all,all));
              B2v.emplace_back(nda::dagger(Xsiu(is,ik,range(ip*nbnd,(ip+1)*nbnd),all)));
            }
            nda::blas::gemm_batch<false>(ComplexType(1.0),Av,B2v,ComplexType(0.0),Cv);

            // use non-blocking copy
            for(int ik=0; ik<nkpts; ++ik) {
              int k2 = qk_to_k2(iq,ik);
              nda::tensor::add(ComplexType(1.0),v_(ik,all,all,all),"wij",
                               ComplexType(1.0),v7d(is,all,ip,k2,all,ik,all),"wij");
            }

          } // iq
        } // ip
      } // is

    } // MEM
    return v;
  }

  void vbias(nda::MemoryArrayOfRank<2> auto const& G, nda::MemoryArrayOfRank<2> auto& v, double dt)
  {
    memory::check_memory_space<MEM>(G,v);
    using nda::range;
    auto all = range::all;
    int nchol = 2 * nkpts * _Luv_().extent(2);
    int nwalk = G.extent(0);
    int nspin  = (walker_type == COLLINEAR) ? 2 : 1;
    int npol  = (walker_type == NONCOLLINEAR) ? 2 : 1;
    int nel  = (walker_type == COLLINEAR ? nup+ndown : nup); // NONCOLLINEAR has ndown=0 
    utils::check_strides(G,v);
    utils::check(v.shape() == std::array<long,2>{nwalk,nchol}, "THC::vbias: Size mismatch.");
    if(haj.extent(0) == 1) // ndet==1, G half rotated
      utils::check(G.extent(1) == nel*npol*NMO, "THC::vbias: Size mismatch.");
    else // ndet>1, full G 
      utils::check(G.extent(1) == nspin*npol*NMO*npol*NMO, "THC::vbias: Size mismatch.");
    // needed to build G3d easily, can build from strides if needed
    utils::check(G.is_contiguous(), "Layout mismatch");
    utils::check(v.is_contiguous(), "Layout mismatch");

    // finish!!!
    utils::check(npol==1, "Error: vbias with npol>1 not yet implemented");

    // scale a by sqrt(dt)
    RealType a(std::sqrt(dt/double(nkpts))*0.5);

    // get array_views to the correct data and correct determinant
    const auto Luv = _Luv_(); 
    int nu = Luv.extent(1);
    int nv = Luv.extent(2);

    // v4d
    auto v4d = nda::reshape(v, std::array<long,4>{nwalk,2,nkpts,nv});

    // zero
    v() = ComplexType(0.0);

    if (haj.extent(0) == 1)
    {
      // can loop over walker blocks when memory is limited
      memory::array_view<MEM,const ComplexType,5> G5d(std::array<long,5>{nwalk,nel,npol,nkpts,nbnd},G.data());
      auto GKK = Gu_from_compact(G5d,0);

      if constexpr (MEM==HOST_MEMORY) {

        memory::buffered_array<MEM,ComplexType,2> Awu(nwalk,nu); 
        memory::buffered_array<MEM,ComplexType,2> Awn(nwalk,nv); 
        memory::buffered_array<MEM,ComplexType,2> Gwu(nwalk,nu); 
        for(int iq=0; iq<nkpts; iq++) {

          Gwu() = ComplexType(0.0);
          for(int ik=0; ik<nkpts; ik++) {
            int k2 = qk_to_k2(iq,ik);
            // G(w,u) += sum_a,j Y(d,s,ik,a,u) * G(w,ik,a,k2,j) * X(s,k2,j,u)
            Gwu() += GKK(ik,k2,all,all);
          }

          //nda::tensor::contract(Twu,"wqu",Luv(iq,all,all),"qun",Awn,"wqn");
          nda::blas::gemm(Gwu,Luv(iq,all,all),Awn);

          //v+(w,q,n) += 0.5*T(w,q,n) 
          v4d(all,0,iq,all) += ComplexType(a) * Awn();

          //v-(w,q,n) += -i*0.5*T(w,q,n) 
          v4d(all,1,iq,all) += ComplexType(0.0,-a) * Awn();

          //if(q != -q) v(w,-q,n) -= i*0.5*T(w,q,n)
          if(iq != minusq(iq)) {
            v4d(all,0,minusq(iq),all) += ComplexType(a) * Awn();
            v4d(all,1,minusq(iq),all) += ComplexType(0.0,a) * Awn();
          } else {
 
            Gwu() = ComplexType(0.0);
            for(int ik=0; ik<nkpts; ik++) {
              int k2 = qk_to_k2(iq,ik);
              // G(w,u) += sum_a,j Y(d,s,k-q,a,u) * G(w,k-q,a,k,j) * X(s,k,j)
              Gwu() += GKK(k2,ik,all,all);
            }
              
            //T(w,n) = sum_u G(w,u)* conj(L(q,u,n))   ***conjugating G and T instead of L***
            Gwu() = nda::conj(Gwu());
            nda::blas::gemm(Gwu,Luv(iq,all,all),Awn);
            Awn() = nda::conj(Awn());
             
            //v+(w,q,n) += 0.5*T(w,q,n) 
            v4d(all,0,iq,all) += ComplexType(a) * Awn();
            //v-(w,q,n) -= i*0.5*T(w,q,n) 
            v4d(all,1,iq,all) += ComplexType(0.0,a) * Awn();

          }

        } // iq
   
      } else {
        utils::check(false,"finish");
/*
        memory::buffered_array<MEM,ComplexType,3> Awu(nwalk,nkpts,nu); 
        memory::array_view<MEM,ComplexType,3> Awn(std::array<long,3>{nwalk,nkpts,nv},Awu.data()); 
        memory::buffered_array<MEM,ComplexType,3> Bwu(nwalk,nkpts,nu); 
        for(int ik=0; ik<nkpts; ik++) {

          // A(w,k2,u) = sum_a,j Y(d,s,k,a,u) * G(w,k,a,k2,j) * X(s,k2,j)
          Gu_from_compact(G5d, Awu, ik, range(nkpts), 0);

          // B(q) = A(k-q), now reorder according to k-q
          for(int iq=0; iq<nkpts; iq++) {
            int k2 = qk_to_k2(iq,ik);
            Bwu(all,iq,all) = Awu(all,k2,all);
          }

          //A(w,q,n) = sum_u B(w,q,u)*L(q,u,n)
          nda::tensor::contract(Bwu,"wqu",Luv,"qun",Awn,"wqn"); 

          //v+(w,q,n) += 0.5*T(w,q,n) 
          nda::tensor::add(ComplexType(a), Awn, "wqn", 
                           ComplexType(1.0), v4d(all,0,all,all), "wqn");

          //v-(w,q,n) += -i*0.5*T(w,q,n) 
          nda::tensor::add(ComplexType(0.0,-a), Awn, "wqn", 
                           ComplexType(1.0), v4d(all,1,all,all), "wqn");

          // how to batch these???
          for(int iq=0; iq<nkpts; iq++) {
            //if(q != -q) v(w,-q,n) -= i*0.5*T(w,q,n)
            if(iq != minusq(iq)) {
              nda::tensor::add(ComplexType(a), Awn(all,iq,all), "wn", 
                               ComplexType(1.0), v4d(all,0,minusq(iq),all), "wn");
              nda::tensor::add(ComplexType(0.0,a), Awn(all,iq,all), "wn", 
                               ComplexType(1.0), v4d(all,1,minusq(iq),all), "wn");
            }
          }

          // how to batch these???
          // now v- contributions for q == -q
          for(int iq=0; iq<nkpts; iq++) {
            if(iq != minusq(iq)) continue;
            // reuse memory
            memory::array_view<MEM,ComplexType,2> A2dwu(std::array<long,2>{nwalk,nu},Awu.data());
            // 3d view with a single kpoint, needed to call Gu_from_compact
            memory::array_view<MEM,ComplexType,3> A3dwu(std::array<long,3>{nwalk,1,nu},Awu.data());
            memory::array_view<MEM,ComplexType,2> B2dwn(std::array<long,2>{nwalk,nv},Bwu.data()); 
            int k2 = qk_to_k2(iq,ik);
 
            // G(w,u) = sum_a,j Y(d,s,k-q,a,u) * G(w,k-q,a,k,j) * X(s,k,j)
            Gu_from_compact(G5d, A3dwu, k2, range(ik,ik+1), 0);
              
            //T(w,n) = sum_u G(w,u)* conj(L(q,u,n))
            nda::tensor::contract(A2dwu,"wu",nda::conj(Luv(iq,all,all)),"un",B2dwn,"wn");
             
            //v+(w,q,n) += 0.5*T(w,q,n) 
            nda::tensor::add(ComplexType(a), B2dwn, "wn", 
                             ComplexType(1.0), v4d(all,0,iq,all), "wn");

            //v-(w,q,n) -= i*0.5*T(w,q,n) 
            nda::tensor::add(ComplexType(0.0,a), B2dwn, "wn", 
                             ComplexType(1.0), v4d(all,1,iq,all), "wn");
          } // iq* 

        } // ik 
*/
      }
    }
    else
    {
      utils::check(false," Error: THC not yet implemented for multiple references.");
      // multideterminant is not half-rotated, so use Likn
      // which spin???
//      memory::array_view<MEM,const ComplexType,4> G3d(std::array<long,4>{nwalk,nspin,npol*NMO,npol*NMO},G.data());
    }
  }

  /// Returns the number of spins and polarizations in the VHS potential.
  std::tuple<int,int> vHS_dims() const {
    return std::make_tuple(_Xsiu_().shape()[0],_Xsiu_().shape()[2]/nbnd);
  }
  int number_of_ke_vectors() const { 
    utils::check(_Zuv_.has_value() or _Zuv_rot_.has_value(), "Missing Zuv/Zuv_rot.");
    if(_Zuv_.has_value()) return nkpts*_Zuv_->extent(1);
    else return nkpts*_Zuv_rot_->extent(1); 
  }
  int number_of_cholesky_vectors() const { return 2 * nkpts * _Luv_().extent(2); }

  nda::array<ComplexType, 2> getHSPotentials() 
  { return nda::array<ComplexType, 2>{}; }

protected:
  // G(k1,k2,w,u) = sum_a,j Y(d,s,k1,a,u) * G(w,k1,a,k2,j) * X(s,k2,p,j,u)
  auto Gu_from_compact(nda::MemoryArrayOfRank<5> auto const& G, int idet)
  {
    using nda::range;
    auto all = range::all;
    int nstot = hij.extent(0);
    int nptot = hij.extent(2)/nbnd;
    int nspin  = (walker_type == COLLINEAR) ? 2 : 1;
    int npol  = (walker_type == NONCOLLINEAR) ? 2 : 1;
    int nel  = (walker_type == COLLINEAR ? nup+ndown : nup); // NONCOLLINEAR has ndown=0 
    bool has_rot = _Xsiu_rot_.has_value();
    // [nstot][nkpts][nptot*nbnd][nu]
    const auto Xsiu = ( has_rot ? (*_Xsiu_rot_)() : _Xsiu_());
    // [nspin][npol][nkpts][nocc_max][nu]  (walker dims, not interaction dims)
    const auto Ysau = ( has_rot ? (*_Ydsau_rot_)()(idet,nda::ellipsis{}) :
                                 _Ydsau_()(idet,nda::ellipsis{}) );
    long nw   = G.extent(0);
    long nu   = Xsiu.extent(3);

    utils::check(G.shape() == std::array<long,5>{nw,nel,npol,nkpts,nbnd}, 
                 "THC::Gu_from_compact: Shape mismatch");

    memory::buffered_array<MEM,ComplexType,4> GKK(nkpts,nkpts,nw,nu);    
    GKK() = ComplexType(0.0);
    ComplexType a = (walker_type == CLOSED) ? ComplexType(2.0) : ComplexType(1.0);
    if constexpr (MEM==HOST_MEMORY) {
      for( int is=0; is<nspin; is++ ) {
        memory::buffered_array<MEM,ComplexType,2> Tau(nelec[is],nu);    
        for( int ip=0; ip<npol; ip++ ) {
          auto [is_,ip_] = interaction_block(is,ip,npol,nstot,nptot);
          for(int iw=0; iw<nw; ++iw) {

            for(int k2=0; k2<nkpts; k2++) {
              auto Xju = Xsiu(is_,k2,range(ip_*nbnd,(ip_+1)*nbnd),all);
              nda::blas::gemm(G(iw,range(is*nup,nup+is*ndown),ip,k2,all),Xju,Tau);

              for(int k1=0; k1<nkpts; k1++) {
                auto const& rows = nocc_per_kp(is,k1);
                int nel_k = int(rows.size());
                auto Yau = Ysau(is,ip,k1,range(nel_k),all);
                for(int i=0; i<nel_k; ++i)
                  GKK(k1,k2,iw,all) += ComplexType(a) * Tau(rows(i),all) * Yau(i,all);
              }  //k1
            } // k2 

          } // iw
        } // npol 
      } // nspin 
    } else {
      utils::check(false,"finish");
    }
    return GKK;
  }

  // G(k1,k2,w,u,v) = sum_a,j Y(d,s,k1,a,u) * G(w,k1,a,k2,j) * X(s,k2,p,j,v)
  auto Guv_from_compact(nda::MemoryArrayOfRank<5> auto const& G, int is, int p1, int p2, int idet)
  {
    using nda::range;
    auto all = range::all;
    int nstot = hij.extent(0);
    int nptot = hij.extent(2)/nbnd;
    int npol  = (walker_type == NONCOLLINEAR) ? 2 : 1;
    int nel  = (walker_type == COLLINEAR ? nup+ndown : nup); // NONCOLLINEAR has ndown=0
    bool has_rot = _Xsiu_rot_.has_value();
    // [nstot][nkpts][nptot*nbnd][nu]
    const auto Xsiu = ( has_rot ? (*_Xsiu_rot_)() : _Xsiu_());
    // [nspin][npol][nkpts][nocc_max][nu]  (walker dims, not interaction dims)
    const auto Ysau = ( has_rot ? (*_Ydsau_rot_)()(idet,nda::ellipsis{}) :
                                 _Ydsau_()(idet,nda::ellipsis{}) );
    long nw   = G.extent(0);
    long nu   = Xsiu.extent(3);

    utils::check(G.shape() == std::array<long,5>{nw,nel,npol,nkpts,nbnd}, 
                 "THC::Guv_from_compact: Shape mismatch");

    memory::buffered_array<MEM,ComplexType,5> GKK(nkpts,nkpts,nw,nu,nu);    
    GKK() = ComplexType(0.0);
    ComplexType a = (walker_type == CLOSED) ? ComplexType(2.0) : ComplexType(1.0);
    if constexpr (MEM==HOST_MEMORY) {
      memory::buffered_array<MEM,ComplexType,2> Tau(nelec[is],nu);
      memory::buffered_array<MEM,ComplexType,2> Tau_k(nelec[is],nu);
      auto [is_,ip_] = interaction_block(is,p2,npol,nstot,nptot);
      for(int iw=0; iw<nw; ++iw) {

        for(int k2=0; k2<nkpts; k2++) {
          auto Xju = Xsiu(is_,k2,range(ip_*nbnd,(ip_+1)*nbnd),all);
          nda::blas::gemm(G(iw,range(is*nup,nup+is*ndown),p2,k2,all),Xju,Tau);

          for(int k1=0; k1<nkpts; k1++) {
            auto const& rows = nocc_per_kp(is,k1);
            int nel_k = int(rows.size());
            auto Yau = Ysau(is,p1,k1,range(nel_k),all);
            if(contiguous_rows(rows)) {
              int r0 = rows(0);
              nda::blas::gemm(ComplexType(a),nda::transpose(Yau),Tau(range(r0,r0+nel_k),all),
                              ComplexType(1.0),GKK(k1,k2,iw,all,all));
            } else {
              auto Tk = Tau_k(range(nel_k),all);
              for(int i=0; i<nel_k; ++i) Tk(i,all) = Tau(rows(i),all);
              nda::blas::gemm(ComplexType(a),nda::transpose(Yau),Tk,
                              ComplexType(1.0),GKK(k1,k2,iw,all,all));
            }
          }  //k1
        } // k2

      } // iw
    } else {
      utils::check(false,"finish");
    }
    return GKK;
  }

protected:
  std::shared_ptr<utils::mpi_context_t<mpi3::communicator>> mpi;

  WALKER_TYPES walker_type;

  int NMO, nup, ndown;
  int nelec[2];
  int nkpts, nbnd;

  // For each (spin,kpoint): the list of occupied PsiT row indices at that kpoint.
  nda::array<nda::array<int,1>,2> nocc_per_kp;

  // BZ information
  nda::array<int,1> minusq;
  nda::array<int,2> qk_to_k2;
  int Q0_index = 0;

  // H1[nspin][nk][npol*nbnd][npol*nbnd]
  memory::const_shared_array<HOST_MEMORY,ComplexType,4> hij;

  // half rotated one body hamiltonian: [ndet][nup+ndn][npol*NMO]. Kept in full basis
  memory::const_shared_array<MEM,ComplexType,3> haj;

  // Xsiu[nspin][nk][npol*nbnd][nu]
  memory::const_shared_array<MEM,ComplexType,4> _Xsiu_;

  // Ydsau[ndet][nspin][ipol][nk][nocc_max][nu]
  memory::const_shared_array<MEM,ComplexType,6> _Ydsau_;

  // Luv[nu][nv]
  memory::const_shared_array<MEM,ComplexType,3> _Luv_;

  // Zuv[nu][nv]
  std::optional<decltype(_Luv_)> _Zuv_;

  // Xsiu_rot[nspin][npol*nbnd][nu_rot]
  std::optional<decltype(_Xsiu_)> _Xsiu_rot_;

  // Ydsau_rot[ndet][nspin][ipol][nocc_max][nu_rot]
  std::optional<decltype(_Ydsau_)> _Ydsau_rot_; 

  // Zuv_rot[nu_rot][nu_rot]
  std::optional<decltype(_Luv_)> _Zuv_rot_; 

  // vexx(i,l) = -0.5 * sum_j <ij|jl> : [nspin][nk][npol*nbnd][npol*nbnd]
  memory::const_shared_array<HOST_MEMORY,ComplexType,4> vexx;

  ComplexType E0;

};

} // namespace afqmc

} // namespace sfqmc

