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

#undef NDEBUG

#include "test_common.hpp"

#include "config.h"
#include "configuration.hpp"
#include "IO/AppAbort.hpp"
#include "utilities/Random.hpp"
#include "test_common.hpp"

#include "AFQMC/parameters.hpp"
#include "IO/app_loggers.h"

#include <stdio.h>
#include <string>
#include <vector>
#include <complex>

#include "AFQMC/Walkers/WalkerSet.hpp"
#include "AFQMC/Walkers/WalkerIO.hpp"

using std::complex;
using std::string;

namespace sfqmc
{
using namespace afqmc;


using namespace afqmc;

template<MEMORY_SPACE MEM>
void sharedwset_basic_walker_features(WALKER_TYPES wtype, bool finiteT)
{
  using Type = std::complex<double>;

  auto& mpi = utils::make_unit_test_mpi_context();

  int NMO = 8, nup = 2, ndown = 2, nwalkers = 10;
  const WalkerSetParameters wlk_params{.name = "wset0", .walker_type = wtype};
  if (wtype == NONCOLLINEAR)
  {
    nup = 4;
    ndown = 0;
  }
  int cnt(0);
  Type base(0.0);
  Type tot_weight(0.0);
  auto rng = std::make_shared<utils::RandomGenerator_t<>>();
  auto wset = [&]() {
  if(!finiteT){
    int M((wtype == NONCOLLINEAR) ? 2 * NMO : NMO);
    int nspin = (wtype == COLLINEAR ? 2 : 1);
    nda::array<Type, 3> initA_h(nspin, M, nup);
    initA_h() = Type(0.0);
    for (int i = 0; i < nup; i++)
      initA_h(0,i,i) = Type(0.22);
    if(wtype == COLLINEAR)
      for (int i = 0; i < ndown; i++)
        initA_h(1,i,i) = Type(0.33);

    // per-spin guess at true widths: alpha (M x nup), beta (M x ndown)
    std::vector<nda::matrix<ComplexType>> initA;
    initA.emplace_back(initA_h(0, nda::ellipsis{}));
    if(wtype == COLLINEAR)
      initA.emplace_back(initA_h(1, nda::range::all, nda::range(ndown)));

    auto ws = WalkerSet<MEM>(mpi, wlk_params, rng, wtype, initA, nwalkers);

    REQUIRE(ws.size() == nwalkers);
    for(int iw = 0; iw < ws.size(); ++iw)
    {
      auto w = ws[iw];
      for(int spin = 0; spin < nspin; spin++) {
        auto sm = w.SlaterMatrix(static_cast<SpinTypes>(spin));
        REQUIRE( sm.extent(0) == M );
        REQUIRE( sm.extent(1) == (spin == 0 ? nup : ndown) );
        REQUIRE(nda::to_host(sm) == initA[spin]());
      }
      w.set_property(WEIGHT,base * 1.0 + 0.5);
      w.set_property(OVLP,base * 1.0 + 0.5);
      w.set_property(E1_,base * 1.0 + 0.5);
      w.set_property(EXX_,base * 1.0 + 0.5);
      w.set_property(EJ_,base * 1.0 + 0.5);
      tot_weight += base * 1.0 + 0.5;
      base += Type(1.0);
      cnt++;
    }
    return ws;
  } else {
    int M((wtype == NONCOLLINEAR) ? 2 * NMO : NMO);
    int nspin = (wtype == COLLINEAR ? 2 : 1);
    // rank-4 finite-T guess {3, nspin, M, M}: slab 0 = U, slab 1 = D (as a
    // diagonal matrix), slab 2 = V.
    nda::array<Type, 4> initUDV_h(3, nspin, M, M);
    initUDV_h() = Type(0.0);
    for (int i = 0; i < M; i++)
    {
      initUDV_h(0,0,i,i) = Type(0.22);
      initUDV_h(1,0,i,i) = Type(0.44);
      initUDV_h(2,0,i,i) = Type(0.66);
    }
    if(wtype == COLLINEAR)
      for (int i = 0; i < M; i++)
      {
        initUDV_h(0,1,i,i) = Type(0.33);
        initUDV_h(1,1,i,i) = Type(0.55);
        initUDV_h(2,1,i,i) = Type(0.77);
      }

    // the guess is always supplied on the host, mirroring the T=0 constructor
    auto initUDV = initUDV_h();

    auto ws = WalkerSet<MEM>(mpi, wlk_params, rng, wtype, initUDV, nwalkers);

    REQUIRE(ws.size() == nwalkers);
    for(int iw = 0; iw < ws.size(); ++iw)
    {
      auto w = ws[iw];
      auto umat = w.UMatrix(Alpha);
      REQUIRE( umat.extent(0) == initUDV.extent(2) );
      REQUIRE( umat.extent(1) == M );
      REQUIRE(nda::to_host(w.UMatrix(Alpha)) == nda::to_host(initUDV(0,0,nda::ellipsis{})));
      if( ws.getWalkerType() == COLLINEAR ) {
        auto umatB = w.UMatrix(Beta);
        REQUIRE( umatB.extent(0) == initUDV.extent(2) );
        REQUIRE( umatB.extent(1) == M );
        REQUIRE( nda::to_host(w.UMatrix(Beta)) == nda::to_host(initUDV(0,1,nda::range::all,nda::range(M))));
      }
      w.set_property(WEIGHT,base * 1.0 + 0.5);
      w.set_property(OVLP,base * 1.0 + 0.5);
      w.set_property(E1_,base * 1.0 + 0.5);
      w.set_property(EXX_,base * 1.0 + 0.5);
      w.set_property(EJ_,base * 1.0 + 0.5);
      w.set_property(LOGSCL_UP,base * 1.0 + 0.5);
      w.set_property(LOGSCL_DN,base * 1.0 + 0.5);
      tot_weight += base * 1.0 + 0.5;
      base += Type(1.0);
      cnt++;
    }
    return ws;
  }
  }();


  REQUIRE(cnt == nwalkers);
  base = Type(0.0);
  cnt = 0;
  for(int iw = 0; iw < wset.size(); ++iw)
  {
    auto w = wset[iw];
    Type d_(base * 1.0 + 0.5);
    REQUIRE(Type(w.get_property(WEIGHT)) == d_);
    REQUIRE(Type(w.get_property(OVLP)) == d_);
    REQUIRE(Type(w.get_property(E1_)) == d_);
    REQUIRE(Type(w.get_property(EXX_)) == d_);
    REQUIRE(Type(w.get_property(EJ_)) == d_);
    if(finiteT){
      REQUIRE(Type(w.get_property(LOGSCL_UP)) == d_);
      REQUIRE(Type(w.get_property(LOGSCL_DN)) == d_);
    }
    base += Type(1.0);
    cnt++;
  }
  wset.reserve(20);
  REQUIRE(wset.capacity() == 20);
  base = Type(0.0);
  cnt = 0;
  for(int iw = 0; iw < wset.size(); ++iw)
  {
    auto w = wset[iw];
    REQUIRE(Type(w.get_property(WEIGHT)) == base * 1.0 + 0.5);
    REQUIRE(Type(w.get_property(OVLP)) == base * 1.0 + 0.5);
    REQUIRE(Type(w.get_property(E1_)) == base * 1.0 + 0.5);
    REQUIRE(Type(w.get_property(EXX_)) == base * 1.0 + 0.5);
    REQUIRE(Type(w.get_property(EJ_)) == base * 1.0 + 0.5);
    if(finiteT){
      REQUIRE(Type(w.get_property(LOGSCL_UP)) == base * 1.0 + 0.5);
      REQUIRE(Type(w.get_property(LOGSCL_DN)) == base * 1.0 + 0.5);
    }
    base += Type(1.0);
    cnt++;
  }
  for (int i = 0; i < wset.size(); i++)
  {
    Type i_(i);
    REQUIRE(Type(wset[i].get_property(WEIGHT)) == i_ * 1.0 + 0.5);
    REQUIRE(Type(wset[i].get_property(OVLP)) == i_ * 1.0 + 0.5);
    REQUIRE(Type(wset[i].get_property(E1_)) == i_ * 1.0 + 0.5);
    REQUIRE(Type(wset[i].get_property(EXX_)) == i_ * 1.0 + 0.5);
    REQUIRE(Type(wset[i].get_property(EJ_)) == i_ * 1.0 + 0.5);
    if(finiteT){
      REQUIRE(Type(wset[i].get_property(LOGSCL_UP)) == i_ * 1.0 + 0.5);
      REQUIRE(Type(wset[i].get_property(LOGSCL_DN)) == i_ * 1.0 + 0.5);
    }
  }
  for (int i = 0; i < wset.size(); i++)
  {
    Type i_(i);
    auto w = wset[i];
    REQUIRE(Type(w.get_property(WEIGHT)) == i_ * 1.0 + 0.5);
    REQUIRE(Type(w.get_property(OVLP)) == i_ * 1.0 + 0.5);
    REQUIRE(Type(w.get_property(E1_)) == i_ * 1.0 + 0.5);
    REQUIRE(Type(w.get_property(EXX_)) == i_ * 1.0 + 0.5);
    REQUIRE(Type(w.get_property(EJ_)) == i_ * 1.0 + 0.5);
    if(finiteT){
      REQUIRE(Type(w.get_property(LOGSCL_UP)) == i_ * 1.0 + 0.5);
      REQUIRE(Type(w.get_property(LOGSCL_DN)) == i_ * 1.0 + 0.5);
    }
  }
  REQUIRE(wset.get_target_population() == nwalkers);
  REQUIRE(wset.get_global_target_population() == nwalkers * mpi->comm.size());
  REQUIRE(wset.GlobalPopulation() == nwalkers * mpi->comm.size());
  REQUIRE(wset.GlobalPopulation() == wset.get_global_target_population());
  REQUIRE(wset.NumBackProp() == 0);
  REQUIRE(wset.GlobalWeight() == tot_weight * Type(mpi->comm.size()));

  wset.scaleWeight(2.0);
  tot_weight *= 2.0;
  REQUIRE(wset.GlobalWeight() == tot_weight * Type(mpi->comm.size()));

  std::vector<ComplexType> Wdata;
  wset.processWalkerData(Wdata);
  wset.popControl();
  REQUIRE_THAT(wset.GlobalWeight(), utils::Approx(static_cast<RealType>(wset.get_global_target_population())));
  REQUIRE(wset.get_target_population() == nwalkers);
  REQUIRE(wset.get_global_target_population() == nwalkers * mpi->comm.size());
  REQUIRE(wset.GlobalPopulation() == nwalkers * mpi->comm.size());
  REQUIRE(wset.GlobalPopulation() == wset.get_global_target_population());
  REQUIRE_THAT(wset.GlobalWeight(), utils::Approx(static_cast<RealType>(wset.get_global_target_population())));
  for (int i = 0; i < wset.size(); i++)
  {
    auto w = wset[i];
    REQUIRE(w.get_property(EXX_) == w.get_property(E1_));
    REQUIRE(w.get_property(EJ_) == w.get_property(E1_));
  }

  if(!finiteT){
    auto SMs = wset.SlaterMatrices(Alpha);
    REQUIRE( SMs.extent(0) == wset.size() ); 
    if( wset.getWalkerType() == COLLINEAR ) { 
      auto SMBs = wset.SlaterMatrices(Beta);
      REQUIRE( SMBs.extent(0) == wset.size() ); 
    }

    // BP
    wset.resize_bp(4,10,2);
    {
      auto F0 = wset.getFields(0);
      auto Fs = wset.getFields();
      memory::array<MEM,ComplexType,2> Fi(F0.shape());
      Fi() = Fs(nda::range::all,0,nda::range::all); 
      wset.storeFields(1,Fi);
      auto WF = wset.getWeightFactors();
      auto WH = wset.getWeightHistory();
      WF() = ComplexType(0.0);
      WH() = ComplexType(0.0);
    }
  }
  else{
    auto UMats = wset.UMatrices(Alpha);
    REQUIRE( UMats.extent(0) == wset.size() ); 
    if( wset.getWalkerType() == COLLINEAR ) {
      auto UMatBs = wset.UMatrices(Beta);
      REQUIRE( UMatBs.extent(0) == wset.size() ); 
    }
    auto DVecs = wset.DMatrices(Alpha);
    REQUIRE( DVecs.extent(0) == wset.size() ); 
    if( wset.getWalkerType() == COLLINEAR ) {
      auto DVecBs = wset.DMatrices(Beta);
      REQUIRE( DVecBs.extent(0) == wset.size() ); 
    }
    auto VMats = wset.VMatrices(Alpha);
    REQUIRE( VMats.extent(0) == wset.size() ); 
    if( wset.getWalkerType() == COLLINEAR ) {
      auto VMatBs = wset.VMatrices(Beta);
      REQUIRE( VMatBs.extent(0) == wset.size() );
    }

    // No BP for finite temperature; exercise the per-sweep clean()/reset() cycle.
    wset.clean();
    wset.reset(nwalkers);
    REQUIRE(wset.size() == nwalkers);
  }
}

template<MEMORY_SPACE MEM>
void sharedwset_walker_io(WALKER_TYPES wtype)
{


  using Type = std::complex<double>;
  auto& mpi = utils::make_unit_test_mpi_context();

  int NMO = 8, nup = 2, ndown = 2, nwalkers = 10;
  if (wtype == NONCOLLINEAR)
  {
    nup = 4;
    ndown = 0;
  }

  int npol = (wtype == NONCOLLINEAR) ? 2 : 1;
  int nspin = wtype == COLLINEAR ? 2 : 1;
  nda::array<Type, 3> initA_h(nspin, npol * NMO, nup);
  initA_h() = 0.0;
  for(int spin = 0; spin < nspin; spin++) {
    for (int i = 0; i < nup; i++) { 
      initA_h(spin,i,i) = Type(0.11 * (spin + 2));
    }
  }
  auto rng = std::make_shared<utils::RandomGenerator_t<>>();

  // per-spin guess at true widths: alpha (npol*NMO x nup), beta (NMO x ndown)
  std::vector<nda::matrix<ComplexType>> initA;
  initA.emplace_back(initA_h(0, nda::ellipsis{}));
  if(wtype == COLLINEAR)
    initA.emplace_back(initA_h(1, nda::range::all, nda::range(ndown)));

  const WalkerSetParameters wlk_params{.name = "wset0", .walker_type = wtype};
  auto wset = WalkerSet<MEM>(mpi, wlk_params, rng, wtype, initA, nwalkers);

  REQUIRE(wset.size() == nwalkers);
  int cnt(0);
  Type base(0.0);
  Type tot_weight(0.0);
  for(int iw = 0; iw < wset.size(); ++iw)
  {
    auto w = wset[iw];
    for(int spin = 0; spin < nspin; spin++) {
      auto sm = w.SlaterMatrix(static_cast<SpinTypes>(spin));
      REQUIRE(sm.extent(0) == npol * NMO);
      REQUIRE(sm.extent(1) == (spin == 0 ? nup : ndown));
      REQUIRE(nda::to_host(sm) == initA[spin]());
    }
    w.set_property(WEIGHT,base * 1.0 + 0.1);
    w.set_property(OVLP,base * 1.0 + 0.2);
    w.set_property(E1_,base * 1.0 + 0.3);
    w.set_property(EXX_,base * 1.0 + 0.4);
    w.set_property(EJ_,base * 1.0 + 0.5);
    tot_weight += base * 1.0 + 0.5; // not used?
    base += Type(1.0);
    cnt++;
  }
  REQUIRE(cnt == nwalkers);

  // dump restart file
  {
    h5::file fh5;
    if(mpi->comm.root()) fh5 = h5::file(std::string("dummy_walkers.h5"),'w');
    dumpToHDF5(wset, fh5);
  }
  mpi->comm.barrier();

  {
    h5::file fh5(std::string("dummy_walkers.h5"),'r');
    auto wset2 = readWalkersFromHDF5<WalkerSet<MEM>>(mpi, wlk_params, rng,
                                                     wtype, fh5, nwalkers, true);
    std::array<walker_data,5> tags = {WEIGHT,OVLP,E1_,EXX_,EJ_};
    for (int i = 0; i < nwalkers; i++)
    {
      CHECK(nda::to_host(wset[i].SlaterMatrix(Alpha)) == nda::to_host(wset2[i].SlaterMatrix(Alpha)));
      for(auto v : tags)
        CHECK(wset[i].get_property(v) == wset2[i].get_property(v));
    }
  }

  mpi->comm.barrier();
  if (mpi->comm.root())
    remove("dummy_walkers.h5");

}

// MAM: Tests are not GPU enabled, fix direct access to GPU memory
TEST_CASE("sharedwset: basic walker features", "[sharedwset]")
{
  sharedwset_basic_walker_features<HOST_MEMORY>(CLOSED, false);
  sharedwset_basic_walker_features<HOST_MEMORY>(COLLINEAR, false);
  sharedwset_basic_walker_features<HOST_MEMORY>(NONCOLLINEAR, false);
  sharedwset_basic_walker_features<HOST_MEMORY>(COLLINEAR, true);
  sharedwset_basic_walker_features<HOST_MEMORY>(NONCOLLINEAR, true);
#if defined(ENABLE_DEVICE)
  sharedwset_basic_walker_features<DEVICE_MEMORY>(CLOSED, false);
  sharedwset_basic_walker_features<DEVICE_MEMORY>(COLLINEAR, false);
  sharedwset_basic_walker_features<DEVICE_MEMORY>(NONCOLLINEAR, false);
  sharedwset_basic_walker_features<DEVICE_MEMORY>(COLLINEAR, true);
  sharedwset_basic_walker_features<DEVICE_MEMORY>(NONCOLLINEAR, true);
#endif
}
TEST_CASE("sharedwset: walker io", "[sharedwset]")
{
  sharedwset_walker_io<HOST_MEMORY>(CLOSED);
  sharedwset_walker_io<HOST_MEMORY>(COLLINEAR);
  sharedwset_walker_io<HOST_MEMORY>(NONCOLLINEAR);
#if defined(ENABLE_DEVICE)
  sharedwset_walker_io<DEVICE_MEMORY>(CLOSED);
  sharedwset_walker_io<DEVICE_MEMORY>(COLLINEAR);
  sharedwset_walker_io<DEVICE_MEMORY>(NONCOLLINEAR);
#endif
}
} // namespace sfqmc
