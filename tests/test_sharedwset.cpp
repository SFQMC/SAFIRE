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

#include "IO/ptree/ptree_utilities.hpp"
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
void sharedwset_basic_walker_features(std::string wtype)
{
  using Type = std::complex<double>;

  auto& mpi = utils::make_unit_test_mpi_context();

  int NMO = 8, nup = 2, ndown = 2, nwalkers = 10;
  if (wtype == "noncollinear")
  {
    nup = 4;
    ndown = 0;
  }
  AFQMCInfo info;
  info.NMO  = NMO;
  info.nup = nup;
  info.ndown = ndown;
  info.name = "walker";
  int cnt(0);
  Type base(0.0);
  Type tot_weight(0.0);
  ptree wlk_pt;
  wlk_pt.put("name","wset0");
  wlk_pt.put("walker_type",wtype);
  auto rng = std::make_shared<utils::RandomGenerator_t<>>();
  auto wset = make_WalkerSet<MEM>(mpi, wlk_pt, info, rng);
  
  if(wtype != "collinear-ft" and wtype != "noncollinear-ft"){
    int M((wtype == "noncollinear") ? 2 * NMO : NMO);
    int nspin = (wtype == "collinear" ? 2 : 1); 
    nda::array<Type, 3> initA_h(nspin, M, nup);
    initA_h() = Type(0.0);
    for (int i = 0; i < nup; i++)
      initA_h(0,i,i) = Type(0.22);
    if(wtype == "collinear")
      for (int i = 0; i < ndown; i++)
        initA_h(1,i,i) = Type(0.33);

    auto initA = memory::to_memory_space<MEM>(initA_h);

    wset.resize(nwalkers, initA);

    REQUIRE(wset.size() == nwalkers);
    //Type base(0.0);
    //Type tot_weight(0.0);
    for (auto it = wset.begin(); it != wset.end(); ++it)
    {
      for(int spin = 0; spin < nspin; spin++) {
        auto sm = it->SlaterMatrix(static_cast<SpinTypes>(spin));
        REQUIRE( sm.extent(0) == initA.extent(1) );	
        REQUIRE( sm.extent(1) == nup );	
        REQUIRE(nda::to_host(sm) == initA_h(spin, nda::ellipsis{}));
      }
      it->set_property(WEIGHT,base * 1.0 + 0.5);
      it->set_property(OVLP,base * 1.0 + 0.5);
      it->set_property(E1_,base * 1.0 + 0.5);
      it->set_property(EXX_,base * 1.0 + 0.5);
      it->set_property(EJ_,base * 1.0 + 0.5);
      tot_weight += base * 1.0 + 0.5;
      base += Type(1.0);
      cnt++;
    }
  } else {
    int M((wtype == "noncollinear-ft") ? 2 * NMO : NMO);
    int nspin = (wtype == "collinear-ft" ? 2 : 1); 
    nda::array<Type, 3> initU_h(nspin, M, M);
    nda::array<Type, 2> initD_h(nspin, M);
    nda::array<Type, 3> initV_h(nspin, M, M);
    initU_h() = Type(0.0);
    for (int i = 0; i < M; i++)
      initU_h(0,i,i) = Type(0.22);
    if(wtype == "collinear-ft")
      for (int i = 0; i < M; i++)
        initU_h(1,i,i) = Type(0.33);
    initD_h() = Type(0.0);
    for (int i = 0; i < M; i++)
      initD_h(0,i) = Type(0.44);
    if(wtype == "collinear-ft")
      for (int i = 0; i < M; i++)
        initD_h(1,i) = Type(0.55);
    initV_h() = Type(0.0);
    for (int i = 0; i < M; i++)
      initV_h(0,i,i) = Type(0.66);
    if(wtype == "collinear-ft")
      for (int i = 0; i < M; i++)
        initV_h(1,i,i) = Type(0.77);

    auto initU = memory::to_memory_space<MEM>(initU_h);
    auto initD = memory::to_memory_space<MEM>(initD_h);
    auto initV = memory::to_memory_space<MEM>(initV_h);

    wset.resize(nwalkers, initU, initD, initV);

    REQUIRE(wset.size() == nwalkers);
    //int cnt(0);
    //Type base(0.0);
    //Type tot_weight(0.0);
    for (auto it = wset.begin(); it != wset.end(); ++it)
    {
      auto umat = it->UMatrix(Alpha);
      REQUIRE( umat.extent(0) == initU.extent(1) );	
      REQUIRE( umat.extent(1) == M );	
      REQUIRE(nda::to_host(it->UMatrix(Alpha)) == initU(0,nda::ellipsis{}));
      if( wset.getWalkerType() == COLLINEAR ) { 
        auto umatB = it->UMatrix(Beta);
        REQUIRE( umatB.extent(0) == initU.extent(1) );	
        REQUIRE( umatB.extent(1) == M );	
        REQUIRE( nda::to_host(it->UMatrix(Beta)) == initU(1,nda::range::all,nda::range(M)));
      }
      it->set_property(WEIGHT,base * 1.0 + 0.5);
      it->set_property(OVLP,base * 1.0 + 0.5);
      it->set_property(E1_,base * 1.0 + 0.5);
      it->set_property(EXX_,base * 1.0 + 0.5);
      it->set_property(EJ_,base * 1.0 + 0.5);
      it->set_property(LOGSCL_UP,base * 1.0 + 0.5);
      it->set_property(LOGSCL_DN,base * 1.0 + 0.5);
      tot_weight += base * 1.0 + 0.5;
      base += Type(1.0);
      cnt++;
    }
  }


  REQUIRE(cnt == nwalkers);
  base = Type(0.0);
  cnt = 0;
  for (auto it = wset.begin(); it != wset.end(); ++it)
  {
    Type d_(base * 1.0 + 0.5);
    REQUIRE(Type(it->get_property(WEIGHT)) == d_);
    REQUIRE(Type(it->get_property(OVLP)) == d_);
    REQUIRE(Type(it->get_property(E1_)) == d_);
    REQUIRE(Type(it->get_property(EXX_)) == d_);
    REQUIRE(Type(it->get_property(EJ_)) == d_);
    if(wtype == "collinear-ft" or wtype == "noncollinear-ft"){
      REQUIRE(Type(it->get_property(LOGSCL_UP)) == d_);
      REQUIRE(Type(it->get_property(LOGSCL_DN)) == d_);
    }
    base += Type(1.0);
    cnt++;
  }
  wset.reserve(20);
  REQUIRE(wset.capacity() == 20);
  base = Type(0.0);
  cnt = 0;
  for (auto it = wset.begin(); it != wset.end(); ++it)
  {
    REQUIRE(Type(it->get_property(WEIGHT)) == base * 1.0 + 0.5);
    REQUIRE(Type(it->get_property(OVLP)) == base * 1.0 + 0.5);
    REQUIRE(Type(it->get_property(E1_)) == base * 1.0 + 0.5);
    REQUIRE(Type(it->get_property(EXX_)) == base * 1.0 + 0.5);
    REQUIRE(Type(it->get_property(EJ_)) == base * 1.0 + 0.5);
    if(wtype == "collinear-ft" or wtype == "noncollinear-ft"){
      REQUIRE(Type(it->get_property(LOGSCL_UP)) == base * 1.0 + 0.5);
      REQUIRE(Type(it->get_property(LOGSCL_DN)) == base * 1.0 + 0.5);
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
    if(wtype == "collinear-ft" or wtype == "noncollinear-ft"){
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
    if(wtype == "collinear-ft" or wtype == "noncollinear-ft"){
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

  if(wtype != "collinear-ft" and wtype != "noncollinear-ft"){
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
    if( wset.getWalkerType() == COLLINEAR_FT ) { 
      auto UMatBs = wset.UMatrices(Beta);
      REQUIRE( UMatBs.extent(0) == wset.size() ); 
    }
    auto DVecs = wset.DMatrices(Alpha);
    REQUIRE( DVecs.extent(0) == wset.size() ); 
    if( wset.getWalkerType() == COLLINEAR_FT ) { 
      auto DVecBs = wset.DMatrices(Beta);
      REQUIRE( DVecBs.extent(0) == wset.size() ); 
    }
    auto VMats = wset.VMatrices(Alpha);
    REQUIRE( VMats.extent(0) == wset.size() ); 
    if( wset.getWalkerType() == COLLINEAR_FT ) { 
      auto VMatBs = wset.VMatrices(Beta);
      REQUIRE( VMatBs.extent(0) == wset.size() ); 
    }

    // No BP for finite temperature
  }
  wset.clean();
  REQUIRE(wset.size() == 0);
  REQUIRE(wset.capacity() == 0);
}

template<MEMORY_SPACE MEM>
void sharedwset_walker_io(std::string wtype)
{


  using Type = std::complex<double>;
  auto& mpi = utils::make_unit_test_mpi_context();

  int NMO = 8, nup = 2, ndown = 2, nwalkers = 10;
  if (wtype == "noncollinear")
  {
    nup = 4;
    ndown = 0;
  }

  AFQMCInfo info;
  info.NMO  = NMO;
  info.nup = nup;
  info.ndown = ndown;
  info.name = "walker";
  int npol = (wtype == "noncollinear") ? 2 : 1;
  int nspin = wtype == "collinear" ? 2 : 1;
  nda::array<Type, 3> initA_h(nspin, npol * NMO, nup);
  initA_h() = 0.0;
  for(int spin = 0; spin < nspin; spin++) {
    for (int i = 0; i < nup; i++) { 
      initA_h(spin,i,i) = Type(0.11 * (spin + 2));
    }
  }
  auto rng = std::make_shared<utils::RandomGenerator_t<>>();

  auto initA = memory::to_memory_space<MEM>(initA_h);
  
  ptree pt0;
  pt0.put("WalkerSet.name","wset0");
  pt0.put("WalkerSet.walker_type",wtype);
  auto wset = make_WalkerSet<MEM>(mpi, pt0.get_child("WalkerSet"), info, rng);
  wset.resize(nwalkers, initA);

  REQUIRE(wset.size() == nwalkers);
  int cnt(0);
  Type base(0.0);
  Type tot_weight(0.0);
  for (auto it = wset.begin(); it != wset.end(); ++it)
  {
    for(int spin = 0; spin < nspin; spin++) {
      auto sm = it->SlaterMatrix(static_cast<SpinTypes>(spin));
      REQUIRE(sm.extent(0) == initA.extent(1));
      REQUIRE(sm.extent(1) == initA.extent(2)); 
      REQUIRE(nda::to_host(sm) == nda::to_host(initA(spin,nda::ellipsis{})));
    }
    it->set_property(WEIGHT,base * 1.0 + 0.1);
    it->set_property(OVLP,base * 1.0 + 0.2);
    it->set_property(E1_,base * 1.0 + 0.3);
    it->set_property(EXX_,base * 1.0 + 0.4);
    it->set_property(EJ_,base * 1.0 + 0.5);
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
    auto wset2 = make_WalkerSet<MEM>(mpi, pt0.get_child("WalkerSet"), info, rng);
    restartFromHDF5(wset2, nwalkers, fh5, true);
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
  sharedwset_basic_walker_features<HOST_MEMORY>("closed");
  sharedwset_basic_walker_features<HOST_MEMORY>("collinear");
  sharedwset_basic_walker_features<HOST_MEMORY>("noncollinear");
  sharedwset_basic_walker_features<HOST_MEMORY>("fullypolarized");
  sharedwset_basic_walker_features<HOST_MEMORY>("collinear-ft");
  sharedwset_basic_walker_features<HOST_MEMORY>("noncollinear-ft");
#if defined(ENABLE_DEVICE)
  sharedwset_basic_walker_features<DEVICE_MEMORY>("closed");
  sharedwset_basic_walker_features<DEVICE_MEMORY>("collinear");
  sharedwset_basic_walker_features<DEVICE_MEMORY>("noncollinear");
  sharedwset_basic_walker_features<DEVICE_MEMORY>("fullypolarized");
#endif
}
TEST_CASE("sharedwset: walker io", "[sharedwset]")
{
  sharedwset_walker_io<HOST_MEMORY>("closed");
  sharedwset_walker_io<HOST_MEMORY>("collinear");
  sharedwset_walker_io<HOST_MEMORY>("noncollinear");
  sharedwset_walker_io<HOST_MEMORY>("fullypolarized");
#if defined(ENABLE_DEVICE)
  sharedwset_walker_io<DEVICE_MEMORY>("closed");
  sharedwset_walker_io<DEVICE_MEMORY>("collinear");
  sharedwset_walker_io<DEVICE_MEMORY>("noncollinear");
  sharedwset_walker_io<DEVICE_MEMORY>("fullypolarized");
#endif
}

// SLOT_LINEAGE tracking through WalkerSetBase::branch(). After branch() replicates each walker according
// to its branching count (dead-walker compaction then replication), SLOT_LINEAGE(w) must hold the
// PRE-branch local slot index of the walker now at slot w, so a conditioned stochastic trial can realign
// its slot-major inner ensemble with the post-branch outer walkers. branch() resets the column to identity
// (set_slot_lineage_identity()) BEFORE any reordering, and the per-walker whole-row copies then carry it
// through both phases -- no per-site bookkeeping. Algorithm-independent invariant: the multiset of
// post-branch lineage values equals { i repeated count[i] times }, since each surviving walker (and every
// clone) traces to its origin slot i, which is replicated exactly count[i] times.
template<MEMORY_SPACE MEM>
void stochastic_branch_lineage_metadata()
{
  using Type = std::complex<double>;
  auto& mpi  = utils::make_unit_test_mpi_context();
  if (mpi->comm.size() != 1)
    return; // single-rank lineage check; the cross-rank path is covered separately

  const int NMO = 6, nup = 2, ndown = 2, nwalk = 6;
  AFQMCInfo info;
  info.NMO   = NMO;
  info.nup   = nup;
  info.ndown = ndown;
  info.name  = "walker";
  ptree wlk_pt;
  wlk_pt.put("name", "wset0");
  wlk_pt.put("walker_type", "closed");
  auto rng  = std::make_shared<utils::RandomGenerator_t<>>();
  auto wset = make_WalkerSet<MEM>(mpi, wlk_pt, info, rng);

  nda::array<Type, 3> initA_h(1, NMO, nup);
  initA_h() = Type(0.0);
  for (int i = 0; i < nup; ++i)
    initA_h(0, i, i) = Type(0.22);
  auto initA = memory::to_memory_space<MEM>(initA_h);
  wset.resize(nwalk, initA);
  REQUIRE(wset.size() == nwalk);

  auto lineage_of = [&](int w) {
    nda::array<ComplexType, 1> lin(nwalk);
    wset.getProperty(SLOT_LINEAGE, lin);
    return int(real(lin(w)) + 0.5); // values are exact small non-negative integers
  };

  // resize() initializes SLOT_LINEAGE to identity.
  for (int w = 0; w < nwalk; ++w)
    REQUIRE(lineage_of(w) == w);

  // Hand-crafted branching counts (sum == nwalk == targetN_per_rank, so nothing spills to Wexcess):
  // slot 0 -> 2 copies, slot 1 killed, slot 2 -> 1, slot 3 -> 3, slots 4 and 5 killed.
  std::vector<int> counts = {2, 0, 1, 3, 0, 0};
  int total               = 0;
  for (int c : counts)
    total += c;
  REQUIRE(total == nwalk);

  std::vector<std::pair<double, int>> buffer(nwalk);
  for (int i = 0; i < nwalk; ++i)
    buffer[i] = {1.0, counts[i]};

  memory::array<MEM, ComplexType, 2> Wexcess(0, wset.single_walker_size());
  wset.branch(buffer.begin(), buffer.end(), Wexcess);
  REQUIRE(wset.size() == nwalk);

  // The multiset of post-branch lineage values must equal the branching counts.
  std::vector<int> hist(nwalk, 0);
  for (int w = 0; w < nwalk; ++w)
  {
    int parent = lineage_of(w);
    REQUIRE(parent >= 0);
    REQUIRE(parent < nwalk);
    hist[parent]++;
  }
  for (int i = 0; i < nwalk; ++i)
    REQUIRE(hist[i] == counts[i]);
}

TEST_CASE("stochastic_branch_lineage_metadata", "[stochastic_wfn]")
{
  app_log(0, "WalkerSetBase::branch() carries SLOT_LINEAGE through compaction + replication.");
  stochastic_branch_lineage_metadata<HOST_MEMORY>();
}
} // namespace sfqmc
