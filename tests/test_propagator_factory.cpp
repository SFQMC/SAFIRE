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

#include "catch2/catch_test_macros.hpp"

#include "config.h"
#include "IO/app_loggers.h"
#include "IO/ptree/ptree_utilities.hpp"
#include "utilities/Random.hpp"
#include "utilities/Timer.hpp"
#include "test_common.hpp"
#include "utilities/check.hpp"

#include <string>
#include <vector>
#include <complex>
#include <cmath>
#include <iomanip>

#include "nda/nda.hpp"
#include "nda/tensor.hpp"
#include "nda/h5.hpp"
#include "numerics/sparse/sparse.hpp"

#include "test_utils.hpp"
#include "AFQMC/Utilities/readWfn.h" 

#include "AFQMC/Hamiltonians/HamiltonianFactory.h"
#include "AFQMC/Hamiltonians/Hamiltonian.hpp"
#include "AFQMC/Wavefunctions/WavefunctionFactory.h"
#include "AFQMC/Propagators/PropagatorFactory.h"
#include "AFQMC/Walkers/WalkerSet.hpp"

using std::cerr;
using std::complex;
using std::cout;
using std::endl;
using std::ifstream;
using std::setprecision;
using std::string;

extern std::string UTEST_HAMIL, UTEST_WFN;

namespace sfqmc
{
using namespace afqmc;

template<MEMORY_SPACE MEM>
void propagator_factory_build(std::shared_ptr<utils::mpi_context_t<boost::mpi3::communicator>> mpi,
             std::string hamil_file, std::string wfn_file, bool dense_trial)
{
  using nda::range;

  int NMO = read_nmo_from_hdf(hamil_file);
  auto[wfn_NMO,nup, ndown] = read_info_from_wfn(wfn_file,"any");
  utils::check(NMO == wfn_NMO, "Error: NMO != wfn_NMO.");
  WALKER_TYPES type         = getWalkerType(wfn_file);
  int nspin                 = (type == COLLINEAR or type == COLLINEAR_FT) ? 2 : 1;
  int npol                  = (type == NONCOLLINEAR or type == NONCOLLINEAR_FT) ? 2 : 1;

  int ntau = 0;
  if(type == COLLINEAR_FT or type == NONCOLLINEAR_FT){
    ntau = nup;
    nup = NMO;
    ndown = NMO;
  }
  std::map<std::string, AFQMCInfo> InfoMap;
  InfoMap.insert(std::pair<std::string, AFQMCInfo>("info0", AFQMCInfo{"info0", NMO, nup, ndown, ntau}));

  ptree ham_pt;
  ham_pt.put("name","ham0");
  ham_pt.put("system","info0");
  ham_pt.put("filename",hamil_file);
  ham_pt.put("shift_1body",true);
  //ham_pt.put("shift_1body",false);

  HamiltonianFactory HamFac(InfoMap);
  HamFac.push("ham0", ham_pt);
  Hamiltonian& ham = HamFac.getHamiltonian(mpi, "ham0");

  int nwalk = 11; 
  std::shared_ptr<utils::RandomGenerator_t<HOST_MEMORY>> rng = std::make_shared<utils::RandomGenerator_t<HOST_MEMORY>>();
  std::shared_ptr<utils::RandomGenerator_t<MEM>> rng_dev = std::make_shared<utils::RandomGenerator_t<MEM>>(utils::make_rng<MEM>(777));

  ptree wlk_pt;
  wlk_pt.put("name","wset0");
  wlk_pt.put("walker_type", walkerTypeToString(type));
  auto wset = make_WalkerSet<MEM>(mpi, wlk_pt, InfoMap["info0"], rng);

  ptree wfn_pt;
  wfn_pt.put("name","wfn0");
  wfn_pt.put("system","info0");
  wfn_pt.put("filename",wfn_file);
  wfn_pt.put("dense_trial",dense_trial);

  WavefunctionFactory<MEM> WfnFac{};
  WfnFac.push("wfn0", wfn_pt);
  auto& wfn = WfnFac.getWavefunction(mpi, "wfn0", type, &ham, nwalk);

  if(type != COLLINEAR_FT and type != NONCOLLINEAR_FT)
  {
    auto initial_guess = WfnFac.getInitialGuess("wfn0"); 
    REQUIRE(initial_guess.shape() == std::array<long,3>{nspin,npol*NMO,nup});
    wset.resize(nwalk, initial_guess);
  }
  else
  {
    auto initial_guess_ft = WfnFac.getInitialGuess_ft("wfn0"); 
    REQUIRE(initial_guess_ft.shape() == std::array<long,4>{3,nspin,npol*NMO,NMO});
    wset.resize(nwalk, initial_guess_ft);
  }

  ptree prop_pt;
  prop_pt.put("name","prop0");
  prop_pt.put("system","info0");
  prop_pt.put("denseP2",true);

  PropagatorFactory<MEM> PropgFac(InfoMap);
  PropgFac.push("prop0", prop_pt);
  auto& prop = PropgFac.getPropagator(mpi, "prop0", wfn, rng_dev);

  std::cout << setprecision(8);
  wfn.Energy(wset);
  {
    ComplexType eav = 0, ov = 0;
    for (auto it = wset.begin(); it != wset.end(); ++it)
    {
      eav += it->get_property(WEIGHT) * (it->energy());
      ov += it->get_property(WEIGHT);
    }
    app_log(1," Initial Energy: {}", (eav / ov).real()); 
  }
  double tot_time = 0;
  RealType dt     = 0.01;
  RealType Eshift = std::abs(wset[0].get_property(OVLP));
  if(type != COLLINEAR_FT and type != NONCOLLINEAR_FT){
    for (int i = 0; i < 10; i++)
    {
      prop.Propagate(wset, Eshift, dt);
      wfn.Energy(wset);
      ComplexType eav = 0, ov = 0;
      for (auto it = wset.begin(); it != wset.end(); ++it)
      {
        eav += it->get_property(WEIGHT) * (it->energy());
        ov += it->get_property(WEIGHT);
      }
      tot_time += dt;
      app_log(1," -- {}  {}  {}",i,tot_time,(eav / ov).real());
      prop.Orthogonalize(wset);
    }
    for (int i = 0; i < 10; i++)
    {
      prop.Propagate(wset, Eshift, 2 * dt);
      wfn.Energy(wset);
      ComplexType eav = 0, ov = 0;
      for (auto it = wset.begin(); it != wset.end(); ++it)
      {
        eav += it->get_property(WEIGHT) * (it->energy());
        ov += it->get_property(WEIGHT);
      }
      tot_time += 2 * dt;
      app_log(1," -- {}  {}  {}",i,tot_time,(eav / ov).real());
      prop.Orthogonalize(wset);
    }
  } 
  else {

    dt = 0.099;
    //int ntau_test = 10;
    for(int i = 0; i < ntau-1; i++)
    {
      prop.Propagate(wset, Eshift, dt, i+1);
      wfn.Energy(wset, i+1);
      ComplexType eav = 0, ov = 0;
      for (auto it = wset.begin(); it != wset.end(); ++it)
      {
        eav += it->get_property(WEIGHT) * (it->energy());
        ov += it->get_property(WEIGHT);
      }
      tot_time += dt;
      app_log(1," -- {}  {}  {}",i,tot_time,(eav / ov).real());
      prop.Orthogonalize(wset);
    }
  }
std::cout<<" setup: " <<AFQMCTimer.elapsed(setup_timer) <<std::endl;
  if(mpi->comm.root()) AFQMCTimer.print_all();
}

TEST_CASE("propagator_factory: build", "[propagator_factory]")
{
  auto& mpi = utils::make_unit_test_mpi_context();

  using namespace utils;

  run_test_with_files([&]<auto MEM>(std::string hamil_file, std::string wfn_file, WALKER_TYPES) {
    propagator_factory_build<MEM>(mpi, hamil_file, wfn_file, true);
    propagator_factory_build<MEM>(mpi, hamil_file, wfn_file, false);
  }, UTEST_HAMIL, UTEST_WFN, TestFiles::RHF | TestFiles::UHF | TestFiles::GHF | TestFiles::NOMSD | TestFiles::FINITE_T | TestFiles::ALL_SYSTEMS);
}

// Direct (BP-independent) smoke for AFQMCBasePropagator::Propagate_free: a bare free-projection field step
// must apply B_T(Y) to the walkers (determinants move and stay finite) even on a STANDARD propagator built
// with free_projection = false (importance sampling / hybrid) -- pinning that the internal free_projection
// toggle works regardless of build mode. This is the propagator-level primitive StochasticWfn uses to draw
// walker-independent free-projection back-propagation references.
template<MEMORY_SPACE MEM>
void propagator_free_projection_step(std::shared_ptr<utils::mpi_context_t<boost::mpi3::communicator>> mpi,
                                     std::string hamil_file, std::string wfn_file)
{
  int NMO                    = read_nmo_from_hdf(hamil_file);
  auto [wfn_NMO, nup, ndown] = read_info_from_wfn(wfn_file, "any");
  utils::check(NMO == wfn_NMO, "Error: NMO != wfn_NMO.");
  WALKER_TYPES type = getWalkerType(wfn_file);
  if (type == COLLINEAR_FT or type == NONCOLLINEAR_FT)
    return; // finite-T uses a different field/step layout; not the target of this smoke.

  std::map<std::string, AFQMCInfo> InfoMap;
  InfoMap.insert(std::pair<std::string, AFQMCInfo>("info0", AFQMCInfo{"info0", NMO, nup, ndown, 0}));

  ptree ham_pt;
  ham_pt.put("name", "ham0");
  ham_pt.put("system", "info0");
  ham_pt.put("filename", hamil_file);
  HamiltonianFactory HamFac(InfoMap);
  HamFac.push("ham0", ham_pt);
  Hamiltonian& ham = HamFac.getHamiltonian(mpi, "ham0");

  const int nwalk = 11;
  std::shared_ptr<utils::RandomGenerator_t<HOST_MEMORY>> rng =
      std::make_shared<utils::RandomGenerator_t<HOST_MEMORY>>();
  std::shared_ptr<utils::RandomGenerator_t<MEM>> rng_dev =
      std::make_shared<utils::RandomGenerator_t<MEM>>(utils::make_rng<MEM>(777));

  ptree wlk_pt;
  wlk_pt.put("name", "wset0");
  wlk_pt.put("walker_type", walkerTypeToString(type));
  auto wset = make_WalkerSet<MEM>(mpi, wlk_pt, InfoMap["info0"], rng);

  ptree wfn_pt;
  wfn_pt.put("name", "wfn0");
  wfn_pt.put("system", "info0");
  wfn_pt.put("filename", wfn_file);
  WavefunctionFactory<MEM> WfnFac(InfoMap);
  WfnFac.push("wfn0", wfn_pt);
  auto& wfn = WfnFac.getWavefunction(mpi, "wfn0", type, &ham, nwalk);
  wset.resize(nwalk, WfnFac.getInitialGuess("wfn0"));

  // Standard propagator: free_projection defaults to false (importance sampling / hybrid).
  ptree prop_pt;
  prop_pt.put("name", "prop0");
  prop_pt.put("system", "info0");
  prop_pt.put("denseP2", true);
  PropagatorFactory<MEM> PropgFac(InfoMap);
  PropgFac.push("prop0", prop_pt);
  auto& prop = PropgFac.getPropagator(mpi, "prop0", wfn, rng_dev);

  // Owning copies: nda::to_host on HOST_MEMORY aliases the live walker storage, so snapshot into owning
  // arrays to actually capture the BEFORE state (else SM0 would track the post-step data).
  nda::array<ComplexType, 3> SM0 = nda::to_host(wset.SlaterMatrices(Alpha)); // snapshot before
  RealType dt = 0.01;
  prop.Propagate_free(wset, dt, 0); // bare free-projection step on an importance-sampling propagator
  nda::array<ComplexType, 3> SM1 = nda::to_host(wset.SlaterMatrices(Alpha)); // after

  double maxdiff = 0.0;
  bool finite    = true;
  for (long i = 0; i < SM1.size(); ++i)
  {
    ComplexType a = *(SM0.data() + i);
    ComplexType b = *(SM1.data() + i);
    if (not std::isfinite(b.real()) or not std::isfinite(b.imag()))
      finite = false;
    double d = std::abs(a - b);
    if (d > maxdiff)
      maxdiff = d;
  }
  CHECK(finite);
  CHECK(maxdiff > 1e-8); // fields were applied: the determinants moved off their input value
}

TEST_CASE("propagator_free_projection_step", "[propagator_factory]")
{
  auto& mpi = utils::make_unit_test_mpi_context();
  app_log(0, "AFQMCBasePropagator::Propagate_free applies bare fields on an importance-sampling propagator.");
  using namespace utils;
  run_test_with_files([&]<auto MEM>(std::string hamil_file, std::string wfn_file, WALKER_TYPES) {
    propagator_free_projection_step<MEM>(mpi, hamil_file, wfn_file);
  }, UTEST_HAMIL, UTEST_WFN, TestFiles::RHF | TestFiles::UHF | TestFiles::NOMSD | TestFiles::ALL_SYSTEMS);
}


} // namespace sfqmc
