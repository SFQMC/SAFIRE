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
             std::string hamil_file, std::string wfn_file, bool dense_trial, bool finiteT)
{
  using nda::range;

  int NMO = read_nmo_from_hdf(hamil_file);
  auto[wfn_NMO,nup, ndown] = read_info_from_wfn(wfn_file,"any");
  utils::check(NMO == wfn_NMO, "Error: NMO != wfn_NMO.");
  WALKER_TYPES type         = getWalkerType(wfn_file);
  int nspin                 = type == COLLINEAR ? 2 : 1;
  int npol                  = type == NONCOLLINEAR ? 2 : 1;
  // finite-T imaginary-time slice count (the wfn "nup" field for a finite-T guess)
  int ntau                  = nup;

  ptree ham_pt;
  ham_pt.put("name","ham0");
  ham_pt.put("filename",hamil_file);
  ham_pt.put("shift_1body",true);
  //ham_pt.put("shift_1body",false);

  HamiltonianFactory HamFac;
  HamFac.push("ham0", ham_pt);
  Hamiltonian& ham = HamFac.getHamiltonian(mpi, "ham0");

  int nwalk = 11; 
  std::shared_ptr<utils::RandomGenerator_t<HOST_MEMORY>> rng = std::make_shared<utils::RandomGenerator_t<HOST_MEMORY>>();
  std::shared_ptr<utils::RandomGenerator_t<MEM>> rng_dev = std::make_shared<utils::RandomGenerator_t<MEM>>(777);

  ptree wlk_pt;
  wlk_pt.put("name","wset0");
  wlk_pt.put("walker_type", walkerTypeToString(type));

  ptree wfn_pt;
  wfn_pt.put("name","wfn0");
  wfn_pt.put("filename",wfn_file);
  wfn_pt.put("dense_trial",dense_trial);

  WavefunctionFactory<MEM> WfnFac{};
  WfnFac.push("wfn0", wfn_pt);
  auto& wfn = WfnFac.getWavefunction(mpi, "wfn0", type, finiteT, &ham, nwalk);

  auto wset = [&]() {
    if(!finiteT)
    {
      auto const& initial_guess = WfnFac.getInitialGuess("wfn0");
      REQUIRE(int(initial_guess.size()) == nspin);
      REQUIRE(initial_guess[0].shape() == std::array<long,2>{npol*NMO,nup});
      return WalkerSet<MEM>(mpi, wlk_pt, rng, type, initial_guess, nwalk);
    }
    else
    {
      auto initial_guess_ft = WfnFac.getInitialGuess_ft("wfn0");
      REQUIRE(initial_guess_ft.shape() == std::array<long,4>{3,nspin,npol*NMO,NMO});
      return WalkerSet<MEM>(mpi, wlk_pt, rng, type, initial_guess_ft, nwalk);
    }
  }();

  ptree prop_pt;
  prop_pt.put("name","prop0");
  prop_pt.put("denseP2",true);

  PropagatorFactory<MEM> PropgFac;
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
  if(!finiteT){
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

  run_test_with_files([&]<auto MEM>(std::string hamil_file, std::string wfn_file, WALKER_TYPES, bool finiteT) {
    propagator_factory_build<MEM>(mpi, hamil_file, wfn_file, true, finiteT);
    propagator_factory_build<MEM>(mpi, hamil_file, wfn_file, false, finiteT);
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
  // finite-T uses a different field/step layout; not the target of this smoke.

  ptree ham_pt;
  ham_pt.put("name", "ham0");
  ham_pt.put("filename", hamil_file);
  HamiltonianFactory HamFac;
  HamFac.push("ham0", ham_pt);
  Hamiltonian& ham = HamFac.getHamiltonian(mpi, "ham0");

  const int nwalk = 11;
  std::shared_ptr<utils::RandomGenerator_t<HOST_MEMORY>> rng =
      std::make_shared<utils::RandomGenerator_t<HOST_MEMORY>>();
  std::shared_ptr<utils::RandomGenerator_t<MEM>> rng_dev =
      std::make_shared<utils::RandomGenerator_t<MEM>>(utils::SeedType(777));

  ptree wlk_pt;
  wlk_pt.put("name", "wset0");
  wlk_pt.put("walker_type", walkerTypeToString(type));

  ptree wfn_pt;
  wfn_pt.put("name", "wfn0");
  wfn_pt.put("filename", wfn_file);
  WavefunctionFactory<MEM> WfnFac{};
  WfnFac.push("wfn0", wfn_pt);
  auto& wfn = WfnFac.getWavefunction(mpi, "wfn0", type, &ham, nwalk);
  auto const& initial_guess = WfnFac.getInitialGuess("wfn0");
  auto wset = WalkerSet<MEM>(mpi, wlk_pt, rng, type, initial_guess, nwalk);

  // Standard propagator: free_projection defaults to false (importance sampling / hybrid).
  ptree prop_pt;
  prop_pt.put("name", "prop0");
  prop_pt.put("denseP2", true);
  PropagatorFactory<MEM> PropgFac;
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
  run_test_with_files([&]<auto MEM>(std::string hamil_file, std::string wfn_file, WALKER_TYPES, bool finiteT) {
    propagator_free_projection_step<MEM>(mpi, hamil_file, wfn_file);
      // No LATTICES: assemble_X aborts with "Finish FP DiscretePropagator" when free_projection is set and
    // the Hamiltonian carries Discrete{Charge,Spin}Propagator field types, i.e. free projection is simply
    // not implemented for the Hubbard-style discrete fields. Pre-existing and unrelated to the stochastic
    // trial; the fixtures were just asking for a code path that does not exist.
  }, UTEST_HAMIL, UTEST_WFN, TestFiles::RHF | TestFiles::UHF | TestFiles::NOMSD | TestFiles::MOLECULES | TestFiles::SOLIDS);
}


namespace {
void mark_stochastic_wfn_input(ptree& pt) { pt.put("type", "stochasticwfn"); }
} // namespace

// Does a stochastic trial survive real outer propagation?
//
// Drives an actual hybrid AFQMCBasePropagator over a DYNAMIC stochastic trial for several steps and
// asserts every walker's weight, energy and overlap stay finite. It is a SURVIVAL smoke, not a parity or
// accuracy check: energy-vs-analytic-AFQMC and variance-vs-free-projection are research-level validation
// done elsewhere. What it catches is the hot path falling over -- NaN weights from a mis-scaled hybrid
// ratio, a dead inner ensemble, a resample that leaves the pool inconsistent with the walkers.
//
// Parameterised on the sampling mode, because that is the ONLY thing that differed between the three
// tests this replaces. They were `stochastic_propagator_step`, `stochastic_conditioned_propagator_step`
// and `stochastic_leapfrog_propagator_step`: two near-identical 80-line bodies plus an alias. The old
// names also mis-described their subject -- "conditioned propagator" reads as a property of the
// propagator, when it is the INNER SAMPLING that is conditioned, and "leapfrog" named an implementation
// detail that stopped being separately selectable when inner_sampling_target absorbed it. One body, one mode
// argument, and names that say what is varied and what is asserted.
template<MEMORY_SPACE MEM>
void stochastic_trial_survives_propagation(std::shared_ptr<utils::mpi_context_t<boost::mpi3::communicator>> mpi,
                                           std::string hamil_file, std::string wfn_file,
                                           std::string const& inner_sampling_target)
{
  if (getWavefunctionType(wfn_file) != NOMSD_WFN)
    return;
  if constexpr (MEM != HOST_MEMORY)
    return; // the un-rotated full-G kernels the dynamic path needs are CPU-only today
  else
  {
    WALKER_TYPES type = afqmc::getWalkerType(wfn_file, "any");
    if (not utils::dynamic_inner_supports(type))
      return; // dynamic inner ensemble: CLOSED/COLLINEAR only

    const int nwalk          = 5;
    const int inner_n_samples = 4;
    const std::string tag    = "stoch_" + inner_sampling_target;

    ptree ham_pt;
    ham_pt.put("name", "ham0");
    ham_pt.put("filename", hamil_file);
    HamiltonianFactory HamFac;
    HamFac.push("ham0", ham_pt);
    Hamiltonian& ham = HamFac.getHamiltonian(mpi, "ham0");

    std::shared_ptr<utils::RandomGenerator_t<>> rng = std::make_shared<utils::RandomGenerator_t<>>();
    std::shared_ptr<utils::RandomGenerator_t<MEM>> rng_dev =
        std::make_shared<utils::RandomGenerator_t<MEM>>(utils::SeedType(7));
    ptree wlk_pt;
    wlk_pt.put("name", "wset0");
    wlk_pt.put("walker_type", walkerTypeToString(type));

    WavefunctionFactory<MEM> WfnFac{};
    ptree pt;
    pt.put("name", tag);
    pt.put("filename", wfn_file);
    mark_stochastic_wfn_input(pt);
    pt.put("inner_n_samples", inner_n_samples);
    pt.put("inner_nsteps", 1);
    pt.put("inner_sampling_target", inner_sampling_target);
    ptree inner_prop;
    inner_prop.put("timestep", 0.01);
    pt.put_child("inner_propagator", inner_prop);
    WfnFac.push(tag, pt);
    auto& wfn = WfnFac.getWavefunction(mpi, tag, type, &ham, nwalk);
    WfnFac.maybe_initialize_stochastic_inner_walkers(wfn, tag, type, wlk_pt);

    auto const& initial_guess = WfnFac.getInitialGuess(tag);
    auto wset = WalkerSet<MEM>(mpi, wlk_pt, rng, type, initial_guess, nwalk);

    // Prime overlaps/energies and pick an energy shift so the hybrid weights stay well-scaled over the
    // test steps -- an unscaled shift makes the weights blow up and the finiteness checks would then be
    // reporting the test's own setup rather than the code under test.
    wfn.Energy(wset);
    RealType Eshift(0.0);
    {
      ComplexType e_sum(0.0);
      for (auto it = wset.begin(); it != wset.end(); ++it)
        e_sum += it->energy();
      Eshift = real(e_sum) / RealType(nwalk);
    }

    // Build the OUTER propagator (default hybrid) bound to the stochastic trial.
    PropagatorFactory<MEM> PropgFac;
    ptree prop_pt;
    prop_pt.put("name", "prop_" + tag);
    prop_pt.put("system", "system0");
    PropgFac.push("prop_" + tag, prop_pt);
    auto& prop = PropgFac.getPropagator(mpi, "prop_" + tag, wfn, rng_dev);

    RealType dt = 0.01;
    for (int step = 0; step < 3; ++step)
    {
      prop.Propagate(wset, Eshift, dt); // one full hot-path step, inner ensemble resampled per the mode
      prop.Orthogonalize(wset);
      wfn.Energy(wset);
      for (auto it = wset.begin(); it != wset.end(); ++it)
      {
        REQUIRE(std::isfinite(real(it->get_property(WEIGHT))));
        REQUIRE(std::isfinite(real(it->energy())));
        REQUIRE(std::isfinite(imag(it->energy())));
        REQUIRE(std::isfinite(real(it->get_property(OVLP))));
      }
    }
  }
}

TEST_CASE("stochastic_free_trial_survives_propagation", "[propagator_factory][stochastic_wfn]")
{
  auto& mpi = utils::make_unit_test_mpi_context();
  app_log(0, "StochasticWfn free-projection inner sampling over a real outer propagator.");
  using namespace utils;
  run_test_with_files([&]<auto MEM>(std::string hamil_file, std::string wfn_file, WALKER_TYPES, bool finiteT) {
    stochastic_trial_survives_propagation<MEM>(mpi, hamil_file, wfn_file, "gaussian");
  }, UTEST_HAMIL, UTEST_WFN, TestFiles::DYNAMIC_INNER);
}

TEST_CASE("stochastic_conditioned_trial_survives_propagation", "[propagator_factory][stochastic_wfn]")
{
  auto& mpi = utils::make_unit_test_mpi_context();
  // The production sampler: walker-conditioned persistent field chains plus the leapfrog reweight, so the
  // step's overlap ratio new/old is Eq. 25 of arXiv:2505.18519 and N(phi) cancels.
  app_log(0, "StochasticWfn walker-conditioned inner sampling over a real outer propagator.");
  using namespace utils;
  run_test_with_files([&]<auto MEM>(std::string hamil_file, std::string wfn_file, WALKER_TYPES, bool finiteT) {
    stochastic_trial_survives_propagation<MEM>(mpi, hamil_file, wfn_file, "walker_overlap");
  }, UTEST_HAMIL, UTEST_WFN, TestFiles::DYNAMIC_INNER);
}

} // namespace sfqmc
