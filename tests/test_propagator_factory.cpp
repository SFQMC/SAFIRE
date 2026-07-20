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
  std::shared_ptr<utils::RandomGenerator_t<MEM>> rng_dev = std::make_shared<utils::RandomGenerator_t<MEM>>(utils::make_rng<MEM>(777));

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
      std::make_shared<utils::RandomGenerator_t<MEM>>(utils::make_rng<MEM>(777));

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
  }, UTEST_HAMIL, UTEST_WFN, TestFiles::RHF | TestFiles::UHF | TestFiles::NOMSD | TestFiles::ALL_SYSTEMS);
}


namespace {
void mark_stochastic_wfn_input(ptree& pt) { pt.put("type", "stochasticwfn"); }
} // namespace

template<MEMORY_SPACE MEM>
void stochastic_propagator_step(std::shared_ptr<utils::mpi_context_t<boost::mpi3::communicator>> mpi,
                                std::string hamil_file, std::string wfn_file)
{
  if (getWavefunctionType(wfn_file) != NOMSD_WFN)
    return;
  if constexpr (MEM != HOST_MEMORY)
    return; // Un-rotated full-G kernels are CPU-only today.
  else
  {
    WALKER_TYPES type = afqmc::getWalkerType(wfn_file, "any");
    if (type != CLOSED)
      return;

    ptree ham_pt;
    ham_pt.put("name", "ham0");
    ham_pt.put("filename", hamil_file);
    HamiltonianFactory HamFac;
    HamFac.push("ham0", ham_pt);
    Hamiltonian& ham = HamFac.getHamiltonian(mpi, "ham0");

    const int nwalk = 11;
    std::shared_ptr<utils::RandomGenerator_t<>> rng = std::make_shared<utils::RandomGenerator_t<>>();
    std::shared_ptr<utils::RandomGenerator_t<MEM>> rng_dev =
        std::make_shared<utils::RandomGenerator_t<MEM>>(utils::make_rng<MEM>(13));
    ptree wlk_pt;
    wlk_pt.put("name", "wset0");
    wlk_pt.put("walker_type", walkerTypeToString(type));

    WavefunctionFactory<MEM> WfnFac{};
    ptree pt;
    pt.put("name", "wfn_stoch_prop");
    pt.put("filename", wfn_file);
    mark_stochastic_wfn_input(pt);
    pt.put("inner_nwalkers", 4);
    pt.put("inner_nsteps", 1);
    ptree inner_prop;
    inner_prop.put("timestep", 0.01);
    pt.put_child("inner_propagator", inner_prop);
    WfnFac.push("wfn_stoch_prop", pt);
    auto& wfn = WfnFac.getWavefunction(mpi, "wfn_stoch_prop", type, &ham, nwalk);
    WfnFac.maybe_initialize_stochastic_inner_walkers(wfn, "wfn_stoch_prop", type, wlk_pt);

    auto const& initial_guess = WfnFac.getInitialGuess("wfn_stoch_prop");
    auto wset = WalkerSet<MEM>(mpi, wlk_pt, rng, type, initial_guess, nwalk);

    // Prime overlaps/energies (anchor ensemble; begin_inner_step armed by the propagator each step)
    // and pick an energy shift so the hybrid weights stay well-scaled over the test steps.
    wfn.Log_Overlap(wset);
    wfn.Energy(wset);
    ComplexType eav(0.0), ow(0.0);
    for (auto it = wset.begin(); it != wset.end(); ++it)
    {
      eav += it->get_property(WEIGHT) * it->energy();
      ow += it->get_property(WEIGHT);
    }
    RealType Eshift = (std::abs(ow) > 1e-12) ? real(eav / ow) : RealType(0);

    // Build the OUTER propagator (default hybrid) bound to the stochastic trial.
    ptree prop_pt;
    prop_pt.put("name", "prop_stoch");
    PropagatorFactory<MEM> PropgFac;
    PropgFac.push("prop_stoch", prop_pt);
    auto& prop = PropgFac.getPropagator(mpi, "prop_stoch", wfn, rng_dev);

    RealType dt = 0.01;
    for (int step = 0; step < 3; ++step)
    {
      prop.Propagate(wset, Eshift, dt); // one full hot-path step; inner ensemble resampled once
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

TEST_CASE("stochastic_propagator_step", "[propagator_factory][stochastic_wfn]")
{
  auto& mpi = utils::make_unit_test_mpi_context();
  app_log(0, "StochasticWfn end-to-end outer propagator step on a dynamic trial.");
  using namespace utils;
  run_test_with_files([&]<auto MEM>(std::string hamil_file, std::string wfn_file, WALKER_TYPES, bool finiteT) {
    stochastic_propagator_step<MEM>(mpi, hamil_file, wfn_file);
  }, UTEST_HAMIL, UTEST_WFN, TestFiles::RHF | TestFiles::UHF | TestFiles::NOMSD | TestFiles::ALL_SYSTEMS);
}

// Walker-conditioned inner sampling. With inner_conditioning = true the inner field paths are
// importance-sampled conditioned on each outer walker phi_w (Eq. 23 of arXiv:2505.18519): the inner
// ensemble is grown to nwalk*inner_nwalkers (block w conditioned on phi_w via the custom force bias
// x_bar(phi_w) = sqrt(dt)*L^var.<phi_T|c+c|phi_w>/<phi_T|phi_w>, built by reusing the inner NOMSD's
// vbias on the OUTER wset). Run a real OUTER AFQMCBasePropagator over the dynamic conditioned trial and
// assert the walkers stay finite. The internal block-structure size checks (inner.size() == nwalk*P) in
// reduce_inner_cross_dm / Log_Overlap validate the nw*P resize. Leapfrog overlap cancellation is tested
// separately; this is a finiteness smoke, not NOMSD parity.
template<MEMORY_SPACE MEM>
void stochastic_conditioned_propagator_step(std::shared_ptr<utils::mpi_context_t<boost::mpi3::communicator>> mpi,
                                            std::string hamil_file, std::string wfn_file, bool leapfrog = false)
{
  if (getWavefunctionType(wfn_file) != NOMSD_WFN)
    return;
  if constexpr (MEM != HOST_MEMORY)
    return; // Walker-conditioned sampling is CPU-only today (full-G kernels CPU-only).
  else
  {
    WALKER_TYPES type = afqmc::getWalkerType(wfn_file, "any");
    if (type != CLOSED)
      return;

    ptree ham_pt;
    ham_pt.put("name", "ham0");
    ham_pt.put("filename", hamil_file);
    HamiltonianFactory HamFac;
    HamFac.push("ham0", ham_pt);
    Hamiltonian& ham = HamFac.getHamiltonian(mpi, "ham0");

    const int nwalk = 11;
    const int inner_nwalkers = 4;
    std::shared_ptr<utils::RandomGenerator_t<>> rng = std::make_shared<utils::RandomGenerator_t<>>();
    std::shared_ptr<utils::RandomGenerator_t<MEM>> rng_dev =
        std::make_shared<utils::RandomGenerator_t<MEM>>(utils::make_rng<MEM>(13));
    ptree wlk_pt;
    wlk_pt.put("name", "wset0");
    wlk_pt.put("walker_type", walkerTypeToString(type));

    WavefunctionFactory<MEM> WfnFac{};
    ptree pt;
    pt.put("name", "wfn_stoch_cond");
    pt.put("filename", wfn_file);
    mark_stochastic_wfn_input(pt);
    pt.put("inner_nwalkers", inner_nwalkers);
    pt.put("inner_nsteps", 1);
    pt.put("inner_conditioning", true);
    pt.put("inner_leapfrog", leapfrog);
    ptree inner_prop;
    inner_prop.put("timestep", 0.01);
    pt.put_child("inner_propagator", inner_prop);
    WfnFac.push("wfn_stoch_cond", pt);
    auto& wfn = WfnFac.getWavefunction(mpi, "wfn_stoch_cond", type, &ham, nwalk);
    WfnFac.maybe_initialize_stochastic_inner_walkers(wfn, "wfn_stoch_cond", type, wlk_pt);

    auto const& initial_guess = WfnFac.getInitialGuess("wfn_stoch_cond");
    auto wset = WalkerSet<MEM>(mpi, wlk_pt, rng, type, initial_guess, nwalk);

    // Prime overlaps/energies and pick an energy shift so the hybrid weights stay well-scaled.
    wfn.Log_Overlap(wset);
    wfn.Energy(wset);
    ComplexType eav(0.0), ow(0.0);
    for (auto it = wset.begin(); it != wset.end(); ++it)
    {
      eav += it->get_property(WEIGHT) * it->energy();
      ow += it->get_property(WEIGHT);
    }
    RealType Eshift = (std::abs(ow) > 1e-12) ? real(eav / ow) : RealType(0);

    // Build the OUTER propagator (default hybrid) bound to the conditioned stochastic trial.
    ptree prop_pt;
    prop_pt.put("name", "prop_stoch_cond");
    PropagatorFactory<MEM> PropgFac;
    PropgFac.push("prop_stoch_cond", prop_pt);
    auto& prop = PropgFac.getPropagator(mpi, "prop_stoch_cond", wfn, rng_dev);

    RealType dt = 0.01;
    for (int step = 0; step < 3; ++step)
    {
      prop.Propagate(wset, Eshift, dt); // hot-path step; inner ensemble resampled (nw*P, conditioned)
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

TEST_CASE("stochastic_conditioned_propagator_step", "[propagator_factory][stochastic_wfn]")
{
  auto& mpi = utils::make_unit_test_mpi_context();
  app_log(0, "StochasticWfn walker-conditioned inner sampling over a real outer propagator.");
  using namespace utils;
  run_test_with_files([&]<auto MEM>(std::string hamil_file, std::string wfn_file, WALKER_TYPES, bool finiteT) {
    stochastic_conditioned_propagator_step<MEM>(mpi, hamil_file, wfn_file);
  }, UTEST_HAMIL, UTEST_WFN, TestFiles::RHF | TestFiles::UHF | TestFiles::NOMSD | TestFiles::ALL_SYSTEMS);
}

// Propagate-then-resample leapfrog (inner_conditioning + inner_leapfrog). At each outer step,
// outer step, begin_inner_step(wset) resamples the inner ensemble conditioned on the OLD walker and
// stores the importance-reweighted old overlap (Sum_p S_p) against it; the post-propagation Log_Overlap
// scores the NEW walker against the SAME ensemble, so the hybrid ratio new/old reproduces Eq. 25 of
// arXiv:2505.18519 exactly and N(phi) cancels. Drives a real OUTER hybrid AFQMCBasePropagator and
// asserts the walkers stay finite over several steps (finiteness smoke; energy-vs-analytic-AFQMC and
// variance reduction vs free projection are the research-level validation). CLOSED+CPU; reuses the conditioned
// driver above with leapfrog = true.
template<MEMORY_SPACE MEM>
void stochastic_leapfrog_propagator_step(std::shared_ptr<utils::mpi_context_t<boost::mpi3::communicator>> mpi,
                                         std::string hamil_file, std::string wfn_file)
{
  stochastic_conditioned_propagator_step<MEM>(mpi, hamil_file, wfn_file, /*leapfrog=*/true);
}

TEST_CASE("stochastic_leapfrog_propagator_step", "[propagator_factory][stochastic_wfn]")
{
  auto& mpi = utils::make_unit_test_mpi_context();
  app_log(0, "StochasticWfn propagate-then-resample leapfrog over a real outer propagator.");
  using namespace utils;
  run_test_with_files([&]<auto MEM>(std::string hamil_file, std::string wfn_file, WALKER_TYPES, bool finiteT) {
    stochastic_leapfrog_propagator_step<MEM>(mpi, hamil_file, wfn_file);
  }, UTEST_HAMIL, UTEST_WFN, TestFiles::RHF | TestFiles::UHF | TestFiles::NOMSD | TestFiles::ALL_SYSTEMS);
}

} // namespace sfqmc
