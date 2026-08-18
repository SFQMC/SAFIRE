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
#include "AFQMC/parameters.hpp"
#include "AFQMC/parameter_defaults.hpp"
#include "utilities/Random.hpp"
#include "utilities/Timer.hpp"
#include "test_common.hpp"
#include "test_stochastic_common.hpp"
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

  HamiltonianFactory HamFac;
  HamFac.push("ham0", HamiltonianParameters{.name = "ham0", .filename = hamil_file, .shift_1body = true});
  Hamiltonian& ham = HamFac.getHamiltonian(mpi, "ham0");

  int nwalk = 11; 
  std::shared_ptr<utils::RandomGenerator_t<HOST_MEMORY>> rng = std::make_shared<utils::RandomGenerator_t<HOST_MEMORY>>();
  std::shared_ptr<utils::RandomGenerator_t<MEM>> rng_dev = std::make_shared<utils::RandomGenerator_t<MEM>>(777);

  const WalkerSetParameters wlk_params{.name = "wset0", .walker_type = type};

  WavefunctionFactory<MEM> WfnFac{};
  WfnFac.push("wfn0", WavefunctionParameters{.name = "wfn0", .filename = wfn_file, .dense_trial = dense_trial});
  auto& wfn = WfnFac.getWavefunction(mpi, "wfn0", type, finiteT, &ham, nwalk);

  auto wset = [&]() {
    if(!finiteT)
    {
      auto const& initial_guess = WfnFac.getInitialGuess("wfn0");
      REQUIRE(int(initial_guess.size()) == nspin);
      REQUIRE(initial_guess[0].shape() == std::array<long,2>{npol*NMO,nup});
      return WalkerSet<MEM>(mpi, wlk_params, rng, type, initial_guess, nwalk);
    }
    else
    {
      auto initial_guess_ft = WfnFac.getInitialGuess_ft("wfn0");
      REQUIRE(initial_guess_ft.shape() == std::array<long,4>{3,nspin,npol*NMO,NMO});
      return WalkerSet<MEM>(mpi, wlk_params, rng, type, initial_guess_ft, nwalk);
    }
  }();

  PropagatorFactory<MEM> PropgFac;
  PropagatorParameters prop_params{.name = "prop0", .denseP2 = true};
  apply_defaults(prop_params, ham.getHamType());
  PropgFac.push("prop0", prop_params);
  auto& prop = PropgFac.getPropagator(mpi, "prop0", wfn, rng_dev);

  std::cout << setprecision(8);
  wfn.Energy(wset);
  {
    ComplexType eav = 0, ov = 0;
    for(int iw = 0; iw < wset.size(); ++iw)
    {
      auto w = wset[iw];
      eav += w.get_property(WEIGHT) * (w.energy());
      ov += w.get_property(WEIGHT);
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
      for(int iw = 0; iw < wset.size(); ++iw)
      {
        auto w = wset[iw];
        eav += w.get_property(WEIGHT) * (w.energy());
        ov += w.get_property(WEIGHT);
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
      for(int iw = 0; iw < wset.size(); ++iw)
      {
        auto w = wset[iw];
        eav += w.get_property(WEIGHT) * (w.energy());
        ov += w.get_property(WEIGHT);
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
      for(int iw = 0; iw < wset.size(); ++iw)
      {
        auto w = wset[iw];
        eav += w.get_property(WEIGHT) * (w.energy());
        ov += w.get_property(WEIGHT);
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

  HamiltonianFactory HamFac;
  HamFac.push("ham0", HamiltonianParameters{.name = "ham0", .filename = hamil_file});
  Hamiltonian& ham = HamFac.getHamiltonian(mpi, "ham0");

  const int nwalk = 11;
  std::shared_ptr<utils::RandomGenerator_t<HOST_MEMORY>> rng =
      std::make_shared<utils::RandomGenerator_t<HOST_MEMORY>>();
  std::shared_ptr<utils::RandomGenerator_t<MEM>> rng_dev =
      std::make_shared<utils::RandomGenerator_t<MEM>>(utils::SeedType(777));

  WalkerSetParameters wlk_pt{.name = "wset0", .walker_type = type};

  WavefunctionFactory<MEM> WfnFac{};
  auto wfn0_pt = WavefunctionParameters{.name = "wfn0", .filename = wfn_file};
  utils::apply_wfn_defaults(wfn0_pt, ham);
  WfnFac.push("wfn0", wfn0_pt);
  auto& wfn = WfnFac.getWavefunction(mpi, "wfn0", type, false, &ham, nwalk);
  auto const& initial_guess = WfnFac.getInitialGuess("wfn0");
  auto wset = WalkerSet<MEM>(mpi, wlk_pt, rng, type, initial_guess, nwalk);

  // Standard propagator: free_projection defaults to false (importance sampling / hybrid).
  PropagatorFactory<MEM> PropgFac;
  PropagatorParameters fp_prop_params{.name = "prop0", .denseP2 = true};
  utils::apply_prop_defaults(fp_prop_params, ham);
  PropgFac.push("prop0", fp_prop_params);
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

TEST_CASE("propagator_factory: free projection step", "[propagator_factory]")
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
                                           StochasticSamplingTarget inner_sampling_target)
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
    const std::string tag    = "stoch_" + nlohmann::json(inner_sampling_target).get<std::string>();

    HamiltonianFactory HamFac;
    HamFac.push("ham0", HamiltonianParameters{.name = "ham0", .filename = hamil_file});
    Hamiltonian& ham = HamFac.getHamiltonian(mpi, "ham0");

    std::shared_ptr<utils::RandomGenerator_t<>> rng = std::make_shared<utils::RandomGenerator_t<>>();
    std::shared_ptr<utils::RandomGenerator_t<MEM>> rng_dev =
        std::make_shared<utils::RandomGenerator_t<MEM>>(utils::SeedType(7));
    WalkerSetParameters wlk_pt{.name = "wset0", .walker_type = type};

    WavefunctionFactory<MEM> WfnFac{};
    WavefunctionParameters pt{.name = tag, .filename = wfn_file, .inner_n_samples = inner_n_samples,
                              .inner_nsteps = 1, .inner_sampling_target = inner_sampling_target,
                              .inner_propagator = PropagatorParameters{.timestep = 0.01}};
    utils::mark_stochastic_wfn_input(pt);
    utils::apply_wfn_defaults(pt, ham);
    WfnFac.push(tag, pt);
    auto& wfn = WfnFac.getWavefunction(mpi, tag, type, false, &ham, nwalk);
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
      for (int iw = 0; iw < wset.size(); ++iw)
        e_sum += ComplexType(wset[iw].energy());
      Eshift = std::real(e_sum) / RealType(nwalk);
    }

    // Build the OUTER propagator (default hybrid) bound to the stochastic trial.
    PropagatorFactory<MEM> PropgFac;
    PropagatorParameters st_prop_params{.name = "prop_" + tag};
    utils::apply_prop_defaults(st_prop_params, ham);
    PropgFac.push("prop_" + tag, st_prop_params);
    auto& prop = PropgFac.getPropagator(mpi, "prop_" + tag, wfn, rng_dev);

    RealType dt = 0.01;
    for (int step = 0; step < 3; ++step)
    {
      prop.Propagate(wset, Eshift, dt); // one full hot-path step, inner ensemble resampled per the mode
      prop.Orthogonalize(wset);
      wfn.Energy(wset);
      for (int iw = 0; iw < wset.size(); ++iw)
      {
        auto w = wset[iw];
        const ComplexType e(w.energy());
        REQUIRE(std::isfinite(std::real(ComplexType(w.get_property(WEIGHT)))));
        REQUIRE(std::isfinite(std::real(e)));
        REQUIRE(std::isfinite(std::imag(e)));
        REQUIRE(std::isfinite(std::real(ComplexType(w.get_property(OVLP)))));
      }
    }
  }
}

TEST_CASE("propagator_factory: stochastic free trial survives", "[propagator_factory][stochastic_wfn]")
{
  auto& mpi = utils::make_unit_test_mpi_context();
  app_log(0, "StochasticWfn free-projection inner sampling over a real outer propagator.");
  using namespace utils;
  run_test_with_files([&]<auto MEM>(std::string hamil_file, std::string wfn_file, WALKER_TYPES, bool finiteT) {
    stochastic_trial_survives_propagation<MEM>(mpi, hamil_file, wfn_file, StochasticSamplingTarget::Gaussian);
  }, UTEST_HAMIL, UTEST_WFN, TestFiles::DYNAMIC_INNER);
}

TEST_CASE("propagator_factory: stochastic conditioned trial survives", "[propagator_factory][stochastic_wfn]")
{
  auto& mpi = utils::make_unit_test_mpi_context();
  // The production sampler: walker-conditioned persistent field chains plus the leapfrog reweight, so the
  // step's overlap ratio new/old is Eq. 25 of arXiv:2505.18519 and N(phi) cancels.
  app_log(0, "StochasticWfn walker-conditioned inner sampling over a real outer propagator.");
  using namespace utils;
  run_test_with_files([&]<auto MEM>(std::string hamil_file, std::string wfn_file, WALKER_TYPES, bool finiteT) {
    stochastic_trial_survives_propagation<MEM>(mpi, hamil_file, wfn_file, StochasticSamplingTarget::WalkerOverlap);
  }, UTEST_HAMIL, UTEST_WFN, TestFiles::DYNAMIC_INNER);
}

} // namespace sfqmc
