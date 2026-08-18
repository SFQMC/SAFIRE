/*
 * This file is distributed under the Apache License, Version 2.0 License.
 * See LICENSE file in top directory for details.
 *
 * Copyright (c) 2021-2025 The Simons Foundation, Inc.
 *
 * You may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *      http://www.apache.org/licenses/LICENSE-2.0
 *
 */

#undef NDEBUG

#include "catch2/catch_test_macros.hpp"

#include "config.h"
#include "AFQMC/config.h"
#include "IO/AppAbort.hpp"

#include "AFQMC/parameters.hpp"
#include "AFQMC/parameter_defaults.hpp"
#include "utilities/Random.hpp"
#include "IO/app_loggers.h"
#include "test_common.hpp"
#include "test_stochastic_common.hpp"

#include "nda/nda.hpp"
#include "nda/tensor.hpp"
#include "nda/h5.hpp"

#include "AFQMC/Hamiltonians/HamiltonianFactory.h"
#include "AFQMC/Hamiltonians/Hamiltonian.hpp"
#include "AFQMC/Wavefunctions/WavefunctionFactory.h"
#include "AFQMC/Walkers/WalkerSetFactory.hpp"
#include "AFQMC/Walkers/WalkerSet.hpp"
#include "AFQMC/Estimators/EstimatorBase.h"
#include "AFQMC/Estimators/EstimatorHandler.h"
#include "AFQMC/Propagators/PropagatorFactory.h"
#include "AFQMC/Estimators/BackPropagatedEstimator.hpp"
#include "test_utils.hpp"
#include "AFQMC/Utilities/AFQMCTimer.h"
#include "AFQMC/Utilities/readWfn.h"

#include <cstdio>
#include <format>


extern std::string UTEST_HAMIL, UTEST_WFN;

namespace sfqmc
{
using namespace afqmc;

template<MEMORY_SPACE MEM>
void estimator_handler_measure_schedule(std::shared_ptr<utils::mpi_context_t<boost::mpi3::communicator>> mpi,
             std::string hamil_file, std::string wfn_file)
{
  using nda::range;
  utils::check(utils::file_exists(hamil_file),
               " Hamiltonian file not found: {}. \n Run unit test with --hamil /path/to/hamil.h5 ", hamil_file);
  utils::check(utils::file_exists(wfn_file),
               " Wavefunction file not found: {}. \n Run unit test with --wfn /path/to/wfn.h5 ", wfn_file);

  int population_control_interval = 10;
  [[maybe_unused]] auto[NMO,nup, ndown] = read_info_from_wfn(wfn_file, "any");
  utils::check(NMO == read_nmo_from_hdf(hamil_file), "NMO differ between hamil and wfn files.");

  std::shared_ptr<utils::RandomGenerator_t<>> rng = std::make_shared<utils::RandomGenerator_t<>>();
  std::shared_ptr<utils::RandomGenerator_t<MEM>> rng_dev = std::make_shared<utils::RandomGenerator_t<MEM>>(777);

  HamiltonianFactory HamFac;
  HamFac.push("ham0", HamiltonianParameters{.name = "ham0", .filename = hamil_file});
  Hamiltonian& ham = HamFac.getHamiltonian(mpi, "ham0");

  WALKER_TYPES type = afqmc::getWalkerType(wfn_file);
  const WalkerSetParameters wlk_params{.name = "wset0", .walker_type = type};

  int nspin            = (type == COLLINEAR) ? 2 : 1;
  int npol             = (type == NONCOLLINEAR) ? 2 : 1;

  int nwalk = 11;
  WavefunctionFactory<MEM> WfnFac{};
  WfnFac.push("wfn0", WavefunctionParameters{.name = "wfn0", .filename = wfn_file, .dense_trial = true});
  auto& wfn = WfnFac.getWavefunction(mpi, "wfn0", type, false, &ham, nwalk);

  PropagatorFactory<MEM> PropgFac;
  PropagatorParameters prop_params{.name = "prop0"};
  apply_defaults(prop_params, ham.getHamType());
  PropgFac.push("prop0", prop_params);
  auto& prop = PropgFac.getPropagator(mpi, "prop0", wfn, rng_dev);

  auto const& initial_guess = WfnFac.getInitialGuess("wfn0");
  REQUIRE(int(initial_guess.size()) == nspin);
  REQUIRE(initial_guess[0].shape() == std::array<long,2>{npol*NMO,nup});
  auto wset = WalkerSet<MEM>(mpi, wlk_params, rng, type, initial_guess, nwalk);
  
  // number of steps to propagate
  int nStep = 200;

  // define / run test cases
  struct test_case {
    std::string name;
    int meas1; // global measure_interval_multiplier, used by BasicEstimator and EnergyEstimator
    int meas2; // measure_interval_multiplier of the back propagation estimator
  };
  const std::vector<test_case> cases = {
    {"case1", 5, 20},
    {"case2", 20, 10},
    {"case3", 5, 7},
    {"case4", 11, 7},
    {"case5", 1, 7}, // test that we default properly
  };

  for (auto test : cases)
  {
    ExecuteParameters exec{
        .wavefunction = std::string{"wfn0"},
        .hamiltonian = std::string{"ham0"},
        .estimator = {
            EstimatorParameters{.name = EstimatorType::energy, .overwrite = true},
            EstimatorParameters{.name = EstimatorType::back_propagation,
                                .equil_multiplier = 0,
                                .bp_walker_ortho_interval = 1,
                                .measure_interval_multiplier = std::vector<int>{test.meas2},
                                .onerdm = OneRDMParameters{}},
        },
        .population_control_interval = population_control_interval,
        // the global multiplier used by BasicEstimator and EnergyEstimator
        .measure_interval_multiplier = test.meas1,
    };
    apply_defaults(exec);

    int measure_interval{};
    {
      int nPopulation = 1;
      float dt = 0.01f;
      float total_time = 0.0f;
      double E1 = 0.0;
      EstimatorHandler<MEM> estim0(mpi, "test_est_handler",
        exec, wset, WfnFac, wfn, prop,
                          HamFac, dt);
    
      // set measurement intervals
      measure_interval = estim0.get_max_common_interval();
      std::cout << "Querying estimator handler with interval " << measure_interval << " (commensurate with all measurement intervals)" << std::endl;

      /* Fake Driver Block */
      std::vector<ComplexType> dummyData;

      for (int iStep = 0; iStep < nStep; ++iStep)
      {
        prop.Propagate(wset, E1, dt);
        total_time += dt;

        if (total_time < 1.0 || (iStep + 1) % nPopulation == 0 || iStep == 0)
        {
          AFQMCTimer.start(popcont_timer);
          wset.processWalkerData(dummyData);
          // Mirror the production driver's PAIR: store -> popControl -> permute. The post-pop hook
          // fails closed without the store, because the magnitudes would otherwise stay indexed by
          // a slot layout popControl has already invalidated.
          wfn.store_inner_blocks_before_pop(wset);
          wset.popControl(); // make this a call to actual pop control
          wfn.permute_inner_blocks_after_pop(wset); // mirror the production driver (no-op here)
          AFQMCTimer.stop(popcont_timer);
          estim0.accumulate_step(total_time, wset, dummyData);
        }

        if ((iStep + 1) % measure_interval == 0 )
        {
          estim0.accumulate_block(total_time, wset);
          estim0.print(iStep + 1, total_time, E1, wset);
        }
      }
    
    }
    // Energy estimator uses meas1 as the global measure_interval_multiplier
    int energy_interval = test.meas1 * population_control_interval;
    int expected_measurements = nStep / energy_interval;
    // read results from "test_est_handler.scalar.dat"
    std::string filename = "test_est_handler.scalar.dat";
    std::ifstream in(filename.c_str());
    utils::check(in.good()," Error opening file in test_est_handler.scalar.dat.");
    int line_count = -1; // first line is header
    std::string line;
    while (std::getline(in, line))
    {
      line_count++;
      std::cout << line << std::endl;
    }
    app_log(1, "\n[TESTS] Running test case: {} \n", test.name);
    CHECK(line_count == expected_measurements);
    in.close();

    mpi->comm.barrier();
    if (mpi->comm.root()) remove(filename.c_str());
    mpi->comm.barrier();
  }
}

TEST_CASE("estimator_handler: measure schedule", "[estimator_handler]")
{
  auto& mpi = utils::make_unit_test_mpi_context();

  std::string hamil = utils::unit_test_base() + "square_4x4_hubbard_nup5_ndn5/ham_collinear.h5";
  std::string wfn   = utils::unit_test_base() + "square_4x4_hubbard_nup5_ndn5/uhf_U0.1_wfn_nup5_ndn5.h5";
  if (UTEST_HAMIL!="" and UTEST_WFN!="") {
    hamil = UTEST_HAMIL;
    wfn = UTEST_WFN;
  }
    
  estimator_handler_measure_schedule<HOST_MEMORY>(mpi, hamil, wfn);
#if defined(ENABLE_DEVICE)
  estimator_handler_measure_schedule<DEVICE_MEMORY>(mpi, hamil, wfn);
#endif
}

// Integration smoke: BackPropagatedEstimator through EstimatorHandler on a DYNAMIC stochastic trial.
// Asserts a finite accumulated 1-RDM. The static BP path is covered by `driver_factory: stochastic bp driver`
// (same estimator block via executeDriver); this keeps the dynamic leg, which has no driver counterpart.
// Free-projection (unconditioned) is not exercised: hybrid weights NaN under pop control.
template<MEMORY_SPACE MEM>
void stochastic_back_propagation_estimator_smoke(
    std::shared_ptr<utils::mpi_context_t<boost::mpi3::communicator>> mpi, std::string hamil_file,
    std::string wfn_file)
{
  if (getWavefunctionType(wfn_file) != NOMSD_WFN)
    return;
  if constexpr (MEM != HOST_MEMORY)
    return;
  else
  {
    WALKER_TYPES type = afqmc::getWalkerType(wfn_file, "any");
    // CLOSED only. Narrower than the engine allows, and narrower than DYNAMIC_INNER supplies: these are
    // whole-run integration smokes, and the BH CLOSED fixture is the one whose forward walk is known
    // stable over a full population-control schedule. (dynamic_inner_supports() would admit COLLINEAR and
    // the next line would discard it, which is what this used to do.)
    if (type != CLOSED)
      return;

    HamiltonianFactory HamFac;
    HamFac.push("ham0", HamiltonianParameters{.name = "ham0", .filename = hamil_file});
    Hamiltonian& ham = HamFac.getHamiltonian(mpi, "ham0");

    const std::string title               = "stoch_bp_dyn_smoke";
    const int nwalk                       = 11;
    const int population_control_interval = DEFAULT_POPULATION_CONTROL_INTERVAL;
    const int bp_measure_multiplier       = 2;
    const int nStep                       = bp_measure_multiplier * population_control_interval * 2;
    const float dt                        = 0.01f;

    if (mpi->comm.root())
    {
      std::remove((title + ".stat.h5").c_str());
      std::remove((title + ".scalar.dat").c_str());
    }
    mpi->comm.barrier();

    std::shared_ptr<utils::RandomGenerator_t<>> rng = std::make_shared<utils::RandomGenerator_t<>>();
    // Construct in place from the seed. Both generators take a SeedType, and CurandRandomGenerator owns a
    // raw handle (copy deleted, move hand-written), so building a temporary to hand to make_shared is
    // what the post-curand ownership API removed -- this call site was missed when the others moved.
    std::shared_ptr<utils::RandomGenerator_t<MEM>> rng_dev =
        std::make_shared<utils::RandomGenerator_t<MEM>>(utils::SeedType(919));

    const WalkerSetParameters wlk_pt{.name = "wset0", .walker_type = type};

    WavefunctionFactory<MEM> WfnFac{};
    WavefunctionParameters wfn_pt{.name = "wfn_stoch_bp_est", .filename = wfn_file, .inner_n_samples = 4,
                                  .inner_nsteps = 1, .inner_sampling_target = StochasticSamplingTarget::WalkerOverlap,
                                  .inner_propagator = PropagatorParameters{.timestep = 0.01}};
    utils::mark_stochastic_wfn_input(wfn_pt);
    utils::apply_wfn_defaults(wfn_pt, ham);
    WfnFac.push("wfn_stoch_bp_est", wfn_pt);
    auto& wfn = WfnFac.getWavefunction(mpi, "wfn_stoch_bp_est", type, false, &ham, nwalk);
    WfnFac.maybe_initialize_stochastic_inner_walkers(wfn, "wfn_stoch_bp_est", type, wlk_pt);
    auto const& initial_guess = WfnFac.getInitialGuess("wfn_stoch_bp_est");
    auto wset = WalkerSet<MEM>(mpi, wlk_pt, rng, type, initial_guess, nwalk);

    PropagatorFactory<MEM> PropgFac;
    PropagatorParameters bp_prop_params{.name = "prop_stoch_bp_est"};
    utils::apply_prop_defaults(bp_prop_params, ham);
    PropgFac.push("prop_stoch_bp_est", bp_prop_params);
    auto& prop = PropgFac.getPropagator(mpi, "prop_stoch_bp_est", wfn, rng_dev);

    wfn.Energy(wset);

    ExecuteParameters exec{
        .wavefunction = std::string{"wfn_stoch_bp_est"},
        .hamiltonian  = std::string{"ham0"},
        .estimator    = {EstimatorParameters{.name                      = EstimatorType::back_propagation,
                                             .equil_multiplier          = 0,
                                             .bp_walker_ortho_interval  = 1,
                                             .path_restoration          = false,
                                             .measure_interval_multiplier = std::vector<int>{bp_measure_multiplier},
                                             .onerdm                    = OneRDMParameters{.name = "one_rdm"}}},
        .population_control_interval = population_control_interval,
        .measure_interval_multiplier = bp_measure_multiplier,
    };
    apply_defaults(exec);

    EstimatorHandler<MEM> estim(mpi, title, exec, wset, WfnFac, wfn, prop,
                                HamFac, dt);

    const int measure_interval = estim.get_max_common_interval();
    std::vector<ComplexType> curData;
    float total_time = 0.0f;
    double Eshift    = 0.0;
    int iBlock       = 0;

    for (int iStep = 0; iStep < nStep; ++iStep)
    {
      prop.Propagate(wset, Eshift, dt);
      total_time += dt;

      if (iStep == 0 || (iStep + 1) % population_control_interval == 0)
      {
        wset.processWalkerData(curData);
        // Mirror the production driver's PAIR: store -> popControl -> permute.
        wfn.store_inner_blocks_before_pop(wset);
        wset.popControl();
        wfn.permute_inner_blocks_after_pop(wset); // mirror the production driver: realign inner blocks
        estim.accumulate_step(total_time, wset, curData);
      }

      if ((iStep + 1) % measure_interval == 0)
      {
        estim.accumulate_block(total_time, wset);
        estim.print(iBlock + 1, total_time, Eshift, wset);
        ++iBlock;
      }
    }

    REQUIRE(iBlock >= 1);
    if (mpi->comm.root())
    {
      h5::file h5file(title + ".stat.h5", 'r');
      utils::require_finite_bp_one_rdm(h5file, "Observables/BackPropagated/FullOneRDM/Average_0", 1);
      std::remove((title + ".stat.h5").c_str());
      std::remove((title + ".scalar.dat").c_str());
    }
    mpi->comm.barrier();
  }
}

// With conditioned sampling the BP references are the outer-NOMSD anchor; the dedicated free-projection
// reference draw applies to the forward-unstable free-projection regime, covered in test_stochastic_wfn.
TEST_CASE("estimator_handler: stochastic bp dynamic",
          "[estimator_handler][stochastic_wfn]")
{
  auto& mpi = utils::make_unit_test_mpi_context();
  app_log(0, "BackPropagatedEstimator on a DYNAMIC conditioned+leapfrog stochastic trial.");
  using namespace utils;
  run_test_with_files([&]<auto MEM>(std::string hamil_file, std::string wfn_file, WALKER_TYPES, bool finiteT) {
    stochastic_back_propagation_estimator_smoke<MEM>(mpi, hamil_file, wfn_file);
  }, UTEST_HAMIL, UTEST_WFN, TestFiles::DYNAMIC_INNER);
}

}
