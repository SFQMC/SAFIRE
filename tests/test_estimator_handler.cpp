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

#include "nda/nda.hpp"
#include "nda/tensor.hpp"
#include "nda/h5.hpp"

#include "AFQMC/Hamiltonians/HamiltonianFactory.h"
#include "AFQMC/Wavefunctions/WavefunctionFactory.h"
#include "AFQMC/Walkers/WalkerSetFactory.hpp"
#include "AFQMC/Estimators/EstimatorBase.h"
#include "AFQMC/Estimators/EstimatorHandler.h"
#include "AFQMC/Propagators/PropagatorFactory.h"
//#include "AFQMC/Estimators/BackPropagatedEstimator.hpp"
#include "test_utils.hpp"
#include "AFQMC/Utilities/AFQMCTimer.h"
#include "AFQMC/Utilities/readWfn.h"


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
          auto popcontrol_time = timers.popcontrol.start();
          wset.processWalkerData(dummyData);
          wset.popControl(); // make this a call to actual pop control
          popcontrol_time.stop();
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

}
