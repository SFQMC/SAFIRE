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

#include "IO/ptree/ptree_utilities.hpp"
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
  auto[NMO,nup, ndown] = read_info_from_wfn(wfn_file, "any");
  utils::check(NMO == read_nmo_from_hdf(hamil_file), "NMO differ between hamil and wfn files.");

  std::shared_ptr<utils::RandomGenerator_t<>> rng = std::make_shared<utils::RandomGenerator_t<>>();
  std::shared_ptr<utils::RandomGenerator_t<MEM>> rng_dev = std::make_shared<utils::RandomGenerator_t<MEM>>(utils::make_rng<MEM>(777));

  std::map<std::string, AFQMCInfo> InfoMap;
  InfoMap.insert(std::pair<std::string, AFQMCInfo>("info0", AFQMCInfo{"info0", NMO, nup, ndown}));

  ptree ham_pt;
  ham_pt.put("name","ham0");
  ham_pt.put("system","info0");
  ham_pt.put("filename",hamil_file);

  HamiltonianFactory HamFac(InfoMap);
  HamFac.push("ham0", ham_pt);
  Hamiltonian& ham = HamFac.getHamiltonian(mpi, "ham0");

  WALKER_TYPES type = afqmc::getWalkerType(wfn_file);
  ptree wlk_pt;
  wlk_pt.put("name","wset0");
  wlk_pt.put("system","info0");
  wlk_pt.put("walker_type", walkerTypeToString(type));

  auto wset = make_WalkerSet<MEM>(mpi, wlk_pt, InfoMap["info0"], rng);

  int nspin            = (type == COLLINEAR) ? 2 : 1;
  int npol             = (type == NONCOLLINEAR) ? 2 : 1;

  ptree wfn_pt;
  wfn_pt.put("name","wfn0");
  wfn_pt.put("system","info0");
  wfn_pt.put("filename",wfn_file);
  wfn_pt.put("dense_trial",true);

  int nwalk = 11;
  WavefunctionFactory<MEM> WfnFac{};
  WfnFac.push("wfn0", wfn_pt);
  auto& wfn = WfnFac.getWavefunction(mpi, "wfn0", type, &ham, nwalk);

  ptree prop_pt;
  prop_pt.put("name","prop0");
  prop_pt.put("system","info0");

  PropagatorFactory<MEM> PropgFac(InfoMap);
  PropgFac.push("prop0", prop_pt);
  auto& prop = PropgFac.getPropagator(mpi, "prop0", wfn, rng_dev);

  auto initial_guess = WfnFac.getInitialGuess("wfn0");
  REQUIRE(initial_guess.shape() == std::array<long,3>{nspin,npol*NMO,nup});
  wset.resize(nwalk, initial_guess);
  
  // number of steps to propagate
  int nStep = 200;

  // define / run test cases
  std::vector<ptree> cases;

  ptree test_case;

  test_case.put("name", "case1");
  test_case.put("meas1", 5);
  test_case.put("meas2", 20);
  cases.push_back(test_case);
  test_case.clear();

  test_case.put("name", "case2");
  test_case.put("meas1", 20);
  test_case.put("meas2", 10);
  cases.push_back(test_case);
  test_case.clear();
  
  test_case.put("name", "case3");
  test_case.put("meas1", 5);
  test_case.put("meas2", 7);
  cases.push_back(test_case);
  test_case.clear();
    
  test_case.put("name", "case4");
  test_case.put("meas1", 11);
  test_case.put("meas2", 7);
  cases.push_back(test_case);
  test_case.clear();
  
  // test that we default properly
  test_case.put("name", "case5");
  test_case.put("meas1", 1);
  test_case.put("meas2", 7);
  cases.push_back(test_case);
  test_case.clear();

  ptree one_rdm;
  one_rdm.put("name","one_rdm");

  for (auto test_ptree: cases)
  {
    //TODO update the PropertyTee for our test case(s) using new parameter names
    ptree est_pt_energy;
    est_pt_energy.put("name","energy");
    est_pt_energy.put("overwrite",true);
    
    app_log(1,"\nEstimator input:\n{}\n",io::to_string(est_pt_energy));

    ptree est_pt_bp;
    est_pt_bp.put("name","back_propagation");
    est_pt_bp.put("measure_interval_multiplier",test_ptree.get<int>("meas2"));
    est_pt_bp.put("equil_multiplier",0);
    est_pt_bp.put("bp_walker_ortho_interval",1);
    est_pt_bp.add_child("onerdm",one_rdm);

    app_log(1,"\nEstimator input:\n{}\n",io::to_string(est_pt_bp));

    ptree est_pt;
    est_pt.add_child("estimator",est_pt_energy);
    est_pt.add_child("estimator",est_pt_bp);
    est_pt.put("population_control_interval",population_control_interval);
    // Set the global measure_interval_multiplier that will be used by BasicEstimator and EnergyEstimator
    est_pt.put("measure_interval_multiplier",test_ptree.get<int>("meas1"));

    // to verify the ptree
    std::cout <<" Test case Ptree:  "<< std::endl;
    std::cout << io::to_string(est_pt) << std::endl;

    int measure_interval{};
    {
      int nPopulation = 1;
      float dt = 0.01f;
      float total_time = 0.0f;
      double E1 = 0.0;
      EstimatorHandler<MEM> estim0(mpi, InfoMap["info0"], "test_est_handler",
        est_pt, wset, WfnFac, wfn, prop,
                          type, HamFac, "ham0", dt);
    
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
          wset.popControl(); // make this a call to actual pop control
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
    int energy_interval = test_ptree.get<int>("meas1") * population_control_interval;
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
    app_log(1, "\n[TESTS] Running test case: {} \n",test_ptree.get<std::string>("name","no name"));
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

  std::string hamil = utils::unit_test_base() + "models/square_4x4_hubbard_nup5_ndn5/afqmc_inputs/ham_collinear.h5";
  std::string wfn   = utils::unit_test_base() + "models/square_4x4_hubbard_nup5_ndn5/afqmc_inputs/uhf_U0.1_wfn_nup5_ndn5.h5";
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
