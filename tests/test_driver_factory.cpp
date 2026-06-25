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
#include "IO/app_loggers.h"
#include "IO/ptree/ptree_utilities.hpp"
#include "utilities/Random.hpp"
#include "utilities/Timer.hpp"
#include "test_common.hpp"
#include "utilities/check.hpp"

#include <string>
#include <vector>
#include <complex>
#include <iomanip>

#include "nda/nda.hpp"
#include "nda/tensor.hpp"
#include "nda/h5.hpp"
#include "numerics/sparse/sparse.hpp"
  
#include "test_utils.hpp"
#include "AFQMC/Utilities/readWfn.h"
#include "AFQMC/Utilities/AFQMCTimer.h"

#include "AFQMC/Hamiltonians/HamiltonianFactory.h"
#include "AFQMC/Wavefunctions/WavefunctionFactory.h"
#include "AFQMC/Propagators/PropagatorFactory.h"
#include "AFQMC/Walkers/WalkerSetFactory.hpp"
#include "AFQMC/Drivers/DriverFactory.h"


extern std::string UTEST_HAMIL, UTEST_WFN;

namespace sfqmc
{
using namespace afqmc;

template<MEMORY_SPACE MEM>
void driver_factory_build(std::shared_ptr<utils::mpi_context_t<boost::mpi3::communicator>> mpi,
             std::string hamil_file, std::string wfn_file,
             WALKER_TYPES walker_type = UNDEFINED_WALKER_TYPE)
{
  std::map<std::string, AFQMCInfo> InfoMap;
  HamiltonianFactory HamFac(InfoMap);
  WalkerSetFactory<MEM> WSetFac(InfoMap);
  WavefunctionFactory<MEM> WfnFac{};
  PropagatorFactory<MEM> PropFac(InfoMap);
  DriverFactory<MEM> DriverFac(mpi, InfoMap, WSetFac, PropFac, WfnFac, HamFac);

  const auto[NMO, nup, ndown] = read_info_from_wfn(wfn_file,"any");
  AFQMCInfo info("sys0",NMO,nup,ndown);
  InfoMap.insert(std::pair<std::string, AFQMCInfo>(info.name, info));

  ptree ham_full;
  ham_full.put("name","ham0");
  ham_full.put("system","sys0");
  ham_full.put("filename",hamil_file);
  HamFac.push("ham0", ham_full);

  ptree wfn_full;
  wfn_full.put("name","wfn0");
  wfn_full.put("system","sys0");
  wfn_full.put("filename",wfn_file);
  WfnFac.push("wfn0", wfn_full);

  ptree wlk_full;
  wlk_full.put("name","wlk0");
  wlk_full.put("system","sys0");

  ptree prop_full;
  prop_full.put("name","prop0");
  prop_full.put("system","sys0");
  PropFac.push("prop0", prop_full);

  ptree wfn_min;
  wfn_min.put("filename",wfn_file);

  ptree ham_min;
  ham_min.put("filename",hamil_file);  

  ptree wlk_min;
  wlk_min.put("max_weight","4.0");

  // KE: Some special walker_types must match the wavefunction type;
  //     if an explicit walker type is provided to this test, use it!
  if (walker_type != UNDEFINED_WALKER_TYPE) {
    wlk_full.put("walker_type", walkerTypeToString(walker_type));
    wlk_min.put("walker_type", walkerTypeToString(walker_type));
  }

  WSetFac.push("wlk0", wlk_full);

  ptree prop_min;
  prop_min.put("hybrid","true");

  // Fix the seed so the test is reproducible.
  constexpr int test_seed = 463;

  ptree exec;
  exec.put("seed", test_seed);

  const bool default_walker = (walker_type == UNDEFINED_WALKER_TYPE);

  if (default_walker) {
    exec.put_child("wavefunction",wfn_min);
    // wfn only - this is invalid unless wfn file and hamil file are the same
    if (hamil_file == wfn_file) {
      app_log(0,"[driver_factory] TEST: wfn only (inline); walker_type={}", walkerTypeToString(walker_type));
      CHECK(DriverFac.executeDriver("afqmc","drv_test",0,exec));
    }
    
    // wfn and ham
    exec.put_child("hamiltonian",ham_min);
    app_log(0,"[driver_factory] TEST: wfn+ham (inline); walker_type={}", walkerTypeToString(walker_type));
    CHECK(DriverFac.executeDriver("afqmc","drv_test",0,exec));

    // wfn, ham, prop
    exec.put_child("propagator",prop_min);
    app_log(0,"[driver_factory] TEST: wfn+ham+prop (inline); walker_type={}", walkerTypeToString(walker_type));
    CHECK(DriverFac.executeDriver("afqmc","drv_test",0,exec));
  }

  // wfn, ham, prop, wlk
  exec.clear();
  exec.put("seed", test_seed);
  exec.put_child("wavefunction",wfn_min);
  exec.put_child("hamiltonian",ham_min);
  exec.put_child("propagator",prop_min);
  exec.put_child("walker_set",wlk_min);
  app_log(0,"[driver_factory] TEST: wfn+ham+prop+wlk (all inline); walker_type={}", walkerTypeToString(walker_type));
  CHECK(DriverFac.executeDriver("afqmc","drv_test",0,exec));

  if (default_walker) {
    // external wfn
    exec.clear();
    exec.put("seed", test_seed);
    exec.put("wavefunction","wfn0");
    if (hamil_file == wfn_file) {
      app_log(0,"[driver_factory] TEST: wfn only (external); walker_type={}", walkerTypeToString(walker_type));
      CHECK(DriverFac.executeDriver("afqmc","drv_test",0,exec));
    }

    // wfn and ham
    exec.put("hamiltonian","ham0");
    app_log(0,"[driver_factory] TEST: wfn+ham (external); walker_type={}", walkerTypeToString(walker_type));
    CHECK(DriverFac.executeDriver("afqmc","drv_test",0,exec));

    // wfn, ham, prop
    exec.put("propagator","prop0");
    app_log(0,"[driver_factory] TEST: wfn+ham+prop (external); walker_type={}", walkerTypeToString(walker_type));
    CHECK(DriverFac.executeDriver("afqmc","drv_test",0,exec));
  }

  // wfn, ham, prop, wlk (all external)
  exec.clear();
  exec.put("seed", test_seed);
  exec.put("wavefunction","wfn0");
  exec.put("hamiltonian","ham0");
  exec.put("propagator","prop0");
  exec.put("walker_set","wlk0");
  app_log(0,"[driver_factory] TEST: wfn+ham+prop+wlk (all external); walker_type={}", walkerTypeToString(walker_type));
  CHECK(DriverFac.executeDriver("afqmc","drv_test",0,exec));

  // mixed external internal
  exec.clear();
  exec.put("seed", test_seed);
  exec.put_child("wavefunction",wfn_min);
  exec.put("walker_set","wlk0");
  if (hamil_file == wfn_file) {
    app_log(0,"[driver_factory] TEST: wfn(inline)+wlk(external); walker_type={}", walkerTypeToString(walker_type));
    CHECK(DriverFac.executeDriver("afqmc","drv_test",0,exec));
  }

  if (default_walker) {
    exec.clear();
    exec.put("seed", test_seed);
    exec.put_child("wavefunction",wfn_min);
    exec.put("hamiltonian","ham0");
    app_log(0,"[driver_factory] TEST: wfn(inline)+ham(external); walker_type={}", walkerTypeToString(walker_type));
    CHECK(DriverFac.executeDriver("afqmc","drv_test",0,exec));
  }

  exec.clear();
  exec.put("seed", test_seed);
  exec.put("wavefunction","wfn0");
  exec.put_child("hamiltonian",ham_min);
  exec.put("walker_set","wlk0");
  app_log(0,"[driver_factory] TEST: wfn(external)+ham(inline)+wlk(external); walker_type={}", walkerTypeToString(walker_type));
  CHECK(DriverFac.executeDriver("afqmc","drv_test",0,exec));

  exec.clear();
  exec.put("seed", test_seed);
  exec.put("wavefunction","wfn0");
  exec.put_child("hamiltonian",ham_min);
  exec.put_child("walker_set",wlk_min);
  app_log(0,"[driver_factory] TEST: wfn(external)+ham(inline)+wlk(inline); walker_type={}", walkerTypeToString(walker_type));
  CHECK(DriverFac.executeDriver("afqmc","drv_test",0,exec));
 
  // many more possibilities (combinatorial...) Add any problematic ones if needed
}

TEST_CASE("driver_factory: build", "[driver_factory]")
{
  auto& mpi = utils::make_unit_test_mpi_context();
  
  using namespace utils;

  run_test_with_files([&]<auto MEM>(std::string hamil_file, std::string wfn_file, WALKER_TYPES walker_type) {
    driver_factory_build<MEM>(mpi, hamil_file, wfn_file, walker_type);
  }, UTEST_HAMIL, UTEST_WFN, TestFiles::RHF | TestFiles::UHF | TestFiles::GHF | TestFiles::NOMSD | TestFiles::ALL_SYSTEMS);
}


} // namespace sfqmc
