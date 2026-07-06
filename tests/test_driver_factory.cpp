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
#include <fstream>
#include <format>

#include "AFQMC/config.h"

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
  bool ft = (walker_type == COLLINEAR_FT or walker_type == NONCOLLINEAR_FT);

  AFQMCInfo info;//("sys0",NMO,nup,ndown,ntau);
  
  info.name = "sys0";
  info.NMO = NMO;
  if(ft){
    //for finite-T wfn dims are [NMO, ntau, 0]
    //walker matrices are NMO x NMO, not NMO x nelec
    info.nup = NMO;
    info.ndown = NMO;
    info.ntau = nup;
  }
  else{
    info.nup = nup;
    info.ndown = ndown;
  }

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
  app_log(0,"[driver_fac] TEST: wfn+ham+prop+wlk (all inline); walker_type={}", walkerTypeToString(walker_type));
  if(ft)
    CHECK(DriverFac.executeDriver("ftafqmc","drv_test",0,exec));
  else
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
  app_log(0,"[driver_fac] TEST: wfn+ham+prop+wlk (all external); walker_type={}", walkerTypeToString(walker_type));
  if(ft)
    CHECK(DriverFac.executeDriver("ftafqmc","drv_test",0,exec));
  else
    CHECK(DriverFac.executeDriver("afqmc","drv_test",0,exec));

  // mixed external internal
  exec.clear();
  exec.put("seed", test_seed);
  exec.put_child("wavefunction",wfn_min);
  exec.put("walker_set","wlk0");
  if (hamil_file == wfn_file) {
    app_log(0,"[driver_fac] TEST: wfn(inline)+wlk(external); walker_type={}", walkerTypeToString(walker_type));
    if(ft)
      CHECK(DriverFac.executeDriver("ftafqmc","drv_test",0,exec));
    else
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
  app_log(0,"[driver_fac] TEST: wfn(external)+ham(inline)+wlk(external); walker_type={}", walkerTypeToString(walker_type));
  if(ft)
    CHECK(DriverFac.executeDriver("ftafqmc","drv_test",0,exec));
  else
    CHECK(DriverFac.executeDriver("afqmc","drv_test",0,exec));

  exec.clear();
  exec.put("seed", test_seed);
  exec.put("wavefunction","wfn0");
  exec.put_child("hamiltonian",ham_min);
  exec.put_child("walker_set",wlk_min);
  app_log(0,"[driver_fac] TEST: wfn(external)+ham(inline)+wlk(inline); walker_type={}", walkerTypeToString(walker_type));
  if(ft)
    CHECK(DriverFac.executeDriver("ftafqmc","drv_test",0,exec));
  else
    CHECK(DriverFac.executeDriver("afqmc","drv_test",0,exec));

  // many more possibilities (combinatorial...) Add any problematic ones if needed
}

TEST_CASE("driver_factory: build", "[driver_factory]")
{
  auto& mpi = utils::make_unit_test_mpi_context();
  
  using namespace utils;

  run_test_with_files([&]<auto MEM>(std::string hamil_file, std::string wfn_file, WALKER_TYPES walker_type) {
    driver_factory_build<MEM>(mpi, hamil_file, wfn_file, walker_type);
  }, UTEST_HAMIL, UTEST_WFN, TestFiles::RHF | TestFiles::UHF | TestFiles::GHF | TestFiles::NOMSD | TestFiles::FINITE_T | TestFiles::ALL_SYSTEMS);
}


namespace {
void mark_stochastic_wfn_input(ptree& pt) { pt.put("type", "stochasticwfn"); }

template<MEMORY_SPACE MEM>
void require_finite_bp_one_rdm(h5::file const& file, std::string const& avg_path, int iblock)
{
  std::string suffix = std::format("{:09d}", iblock);
  nda::array<ComplexType, 1> read_data;
  ComplexType denom{};
  {
    h5::group root(file);
    utils::h5_read(root, avg_path + "/one_rdm_" + suffix, read_data);
    h5::read(root, avg_path + "/denominator_" + suffix, denom);
  }
  REQUIRE(read_data.size() > 0);
  REQUIRE(std::abs(denom) > 0.0);
  for (auto v : read_data)
  {
    REQUIRE(std::isfinite(real(v)));
    REQUIRE(std::isfinite(imag(v)));
  }
}
} // namespace

// Integration smoke: minimal DriverFactory run with type: stochasticwfn (static delegate limit,
// inner_nsteps = 0) and a back_propagation estimator block.
template<MEMORY_SPACE MEM>
void stochastic_back_propagation_driver_smoke(
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
    if (type != CLOSED)
      return;

    const int population_control_interval = DEFAULT_POPULATION_CONTROL_INTERVAL;
    const int bp_measure_multiplier       = 2;
    const int nStep                       = bp_measure_multiplier * population_control_interval * 2;
    const std::string title               = "stoch_bp_drv_smoke";

    std::map<std::string, AFQMCInfo> InfoMap;
    HamiltonianFactory HamFac(InfoMap);
    WalkerSetFactory<MEM> WSetFac(InfoMap);
    WavefunctionFactory<MEM> WfnFac{};
    PropagatorFactory<MEM> PropFac(InfoMap);
    DriverFactory<MEM> DriverFac(mpi, InfoMap, WSetFac, PropFac, WfnFac, HamFac);

    ptree wfn_min;
    wfn_min.put("filename", wfn_file);
    mark_stochastic_wfn_input(wfn_min);
    wfn_min.put("inner_nwalkers", 4);
    wfn_min.put("inner_nsteps", 0);

    ptree ham_min;
    ham_min.put("filename", hamil_file);

    ptree wlk_min;
    wlk_min.put("max_weight", "4.0");
    wlk_min.put("walker_type", walkerTypeToString(CLOSED));

    ptree prop_min;
    prop_min.put("hybrid", "true");

    ptree one_rdm;
    one_rdm.put("name", "one_rdm");
    ptree est_bp;
    est_bp.put("name", "back_propagation");
    est_bp.put("measure_interval_multiplier", bp_measure_multiplier);
    est_bp.put("equil_multiplier", 0);
    est_bp.put("bp_walker_ortho_interval", 1);
    est_bp.put("path_restoration", "no");
    est_bp.put("onerdm.nskip_output", 0);
    est_bp.add_child("onerdm", one_rdm);

    ptree exec;
    exec.put("seed", 463);
    exec.put("steps", nStep);
    exec.put("timestep", 0.01);
    exec.put("population_control_interval", population_control_interval);
    exec.put("measure_interval_multiplier", bp_measure_multiplier);
    exec.put("n_walkers_per_mpi_task", 11);
    exec.put_child("wavefunction", wfn_min);
    exec.put_child("hamiltonian", ham_min);
    exec.put_child("propagator", prop_min);
    exec.put_child("walker_set", wlk_min);
    exec.add_child("estimator", est_bp);

    if (mpi->comm.root())
    {
      std::remove((title + ".stat.h5").c_str());
      std::remove((title + ".scalar.dat").c_str());
    }
    mpi->comm.barrier();

    CHECK(DriverFac.executeDriver("afqmc", title, 0, exec));

    mpi->comm.barrier();
    if (mpi->comm.root())
    {
      std::string scalar_file = title + ".scalar.dat";
      std::ifstream in(scalar_file.c_str());
      CHECK(in.good());
      in.close();
      h5::file h5file(title + ".stat.h5", 'r');
      require_finite_bp_one_rdm<MEM>(h5file, "Observables/BackPropagated/FullOneRDM/Average_0", 1);
      std::remove(scalar_file.c_str());
      std::remove((title + ".stat.h5").c_str());
    }
    mpi->comm.barrier();
  }
}

TEST_CASE("stochastic_back_propagation_driver_smoke", "[driver_factory][stochastic_wfn]")
{
  auto& mpi = utils::make_unit_test_mpi_context();
  app_log(0, "DriverFactory AFQMC run with stochastic trial + back_propagation estimator.");
  using namespace utils;
  run_test_with_files([&]<auto MEM>(std::string hamil_file, std::string wfn_file, WALKER_TYPES) {
    stochastic_back_propagation_driver_smoke<MEM>(mpi, hamil_file, wfn_file);
  }, UTEST_HAMIL, UTEST_WFN, TestFiles::RHF | TestFiles::UHF | TestFiles::NOMSD | TestFiles::ALL_SYSTEMS);
}

} // namespace sfqmc
