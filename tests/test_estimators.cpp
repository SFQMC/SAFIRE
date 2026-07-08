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

#include "IO/ptree/ptree_utilities.hpp"
#include "utilities/Random.hpp"
#include "utilities/check.hpp"
#include "test_common.hpp"
#include "utilities/h5_utils.hpp"
#include "IO/app_loggers.h"

#include <nda/nda.hpp>
#include <nda/tensor.hpp>
#include <nda/h5.hpp>

#include <string>
#include <vector>
#include <complex>
#include <format>
#include <fstream>

#include "AFQMC/config.h"
#include "AFQMC/Hamiltonians/HamiltonianFactory.h"
#include "AFQMC/Wavefunctions/WavefunctionFactory.h"
#include "AFQMC/Estimators/EstimatorBase.h"
#include "AFQMC/Estimators/BackPropagatedEstimator.hpp"
#include "AFQMC/Propagators/PropagatorFactory.h"
#include "AFQMC/Utilities/readWfn.h"
#include "test_utils.hpp"


extern std::string UTEST_HAMIL, UTEST_WFN;

namespace sfqmc
{
using namespace afqmc;

namespace
{
inline std::string test_hdf_name(std::string const& wfn_file,
                                 std::string const& hamil_file,
                                 std::string const& suffix)
{
  auto stem = [](std::string const& p) {
    auto s = p.find_last_of("\\/");
    auto e = p.find_last_of(".");
    return p.substr(s + 1, e - s - 1);
  };
  return stem(wfn_file) + "_" + stem(hamil_file) + "_" + suffix + ".h5";
}

template<MEMORY_SPACE MEM>
void verify_bp_matches_mixed(h5::file const& file, std::string const& avg_path, int iblock,
                             WALKER_TYPES type, int NMO, int nup, int ndown,
                             Wavefunction<MEM>& wfn, WalkerSet<MEM>& wset)
{
  int nspin = (type == COLLINEAR) ? 2 : 1;
  int npol  = (type == NONCOLLINEAR) ? 2 : 1;

  std::string suffix = std::format("{:09d}", iblock);

  nda::array<ComplexType, 1> read_data;
  ComplexType denom{};
  {
    h5::group root(file);
    utils::h5_read(root, avg_path + "/one_rdm_" + suffix, read_data);
    h5::read(root, avg_path + "/denominator_" + suffix, denom);
  }

  // Since no back propagation has been performed (dt=0), the BP RDM should
  // equal the mixed estimate.
  REQUIRE(read_data.size() == nspin * npol * NMO * npol * NMO);
  auto BPRDM = nda::reshape(read_data, std::array<long, 3>{nspin, npol * NMO, npol * NMO});
  BPRDM *= 1.0 / denom;

  ComplexType trace{};
  for(int spin = 0; spin < nspin; spin++) {
    for(int i = 0; i < npol * NMO; ++i) {
      trace += BPRDM(spin, i, i);
    }
  }
  if(type == CLOSED) {
    trace *= 2;
  }
  REQUIRE_THAT(trace.real(), utils::Approx(nup + ndown));
  REQUIRE_THAT(trace.imag(), utils::Approx(0.0, 1e-9));

  memory::array<MEM, ComplexType, 2> Gw(wset.size(), nspin * npol * NMO * npol * NMO);
  wfn.MixedDensityMatrix(wset, Gw, false);
  auto G = nda::reshape(Gw(0,nda::ellipsis{}), std::array<long, 3>{nspin, npol * NMO, npol * NMO});
  CHECK_THAT(G, utils::Approx(BPRDM));
}
} // namespace

template<MEMORY_SPACE MEM>
void estimators_reduced_density_matrix(std::shared_ptr<utils::mpi_context_t<boost::mpi3::communicator>> mpi,
             std::string hamil_file, std::string wfn_file)
{
  auto [NMO, nup, ndown] = read_info_from_wfn(wfn_file, "any");
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
  auto& ham = HamFac.getHamiltonian(mpi, "ham0");

  WALKER_TYPES type = afqmc::getWalkerType(wfn_file);
  ptree wlk_pt;
  wlk_pt.put("name","wset0");
  wlk_pt.put("system","info0");
  wlk_pt.put("walker_type", walkerTypeToString(type));
  auto wset = make_WalkerSet<MEM>(mpi, wlk_pt, InfoMap["info0"], rng);

  int nspin = (type == COLLINEAR) ? 2 : 1;
  int npol  = (type == NONCOLLINEAR) ? 2 : 1;

  ptree wfn_pt;
  wfn_pt.put("name","wfn0");
  wfn_pt.put("system","info0");
  wfn_pt.put("filename",wfn_file);

  int nwalk = 2;
  WavefunctionFactory<MEM> WfnFac{};
  WfnFac.push("wfn0", wfn_pt);
  auto& wfn = WfnFac.getWavefunction(mpi, "wfn0", type, false, &ham, nwalk);

  ptree prop_pt;
  prop_pt.put("name","prop0");
  prop_pt.put("system","info0");

  PropagatorFactory<MEM> PropgFac(InfoMap);
  PropgFac.push("prop0", prop_pt);
  auto& prop = PropgFac.getPropagator(mpi, "prop0", wfn, rng_dev);

  auto initial_guess = WfnFac.getInitialGuess("wfn0");
  REQUIRE(initial_guess.shape() == std::array<long,3>{nspin,npol*NMO,nup});
  wset.resize(nwalk, initial_guess);

  // generate P1 with dt=0 so the BP RDM should match the mixed estimate
  // we cannot actually use exactly 0 because that changes the sparsity structure in model hamiltonians
  prop.generateP1(1e-10, wset.getWalkerType());

  using EstimPtr = std::unique_ptr<EstimatorBase<MEM>>;

  // ---- Run 1: scalar measure_interval_multiplier ----
  {
    ptree est_pt;
    est_pt.put("name","back_propagation");
    est_pt.put("measure_interval_multiplier", 2); // test with single value
    est_pt.put("path_restoration","no");
    est_pt.put("onerdm.nskip_output",0);
    app_log(1,"\nEstimator input:\n{}\n",io::to_string(est_pt));

    std::vector<EstimPtr> estimators;
    estimators.push_back(std::make_unique<BackPropagatedEstimator<MEM>>(
        mpi, InfoMap["info0"], "none", est_pt, type, wset, wfn, prop, true));

    std::string file = test_hdf_name(UTEST_WFN, UTEST_HAMIL, "run1");
    std::ofstream out;
    {
      h5::file h5out{};
      for (int iblock = 0; iblock < 10*afqmc::DEFAULT_POPULATION_CONTROL_INTERVAL; ++iblock)
      {
        wset.advanceBPPos();
        estimators[0]->accumulate_block(iblock*0.01, wset);
        estimators[0]->print(out, h5out, wset);
      }
      verify_bp_matches_mixed<MEM>(h5out,
          "Observables/BackPropagated/FullOneRDM/Average_0", 5,
          type, NMO, nup, ndown, wfn, wset);
    }

  }

  // ---- Run 2: vector measure_interval_multiplier ----
  {
    ptree est_pt2;
    est_pt2.put("name","back_propagation");

    std::vector<int> nback_prop_interval_multipliers = {1, 2, 3};
    ptree temp_tree;
    for (const auto& value : nback_prop_interval_multipliers) {
      ptree item;
      item.put("", value);
      temp_tree.push_back(std::make_pair("", item));
    }
    est_pt2.add_child("measure_interval_multiplier", temp_tree);
    est_pt2.put("path_restoration","no");
    est_pt2.put("onerdm.nskip_output",0);
    app_log(1,"\nEstimator input:\n{}\n",io::to_string(est_pt2));

    std::vector<EstimPtr> estimators2;
    estimators2.push_back(std::make_unique<BackPropagatedEstimator<MEM>>(
        mpi, InfoMap["info0"], "none", est_pt2, type, wset, wfn, prop, true));

    std::string file = test_hdf_name(wfn_file, hamil_file, "run2");
    std::ofstream out;
    {
      h5::file h5out{};
      // 6 * pc_interval = 60 steps, with max_nback_prop = 3 * pc_interval = 30
      // => iblock reaches 2; check Average_1 (multiplier=2) written at iblock=2
      for (int iblock = 0; iblock < 6*afqmc::DEFAULT_POPULATION_CONTROL_INTERVAL; ++iblock)
      {
        wset.advanceBPPos();
        estimators2[0]->accumulate_block(iblock*0.01, wset);
        estimators2[0]->print(out, h5out, wset);
      }
      verify_bp_matches_mixed<MEM>(h5out,
          "Observables/BackPropagated/FullOneRDM/Average_1", 2,
          type, NMO, nup, ndown, wfn, wset);
    }

  }
}

TEST_CASE("estimators: reduced density matrix", "[estimators]")
{
  auto& mpi = utils::make_unit_test_mpi_context();

  using namespace utils;

  run_test_with_files([&]<auto MEM>(std::string hamil_file, std::string wfn_file, WALKER_TYPES, bool) {
    estimators_reduced_density_matrix<MEM>(mpi, hamil_file, wfn_file);
  }, UTEST_HAMIL, UTEST_WFN, TestFiles::RHF | TestFiles::UHF | TestFiles::GHF | TestFiles::NOMSD | TestFiles::PHMSD | TestFiles::ALL_SYSTEMS);
}

} // namespace sfqmc
