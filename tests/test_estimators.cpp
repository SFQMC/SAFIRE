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

#include "AFQMC/parameters.hpp"
#include "AFQMC/parameter_defaults.hpp"
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
  std::shared_ptr<utils::RandomGenerator_t<MEM>> rng_dev = std::make_shared<utils::RandomGenerator_t<MEM>>(777);

  HamiltonianFactory HamFac;
  HamFac.push("ham0", HamiltonianParameters{.name = "ham0", .filename = hamil_file});
  auto& ham = HamFac.getHamiltonian(mpi, "ham0");

  WALKER_TYPES type = afqmc::getWalkerType(wfn_file);
  const WalkerSetParameters wlk_params{.name = "wset0", .walker_type = type};

  int nspin = (type == COLLINEAR) ? 2 : 1;
  int npol  = (type == NONCOLLINEAR) ? 2 : 1;

  int nwalk = 2;
  WavefunctionFactory<MEM> WfnFac{};
  WavefunctionParameters wfn_params{.name = "wfn0", .filename = wfn_file};
  apply_defaults(wfn_params, ham.getHamType());
  WfnFac.push("wfn0", wfn_params);
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

  // generate P1 with dt=0 so the BP RDM should match the mixed estimate
  // we cannot actually use exactly 0 because that changes the sparsity structure in model hamiltonians
  prop.generateP1(1e-10, wset.getWalkerType());

  using EstimPtr = std::unique_ptr<EstimatorBase<MEM>>;

  constexpr int pop_control_interval = afqmc::DEFAULT_POPULATION_CONTROL_INTERVAL;

  // ---- Run 1: scalar measure_interval_multiplier ----
  {
    // test with a single value
    const EstimatorParameters est_params{.name = EstimatorType::back_propagation,
                                         .measure_interval_multiplier = std::vector<int>{2},
                                         .onerdm = OneRDMParameters{}};

    std::vector<EstimPtr> estimators;
    estimators.push_back(std::make_unique<BackPropagatedEstimator<MEM>>(
        mpi, "none", est_params, pop_control_interval, wset, wfn, prop, true));

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
    const EstimatorParameters est_params{.name = EstimatorType::back_propagation,
                                         .measure_interval_multiplier = std::vector<int>{1, 2, 3},
                                         .onerdm = OneRDMParameters{}};

    std::vector<EstimPtr> estimators2;
    estimators2.push_back(std::make_unique<BackPropagatedEstimator<MEM>>(
        mpi, "none", est_params, pop_control_interval, wset, wfn, prop, true));

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
