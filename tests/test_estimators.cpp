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
#include "utilities/check_shape.hpp"
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
#include <memory>

#include "AFQMC/config.h"
#include "AFQMC/Hamiltonians/HamiltonianFactory.h"
#include "AFQMC/Wavefunctions/WavefunctionFactory.h"
#include "AFQMC/Estimators/EstimatorBase.h"
#include "AFQMC/Estimators/Measurements.hpp"
#include "AFQMC/Estimators/BackPropEstimator.hpp"
#include "AFQMC/Propagators/PropagatorFactory.h"
#include "AFQMC/Utilities/readWfn.h"
#include "test_utils.hpp"


extern std::string UTEST_HAMIL, UTEST_WFN;

namespace sfqmc
{
using namespace afqmc;

namespace
{
template<MEMORY_SPACE MEM>
void verify_bp_matches_mixed(Measurements& meas, std::string const& obs_path, int ibin,
                             WALKER_TYPES type, int NMO, int nup, int ndown,
                             Wavefunction<MEM>& wfn, WalkerSet<MEM>& wset)
{
  auto [nspin, npol] = walkerTypeToDims(type);
  int npolNMO = npol * NMO;

  nda::array<ComplexType, 4> bins;
  {
    h5::file file{};
    h5::group root(file);
    meas.write(root);
    h5::read(root, obs_path + "/bins", bins);
  }

  // The measurement output already divides by the weight denominator, so the bins hold
  // the normalized 1RDM directly.
  utils::check_shape(bins, obs_path, bins.extent(0), nspin, npolNMO, npolNMO);
  REQUIRE(ibin < bins.extent(0));
  auto BPRDM = bins(ibin, nda::ellipsis{});

  // Since no back propagation has been performed (dt=0), the BP RDM should
  // equal the mixed estimate.
  ComplexType trace{};
  for(int spin = 0; spin < nspin; spin++) {
    for(int i = 0; i < npolNMO; ++i) {
      trace += BPRDM(spin, i, i);
    }
  }
  if(type == CLOSED) {
    trace *= 2;
  }
  REQUIRE_THAT(trace.real(), utils::Approx(nup + ndown));
  REQUIRE_THAT(trace.imag(), utils::Approx(0.0, 1e-9));

  memory::array<MEM, ComplexType, 2> Gw(wset.size(), nspin * npolNMO * npolNMO);
  wfn.MixedDensityMatrix(wset, Gw, false);
  auto G = nda::reshape(Gw(0,nda::ellipsis{}), std::array<long, 3>{nspin, npolNMO, npolNMO});
  CHECK_THAT(G, utils::Approx(BPRDM));
}

/// Runs the estimator over nblocks measurement blocks. The walkers are never propagated, so
/// only the back propagation history position advances.
template<MEMORY_SPACE MEM>
void run_measurement_blocks(utils::mpi_context_t<boost::mpi3::communicator>& mpi,
                            EstimatorBase<MEM>& estimator, Measurements& meas,
                            WalkerSet<MEM>& wset, int pop_control_interval, long nblocks)
{
  for(long measureBlock = 1; measureBlock <= nblocks; ++measureBlock) {
    // one measurement block worth of propagation steps
    for(int k = 0; k < pop_control_interval; ++k) {
      wset.advanceHistoryPos();
    }
    estimator.measure(mpi, measureBlock, meas, wset);
  }
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

  auto [nspin, npol] = walkerTypeToDims(type);

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

  constexpr int pop_control_interval = afqmc::DEFAULT_POPULATION_CONTROL_INTERVAL;
  constexpr long nblocks = 8;

  // ---- Run 1: single measure_interval_multiplier ----
  {
    const EstimatorParameters est_params{.name = EstimatorType::back_propagation,
                                         .measure_interval_multiplier = std::vector<int>{2},
                                         .onerdm = OneRDMParameters{}};

    std::unique_ptr<EstimatorBase<MEM>> estimator = std::make_unique<BackPropEstimator<MEM>>(
        *mpi, est_params, pop_control_interval, wset, wfn, prop);

    // measurements land on blocks 2 and 5, since the BP anchor resets after 2 blocks
    Measurements meas{};
    run_measurement_blocks(*mpi, *estimator, meas, wset, pop_control_interval, nblocks);
    verify_bp_matches_mixed<MEM>(meas, "BackPropEstimator/Steps=2/OneRDM", 0,
                                 type, NMO, nup, ndown, wfn, wset);
  }

  // ---- Run 2: multiple measure_interval_multipliers ----
  {
    const EstimatorParameters est_params{.name = EstimatorType::back_propagation,
                                         .measure_interval_multiplier = std::vector<int>{1, 2, 3},
                                         .onerdm = OneRDMParameters{}};

    std::unique_ptr<EstimatorBase<MEM>> estimator = std::make_unique<BackPropEstimator<MEM>>(
        *mpi, est_params, pop_control_interval, wset, wfn, prop);

    // the anchor resets after 3 blocks, so Steps=2 is measured on blocks 2 and 6
    Measurements meas{};
    run_measurement_blocks(*mpi, *estimator, meas, wset, pop_control_interval, nblocks);
    verify_bp_matches_mixed<MEM>(meas, "BackPropEstimator/Steps=2/OneRDM", 0,
                                 type, NMO, nup, ndown, wfn, wset);
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
