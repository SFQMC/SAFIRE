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
#include <filesystem>
#include <format>
#include <map>
#include <memory>

#include "AFQMC/config.h"
#include "AFQMC/Hamiltonians/Hamiltonian.hpp"
#include "AFQMC/Wavefunctions/Wavefunction.hpp"
#include "AFQMC/Estimators/EstimatorBase.h"
#include "AFQMC/Estimators/Measurements.hpp"
#include "AFQMC/Estimators/BackPropEstimator.hpp"
#include "AFQMC/Estimators/Estimators.hpp"
#include "AFQMC/Propagators/AFQMCBasePropagator.h"
#include "AFQMC/Propagators/Propagator.hpp"
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

/// Runs every estimator of the set over the measurement blocks [first, last]. The walkers are
/// never propagated, so only the back propagation history position advances.
template<MEMORY_SPACE MEM>
void run_measurement_blocks(utils::mpi_context_t<boost::mpi3::communicator>& mpi,
                            Estimators<MEM>& estimators, WalkerSet<MEM>& wset,
                            int pop_control_interval, long first, long last)
{
  for(long measureBlock = first; measureBlock <= last; ++measureBlock) {
    for(int k = 0; k < pop_control_interval; ++k) {
      wset.advanceHistoryPos();
    }
    estimators.measure(mpi, measureBlock, wset);
  }
}

/// Number of bins stored in the "bins" dataset of `group`. The bin count is the leading
/// extent, which is the one thing every observable's dataset has in common: the remaining
/// rank varies with the observable, and complex values add a trailing dimension on top.
long bin_count(h5::group const& group)
{
  h5::dataset dset = group.open_dataset("bins");
  h5::dataspace dspace = H5Dget_space(dset);
  std::vector<hsize_t> dims(H5Sget_simple_extent_ndims(dspace));
  H5Sget_simple_extent_dims(dspace, dims.data(), nullptr);
  utils::check(!dims.empty(), "the dataset 'bins' is a scalar");
  return long(dims[0]);
}

void collect_bins(h5::group const& group, std::string const& path, std::map<std::string, long>& bins)
{
  if(group.has_dataset("bins")) {
    bins[path] = bin_count(group);
  }
  for(auto const& name : group.get_all_subgroup_names()) {
    collect_bins(group.open_group(name), path.empty() ? name : std::format("{}/{}", path, name), bins);
  }
}

/// Every observable of a results file, as the path below "Measurements/Stage0" that holds its
/// "bins" dataset mapped to the number of bins in it. Reports a missing group without
/// throwing, so that a failure cannot unwind root past a later collective.
std::map<std::string, long> collect_bins(std::filesystem::path const& file)
{
  std::map<std::string, long> bins;

  h5::file out(file.string(), 'r');
  h5::group root(out);
  if(!root.has_subgroup("Measurements")) {
    FAIL_CHECK(std::format("'{}' holds no 'Measurements' group", file.string()));
    return bins;
  }
  h5::group measurements = root.open_group("Measurements");
  if(!measurements.has_subgroup("Stage0")) {
    FAIL_CHECK(std::format("'{}' holds no 'Measurements/Stage0' group", file.string()));
    return bins;
  }
  collect_bins(measurements.open_group("Stage0"), "", bins);

  return bins;
}

/// The observables of an estimator that owns an Observables<MEM>, below its output prefix.
/// PairCorr is named after the orbital pair maps it was given, here the single map "s".
void expect_observables(std::map<std::string, long>& expected, std::string_view prefix, long nbins)
{
  for(auto const* name : {"OneRDM", "TwoRDM", "DiagonalTwoRDM", "SpinCorr", "PairCorr/s_s"}) {
    expected[std::format("{}/{}", prefix, name)] = nbins;
  }
}

/// Compares the observables of a results file against the expected ones by name, reporting
/// every missing, surplus and miscounted dataset rather than stopping at the first.
void check_bins(std::map<std::string, long> const& got, std::map<std::string, long> const& expected)
{
  for(auto const& [path, nbins] : expected) {
    auto const it = got.find(path);
    if(it == got.end()) {
      FAIL_CHECK(std::format("missing dataset 'Measurements/Stage0/{}/bins'", path));
    } else if(it->second != nbins) {
      FAIL_CHECK(std::format("'Measurements/Stage0/{}/bins' holds {} bins, expected {}", path, it->second, nbins));
    }
  }
  for(auto const& [path, nbins] : got) {
    if(!expected.contains(path)) {
      FAIL_CHECK(std::format("unexpected dataset 'Measurements/Stage0/{}/bins' with {} bins", path, nbins));
    }
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

  Hamiltonian ham = Hamiltonian::from_params(mpi, HamiltonianParameters{.name = "ham0", .filename = hamil_file});

  WALKER_TYPES type = afqmc::getWalkerType(wfn_file);
  const WalkerSetParameters wlk_params{.name = "wset0", .walker_type = type};

  auto [nspin, npol] = walkerTypeToDims(type);

  int nwalk = 2;
  WavefunctionParameters wfn_params{.name = "wfn0", .filename = wfn_file};
  apply_defaults(wfn_params, ham.getHamType());
  auto wfn = Wavefunction<MEM>::from_params(mpi, wfn_params, type, false, ham, nwalk);

  PropagatorParameters prop_params{.name = "prop0"};
  apply_defaults(prop_params, ham.getHamType());
  Propagator<MEM> prop{AFQMCBasePropagator<MEM>(prop_params, mpi, wfn, rng_dev)};

  auto const& initial_guess = wfn.initial_guess();
  REQUIRE(int(initial_guess.slater().size()) == nspin);
  REQUIRE(initial_guess.slater()[0].shape() == std::array<long,2>{npol*NMO,nup});
  auto wset = WalkerSet<MEM>(mpi, rng, wlk_params, initial_guess, nwalk);

  // generate P1 with dt=0 so the BP RDM should match the mixed estimate
  // we cannot actually use exactly 0 because that changes the sparsity structure in model hamiltonians
  prop.generateP1(1e-10, wset.getWalkerType());

  constexpr int pop_control_interval = afqmc::DEFAULT_POPULATION_CONTROL_INTERVAL;
  constexpr long nblocks = 8;

  // ---- Run 1: single measure_interval_multiplier ----
  {
    const BackPropEstimatorParameters est_params{
        .measure_interval_multiplier = std::vector<int>{2},
        .walker_ortho_interval       = afqmc::DEFAULT_WALKER_ORTHO_INTERVAL,
        .onerdm                      = OneRDMParameters{}};

    std::unique_ptr<EstimatorBase<MEM>> estimator = std::make_unique<BackPropEstimator<MEM>>(
        *mpi, est_params, pop_control_interval, wset, wfn, prop);

    // the anchor starts at block -1 and resets after 2 blocks, so measurements land on the
    // odd blocks 1, 3, 5, 7
    Measurements meas{};
    run_measurement_blocks(*mpi, *estimator, meas, wset, pop_control_interval, nblocks);
    verify_bp_matches_mixed<MEM>(meas, "BackPropEstimator/Steps=2/OneRDM", 0,
                                 type, NMO, nup, ndown, wfn, wset);
  }

  // ---- Run 2: multiple measure_interval_multipliers ----
  {
    const BackPropEstimatorParameters est_params{
        .measure_interval_multiplier = std::vector<int>{1, 2, 3},
        .walker_ortho_interval       = afqmc::DEFAULT_WALKER_ORTHO_INTERVAL,
        .onerdm                      = OneRDMParameters{}};

    std::unique_ptr<EstimatorBase<MEM>> estimator = std::make_unique<BackPropEstimator<MEM>>(
        *mpi, est_params, pop_control_interval, wset, wfn, prop);

    // the anchor resets after 3 blocks, so Steps=2 is measured on blocks 1, 4 and 7
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

/// Stands every estimator type up side by side with all the observables it supports, and
/// checks that the results file holds exactly the expected datasets with the number of bins
/// each estimator's measurement schedule implies, both on the first write and after a second
/// one has appended to it.
template<MEMORY_SPACE MEM>
void estimators_all_observables(std::shared_ptr<utils::mpi_context_t<boost::mpi3::communicator>> mpi)
{
  // collinear walkers are the only ones every observable supports: TwoRDM is implemented for
  // them alone, and PairCorr rejects CLOSED
  std::string const inputs = utils::unit_test_base() + "BH/";
  std::string const hamil_file = inputs + "afqmc_H_rhf_collinear.h5";
  std::string const wfn_file   = inputs + "afqmc_uhf_nomsd.h5";

  int const NMO = read_nmo_from_hdf(hamil_file);

  std::shared_ptr<utils::RandomGenerator_t<>> rng = std::make_shared<utils::RandomGenerator_t<>>();
  std::shared_ptr<utils::RandomGenerator_t<MEM>> rng_dev = std::make_shared<utils::RandomGenerator_t<MEM>>(777);

  Hamiltonian ham = Hamiltonian::from_params(mpi, HamiltonianParameters{.name = "ham0", .filename = hamil_file});

  WALKER_TYPES type = afqmc::getWalkerType(wfn_file);
  REQUIRE(type == COLLINEAR);
  const WalkerSetParameters wlk_params{.name = "wset0", .walker_type = type};

  int const nwalk = 4;
  WavefunctionParameters wfn_params{.name = "wfn0", .filename = wfn_file};
  apply_defaults(wfn_params, ham.getHamType());
  auto wfn = Wavefunction<MEM>::from_params(mpi, wfn_params, type, false, ham, nwalk);

  PropagatorParameters prop_params{.name = "prop0"};
  apply_defaults(prop_params, ham.getHamType());
  Propagator<MEM> prop{AFQMCBasePropagator<MEM>(prop_params, mpi, wfn, rng_dev)};
  prop.generateP1(1e-10, type);

  auto const& initial_guess = wfn.initial_guess();

  // the pair correlators are the one observable that needs a file of its own. The identity
  // map is enough to exercise it and the way its output is named after the map.
  utils::TemporaryDirectory tmpdir;
  std::string const map_file = (tmpdir / "orbital_map.h5").string();
  if(mpi->comm.root()) {
    nda::array<int, 2> identity(NMO, 1);
    for(int i = 0; i < NMO; i++) {
      identity(i, 0) = i;
    }
    h5::file file(map_file, 'w');
    h5::group orbital_map = h5::group(file).create_group("orbital_map");
    h5::write(orbital_map, "s", identity);
  }
  mpi->comm.barrier();

  // An observable is measured if and only if its parameter block is present, so this asks for
  // all of them, on whichever estimator parameters it is handed. Every hand-built block also
  // has to set what resolve_defaults would have filled in -- the wavefunction, the hamiltonian,
  // the measurement interval and, for back propagation, the orthogonalization interval --
  // because it never ran here.
  auto with_observables = [&](auto params) {
    params.onerdm      = OneRDMParameters{};
    params.diag_twordm = DiagonalTwoRDMParameters{};
    params.twordm      = TwoRDMParameters{};
    params.paircorr    = PairCorrParameters{.pairs = {.filename = map_file, .group = "orbital_map"}};
    params.spincorr    = SpinCorrParameters{};
    return params;
  };

  auto back_propagated = [&](std::vector<int> multipliers) {
    return with_observables(BackPropEstimatorParameters{
        .wavefunction                = "wfn0",
        .hamiltonian                 = "ham0",
        .measure_interval_multiplier = std::move(multipliers),
        .walker_ortho_interval       = afqmc::DEFAULT_WALKER_ORTHO_INTERVAL});
  };

  constexpr int pop_control_interval = afqmc::DEFAULT_POPULATION_CONTROL_INTERVAL;

  // the two writes have to see different bin counts for the append to mean anything, which
  // 4 blocks followed by 2 more gives for every estimator below
  constexpr long nblocks_first = 4;
  constexpr long nblocks_total = 6;

  // A back propagation estimator anchors at block -1 and re-anchors once bp_step reaches its
  // largest multiplier, so with {1, 2} the first block already measures Steps=2 and the
  // blocks after it alternate: Steps=2 on the odd ones, Steps=1 on the even ones.
  auto expect_back_propagated = [](std::map<std::string, long>& expected, std::string_view prefix,
                                   long nblocks) {
    expect_observables(expected, std::format("{}/Steps=1", prefix), nblocks / 2);
    expect_observables(expected, std::format("{}/Steps=2", prefix), (nblocks + 1) / 2);
  };

  // ---- energy, mixed and back propagation, sharing one walker set and one results file ----
  {
    const ExecuteParameters exec{
        .estimators = EstimatorParameters{
            .energy   = EnergyEstimatorParameters{.wavefunction                = "wfn0",
                                                  .hamiltonian                 = "ham0",
                                                  .measure_interval_multiplier = 1},
            .mixed    = with_observables(
                MixedEstimatorParameters{.wavefunction                = "wfn0",
                                         .hamiltonian                 = "ham0",
                                         .measure_interval_multiplier = 2}),
            .backprop = back_propagated({1, 2})},
        .population_control_interval = pop_control_interval,
        .n_walkers_per_mpi_task = nwalk};

    auto wset = WalkerSet<MEM>(mpi, rng, wlk_params, initial_guess, nwalk);
    Estimators<MEM> estimators{
      mpi, 0, exec, wset, wfn, prop,
      [&](std::string const&, std::string const&) -> Wavefunction<MEM>& { return wfn; }};
    // the PairCorr constructor was the last read of the temporary directory on every rank but
    // root, which is the one that deletes it
    mpi->comm.barrier();

    auto const results = tmpdir / "run_a.results.h5";

    auto expected_bins = [&](long nblocks) {
      std::map<std::string, long> expected;
      for(auto const* name : {"Energy", "OnebodyEnergy", "ExchangeEnergy", "CoulombEnergy", "Overlap"}) {
        expected[name] = nblocks;
      }
      expect_observables(expected, "MixedEstimator", nblocks / 2);
      expect_back_propagated(expected, "BackPropEstimator", nblocks);
      return expected;
    };

    // only root records any bin, and Estimators::write is not guarded, so root alone writes
    run_measurement_blocks(*mpi, estimators, wset, pop_control_interval, 1, nblocks_first);
    if(mpi->comm.root()) {
      estimators.write(results);
      check_bins(collect_bins(results), expected_bins(nblocks_first));
    }

    // a write flushes the complete bins and drops them, so the second one has to grow the
    // datasets the first one created rather than start over
    run_measurement_blocks(*mpi, estimators, wset, pop_control_interval, nblocks_first + 1, nblocks_total);
    if(mpi->comm.root()) {
      estimators.write(results);
      check_bins(collect_bins(results), expected_bins(nblocks_total));
    }
  }

  // ---- time evolved operators, on a walker set of their own: both back propagation
  // estimators reshape the back propagation buffers of the walker set in their constructor,
  // so they cannot share one ----
  {
    // the energy estimator is the one that is present by default, so removing it takes an
    // explicit null -- here it would only add datasets this case does not care about
    const ExecuteParameters exec{
        .estimators = EstimatorParameters{.energy          = std::nullopt,
                                          .time_evolved_bp = back_propagated({1, 2})},
        .population_control_interval = pop_control_interval,
        .n_walkers_per_mpi_task = nwalk};

    auto wset = WalkerSet<MEM>(mpi, rng, wlk_params, initial_guess, nwalk);
    Estimators<MEM> estimators{
      mpi, 0, exec, wset, wfn, prop,
      [&](std::string const&, std::string const&) -> Wavefunction<MEM>& { return wfn; }};
    mpi->comm.barrier();

    auto const results = tmpdir / "run_b.results.h5";

    auto expected_bins = [&](long nblocks) {
      std::map<std::string, long> expected;
      expect_back_propagated(expected, "TimeEvolvedBP", nblocks);
      return expected;
    };

    run_measurement_blocks(*mpi, estimators, wset, pop_control_interval, 1, nblocks_first);
    if(mpi->comm.root()) {
      estimators.write(results);
      check_bins(collect_bins(results), expected_bins(nblocks_first));
    }

    run_measurement_blocks(*mpi, estimators, wset, pop_control_interval, nblocks_first + 1, nblocks_total);
    if(mpi->comm.root()) {
      estimators.write(results);
      check_bins(collect_bins(results), expected_bins(nblocks_total));
    }
  }
}

TEST_CASE("estimators: all estimators write results", "[estimators]")
{
  auto& mpi = utils::make_unit_test_mpi_context();

  utils::catch_test_exceptions("estimators: all estimators write results", [&] {
    estimators_all_observables<HOST_MEMORY>(mpi);
#if defined(ENABLE_DEVICE)
    estimators_all_observables<DEVICE_MEMORY>(mpi);
#endif
  });
}

/// With local energy importance sampling the propagator evaluates the local energy itself and
/// leaves its components on the walkers, so the energy estimator reads them off instead of
/// recomputing them. That shortcut has to produce what the recomputation it replaces would.
template<MEMORY_SPACE MEM>
void estimators_local_energy_matches_recomputation(
    std::shared_ptr<utils::mpi_context_t<boost::mpi3::communicator>> mpi)
{
  std::string const inputs     = utils::unit_test_base() + "BH/";
  std::string const hamil_file = inputs + "afqmc_H_rhf_collinear.h5";
  std::string const wfn_file   = inputs + "afqmc_uhf_nomsd.h5";

  std::shared_ptr<utils::RandomGenerator_t<>> rng = std::make_shared<utils::RandomGenerator_t<>>();
  std::shared_ptr<utils::RandomGenerator_t<MEM>> rng_dev = std::make_shared<utils::RandomGenerator_t<MEM>>(777);

  Hamiltonian ham = Hamiltonian::from_params(mpi, HamiltonianParameters{.name = "ham0", .filename = hamil_file});

  WALKER_TYPES type = afqmc::getWalkerType(wfn_file);
  const WalkerSetParameters wlk_params{.name = "wset0", .walker_type = type};

  int const nwalk = 4;
  WavefunctionParameters wfn_params{.name = "wfn0", .filename = wfn_file};
  apply_defaults(wfn_params, ham.getHamType());
  auto wfn = Wavefunction<MEM>::from_params(mpi, wfn_params, type, false, ham, nwalk);

  // hybrid propagation would leave nothing but the overlap on the walkers, so the shortcut
  // this exercises hangs off hybrid being false
  PropagatorParameters prop_params{.name = "prop0", .hybrid = false};
  apply_defaults(prop_params, ham.getHamType());
  Propagator<MEM> prop{AFQMCBasePropagator<MEM>(prop_params, mpi, wfn, rng_dev)};

  constexpr double dt                = 0.005;
  constexpr int pop_control_interval = afqmc::DEFAULT_POPULATION_CONTROL_INTERVAL;

  auto wset = WalkerSet<MEM>(mpi, rng, wlk_params, wfn.initial_guess(), nwalk);

  // a single step is enough, and it has to be a real one: the local energy only lands on the
  // walkers when the propagator actually runs its local energy update
  prop.generateP1(dt, type);
  prop.Propagate(wset, 0.0, dt);

  const ExecuteParameters exec{
      .estimators = EstimatorParameters{
          .energy = EnergyEstimatorParameters{.wavefunction                = "wfn0",
                                              .hamiltonian                 = "ham0",
                                              .measure_interval_multiplier = 1}},
      .population_control_interval = pop_control_interval,
      .n_walkers_per_mpi_task = nwalk};

  Estimators<MEM> estimators{
      mpi, 0, exec, wset, wfn, prop,
      [&](std::string const&, std::string const&) -> Wavefunction<MEM>& { return wfn; }};
  estimators.measure(*mpi, 1, wset);

  // the recomputation, averaged the way MeasurementOutput averages: weighted, reduced over the
  // ranks and divided by the summed weight
  int const nw = wset.size();
  memory::buffered_array<MEM,ComplexType,1> weights(nw);
  wset.getProperty(WEIGHT, weights);
  memory::buffered_array<MEM,ComplexType,2> localEnergy(nw,3);
  memory::buffered_array<MEM,ComplexType,1> ovlp(nw);
  wfn.Energy(wset, localEnergy, ovlp, wset.getTauStep());

  auto weights_h     = nda::to_host(weights);
  auto localEnergy_h = nda::to_host(localEnergy);
  auto ovlp_h        = nda::to_host(ovlp);

  nda::array<ComplexType,1> expected(5);
  expected() = 0.0;
  for(int iw = 0; iw < nw; ++iw) {
    for(int k = 0; k < 3; ++k) {
      expected(k + 1) += weights_h(iw) * localEnergy_h(iw, k);
    }
    expected(4) += weights_h(iw) * ovlp_h(iw);
  }
  expected(0) = expected(1) + expected(2) + expected(3);

  ComplexType denominator = nda::sum(weights_h);
  for(int k = 0; k < expected.size(); ++k) {
    expected(k) = mpi->comm.reduce_value(expected(k));
  }
  denominator = mpi->comm.reduce_value(denominator);

  // only root records any bin, and Estimators::write is not guarded, so root alone writes
  utils::TemporaryDirectory tmpdir;
  if(mpi->comm.root()) {
    auto const results = tmpdir / "local_energy.results.h5";
    estimators.write(results);

    h5::file out(results.string(), 'r');
    h5::group root(out);

    constexpr std::array<char const*,5> names{"Energy", "OnebodyEnergy", "ExchangeEnergy",
                                              "CoulombEnergy", "Overlap"};
    for(int k = 0; k < int(names.size()); ++k) {
      nda::array<ComplexType,1> bins;
      h5::read(root, std::format("Measurements/Stage0/{}/bins", names[k]), bins);
      REQUIRE(bins.extent(0) == 1);
      // the two paths evaluate the energy on either side of the walker update, so they agree
      // to roundoff rather than exactly
      CHECK_THAT(bins(0), utils::Approx(expected(k) / denominator, 1e-10, 1e-10));
    }
  }
}

TEST_CASE("estimators: local energy is read off the walkers", "[estimators]")
{
  auto& mpi = utils::make_unit_test_mpi_context();

  utils::catch_test_exceptions("estimators: local energy is read off the walkers", [&] {
    estimators_local_energy_matches_recomputation<HOST_MEMORY>(mpi);
#if defined(ENABLE_DEVICE)
    estimators_local_energy_matches_recomputation<DEVICE_MEMORY>(mpi);
#endif
  });
}

} // namespace sfqmc
