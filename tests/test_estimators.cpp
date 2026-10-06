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
#include "AFQMC/Walkers/population_control.hpp"
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

/// Runs the estimator over nsteps propagation steps, the way the driver does. The walkers are
/// never propagated, so only the back propagation history position advances.
template<MEMORY_SPACE MEM>
void run_measurement_steps(utils::mpi_context_t<boost::mpi3::communicator>& mpi,
                           EstimatorBase<MEM>& estimator, Measurements& meas,
                           WalkerSet<MEM>& wset, long nsteps)
{
  for(long step = 0; step < nsteps; ++step) {
    wset.advanceHistoryPos();
    estimator.measure(mpi, step, meas, wset);
  }
}

/// Runs every estimator of the set over the steps [first, last). The walkers are never
/// propagated, so only the back propagation history position advances.
template<MEMORY_SPACE MEM>
void run_measurement_steps(utils::mpi_context_t<boost::mpi3::communicator>& mpi,
                           Estimators<MEM>& estimators, WalkerSet<MEM>& wset,
                           long first, long last)
{
  for(long step = first; step < last; ++step) {
    wset.advanceHistoryPos();
    estimators.measure(mpi, step, wset);
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
}
} // namespace

template<MEMORY_SPACE MEM>
void estimators_reduced_density_matrix(std::shared_ptr<utils::mpi_context_t<boost::mpi3::communicator>> mpi,
             std::string hamil_file, std::string wfn_file)
{
  auto [NMO, nup, ndown] = read_info_from_wfn(wfn_file, "any");

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
  // propagate with dt=0 so the BP RDM should match the mixed estimate
  // we cannot actually use exactly 0 because that changes the sparsity structure in model hamiltonians
  Propagator<MEM> prop{AFQMCBasePropagator<MEM>(prop_params, mpi, wfn, rng_dev, 1e-10)};

  auto const& initial_guess = wfn.initial_guess();
  REQUIRE(int(initial_guess.slater().size()) == nspin);
  REQUIRE(initial_guess.slater()[0].shape() == std::array<long,2>{npol*NMO,nup});
  auto wset = WalkerSet<MEM>(mpi, rng, wlk_params, initial_guess, nwalk);

  constexpr long nsteps = 8;

  // ---- Run 1: single back propagation length ----
  {
    const BackPropEstimatorParameters est_params{
        .propagation_steps     = std::vector<int>{2},
        .walker_ortho_interval = afqmc::DEFAULT_WALKER_ORTHO_INTERVAL,
        .onerdm                = OneRDMParameters{}};

    std::unique_ptr<EstimatorBase<MEM>> estimator = std::make_unique<BackPropEstimator<MEM>>(
        *mpi, est_params, wset, wfn, prop);

    // the anchor starts before step 0 and resets after 2 steps, so measurements land on the
    // odd steps 1, 3, 5, 7
    Measurements meas{};
    run_measurement_steps(*mpi, *estimator, meas, wset, nsteps);
    verify_bp_matches_mixed<MEM>(meas, "BackPropEstimator/Steps=2/OneRDM", 0,
                                 type, NMO, nup, ndown, wfn, wset);
  }

  // ---- Run 2: multiple back propagation lengths ----
  {
    const BackPropEstimatorParameters est_params{
        .propagation_steps     = std::vector<int>{1, 2, 3},
        .walker_ortho_interval = afqmc::DEFAULT_WALKER_ORTHO_INTERVAL,
        .onerdm                = OneRDMParameters{}};

    std::unique_ptr<EstimatorBase<MEM>> estimator = std::make_unique<BackPropEstimator<MEM>>(
        *mpi, est_params, wset, wfn, prop);

    // the anchor resets after 3 steps, so Steps=2 is measured on steps 1, 4 and 7
    Measurements meas{};
    run_measurement_steps(*mpi, *estimator, meas, wset, nsteps);
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

  int const NMO = std::get<0>(read_info_from_wfn(wfn_file, "any"));

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
  Propagator<MEM> prop{AFQMCBasePropagator<MEM>(prop_params, mpi, wfn, rng_dev, 1e-10)};

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

  // the windows here are one and two steps long, so the walkers have to be orthogonalized
  // every step: the time evolved operators are regularized with the walker Slater matrix
  auto back_propagated = [&](std::vector<int> steps) {
    return with_observables(BackPropEstimatorParameters{.wavefunction          = "wfn0",
                                                        .hamiltonian           = "ham0",
                                                        .propagation_steps     = std::move(steps),
                                                        .walker_ortho_interval = 1});
  };

  constexpr int pop_control_interval = afqmc::DEFAULT_POPULATION_CONTROL_INTERVAL;

  // the two writes have to see different bin counts for the append to mean anything, which
  // 4 steps followed by 2 more gives for every estimator below
  constexpr long nsteps_first = 4;
  constexpr long nsteps_total = 6;

  // A back propagation estimator anchors at step 0 and re-anchors once bp_step reaches its
  // longest window, so with {1, 2} it measures Steps=1 on the odd steps and Steps=2 on the
  // even ones.
  auto expect_back_propagated = [](std::map<std::string, long>& expected, std::string_view prefix,
                                   long nsteps) {
    expect_observables(expected, std::format("{}/Steps=1", prefix), (nsteps + 1) / 2);
    expect_observables(expected, std::format("{}/Steps=2", prefix), nsteps / 2);
  };

  // ---- energy, mixed and back propagation, sharing one walker set and one results file ----
  {
    const ExecuteParameters exec{
        .estimators = EstimatorParameters{
            .energy   = EnergyEstimatorParameters{.wavefunction     = "wfn0",
                                                  .hamiltonian      = "ham0",
                                                  .measure_interval = 1},
            .mixed    = with_observables(
                MixedEstimatorParameters{.wavefunction     = "wfn0",
                                         .hamiltonian      = "ham0",
                                         .measure_interval = 2}),
            .backprop = back_propagated({1, 2})},
        // deliberately not the default and no divisor of any interval above: no measurement
        // schedule may depend on the population control interval any more
        .population_control_interval = 3,
        .num_walkers = nwalk * mpi->comm.size()};

    auto wset = WalkerSet<MEM>(mpi, rng, wlk_params, initial_guess, nwalk);
    Estimators<MEM> estimators{
      mpi, 0, exec, wset, wfn, prop,
      [&](std::string const&, std::string const&) -> Wavefunction<MEM>& { return wfn; }};
    // the PairCorr constructor was the last read of the temporary directory on every rank but
    // root, which is the one that deletes it
    mpi->comm.barrier();

    auto const results = tmpdir / "run_a.results.h5";

    auto expected_bins = [&](long nsteps) {
      std::map<std::string, long> expected;
      for(auto const* name : {"Energy", "OnebodyEnergy", "ExchangeEnergy", "CoulombEnergy", "Overlap",
                              "ProjectionTime", "EffectiveNumWalkers", "Phase", "TotalWeight"}) {
        expected[name] = nsteps;
      }
      expect_observables(expected, "MixedEstimator", nsteps / 2);
      expect_back_propagated(expected, "BackPropEstimator", nsteps);
      return expected;
    };

    // only root records any bin, and Estimators::write is not guarded, so root alone writes
    run_measurement_steps(*mpi, estimators, wset, 0, nsteps_first);
    if(mpi->comm.root()) {
      estimators.write(results);
      check_bins(collect_bins(results), expected_bins(nsteps_first));
    }

    // a write flushes the complete bins and drops them, so the second one has to grow the
    // datasets the first one created rather than start over
    run_measurement_steps(*mpi, estimators, wset, nsteps_first, nsteps_total);
    if(mpi->comm.root()) {
      estimators.write(results);
      check_bins(collect_bins(results), expected_bins(nsteps_total));
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
        .num_walkers = nwalk * mpi->comm.size()};

    auto wset = WalkerSet<MEM>(mpi, rng, wlk_params, initial_guess, nwalk);
    Estimators<MEM> estimators{
      mpi, 0, exec, wset, wfn, prop,
      [&](std::string const&, std::string const&) -> Wavefunction<MEM>& { return wfn; }};
    mpi->comm.barrier();

    auto const results = tmpdir / "run_b.results.h5";

    auto expected_bins = [&](long nsteps) {
      std::map<std::string, long> expected;
      expect_back_propagated(expected, "TimeEvolvedBP", nsteps);
      return expected;
    };

    run_measurement_steps(*mpi, estimators, wset, 0, nsteps_first);
    if(mpi->comm.root()) {
      estimators.write(results);
      check_bins(collect_bins(results), expected_bins(nsteps_first));
    }

    run_measurement_steps(*mpi, estimators, wset, nsteps_first, nsteps_total);
    if(mpi->comm.root()) {
      estimators.write(results);
      check_bins(collect_bins(results), expected_bins(nsteps_total));
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
  constexpr double dt                = 0.005;
  constexpr int pop_control_interval = afqmc::DEFAULT_POPULATION_CONTROL_INTERVAL;

  PropagatorParameters prop_params{.name = "prop0", .hybrid = false};
  apply_defaults(prop_params, ham.getHamType());
  Propagator<MEM> prop{AFQMCBasePropagator<MEM>(prop_params, mpi, wfn, rng_dev, dt)};

  auto wset = WalkerSet<MEM>(mpi, rng, wlk_params, wfn.initial_guess(), nwalk);

  // a single step is enough, and it has to be a real one: the local energy only lands on the
  // walkers when the propagator actually runs its local energy update
  prop.Propagate(wset, 0.0);

  const ExecuteParameters exec{
      .estimators = EstimatorParameters{
          .energy = EnergyEstimatorParameters{.wavefunction     = "wfn0",
                                              .hamiltonian      = "ham0",
                                              .measure_interval = 1}},
      .population_control_interval = pop_control_interval,
      .timestep = dt,
      .num_walkers = nwalk * mpi->comm.size()};

  Estimators<MEM> estimators{
      mpi, 0, exec, wset, wfn, prop,
      [&](std::string const&, std::string const&) -> Wavefunction<MEM>& { return wfn; }};
  constexpr long step = 3;
  estimators.measure(*mpi, step, wset);

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
    // Energy returns the log overlap
    expected(4) += weights_h(iw) * std::exp(ovlp_h(iw));
  }
  expected(0) = expected(1) + expected(2) + expected(3);

  ComplexType denominator = nda::sum(weights_h);
  for(int k = 0; k < expected.size(); ++k) {
    expected(k) = mpi->comm.reduce_value(expected(k));
  }
  denominator = mpi->comm.reduce_value(denominator);

  RealType const abs_sum = mpi->comm.reduce_value(nda::sum(nda::abs(weights_h)));
  RealType const abs2_sum = mpi->comm.reduce_value(nda::sum(nda::abs2(weights_h)));

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

    auto read_bin = [&]<typename T>(std::string_view name, T) {
      nda::array<T,1> bins;
      h5::read(root, std::format("Measurements/Stage0/{}/bins", name), bins);
      REQUIRE(bins.extent(0) == 1);
      return bins(0);
    };
    // measured after the propagation of the step, so step 3 has propagated for four timesteps
    CHECK_THAT(read_bin("ProjectionTime", RealType{}), utils::Approx((step + 1) * dt));
    CHECK_THAT(read_bin("EffectiveNumWalkers", RealType{}), utils::Approx(abs_sum * abs_sum / abs2_sum));
    CHECK_THAT(read_bin("Phase", ComplexType{}), utils::Approx(denominator / abs_sum));
    CHECK_THAT(read_bin("TotalWeight", ComplexType{}), utils::Approx(denominator));
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

namespace
{
/// One entry of a back propagation record, carrying the tag of the walker it belongs to.
/// `salt` separates the three records, so that a copy taken from the wrong one cannot pass.
ComplexType bp_record_entry(int salt, int tag, int a, int b)
{
  return ComplexType(tag + 0.125 * a + 8192.0 * salt, 0.5 - 0.0625 * b - 4096.0 * salt);
}

/// The walker tag a record entry written by bp_record_entry(0, tag, 0, 0) names.
int bp_record_tag(ComplexType entry) { return int(std::lround(entry.real())); }

/// Sets up a population whose weights straddle the branching thresholds, so that population
/// control has to duplicate a walker and kill another rather than pass the population through.
template<MEMORY_SPACE MEM>
void skew_weights(WalkerSet<MEM>& wset)
{
  int const nwalk = wset.size();
  memory::buffered_array<HOST_MEMORY, ComplexType, 1> weights(nwalk);
  wset.getProperty(WEIGHT, weights);
  weights(0) *= 1e-3;
  weights(nwalk - 1) *= 5.0;
  auto staged = memory::to_memory_space<MEM>(nda::array<ComplexType, 1>{weights()});
  wset.setProperty(WEIGHT, staged);
}
} // namespace

/// A back propagation window is a number of steps of its own now, so a branching event can
/// land anywhere inside one rather than only on its boundary. Everything the estimator reads
/// back over the window -- the field ring, the weight factors and the anchor Slater matrix --
/// lives on the walker, so it has to follow the walker that a branch duplicated or a load
/// balance moved, all of it and from the same parent. Each record here names the walker it
/// was written for, which is what lets a survivor be checked without knowing which parent it
/// came from. Rank 0 holds one walker more than the others, as an uneven split of the walkers
/// over the ranks leaves it, so that a run on several ranks exchanges between unequal ones.
template<MEMORY_SPACE MEM>
void estimators_bp_record_survives_population_control(
    std::shared_ptr<utils::mpi_context_t<boost::mpi3::communicator>> mpi,
    Wavefunction<MEM>& wfn, WalkerSetParameters const& wlk_params,
    std::shared_ptr<utils::RandomGenerator_t<>> rng, int nwalk)
{
  constexpr int nbp = 3;
  constexpr int nCV = 5;

  int const rank   = mpi->comm.rank();
  int const nlocal = nwalk + (rank == 0 ? 1 : 0);

  auto wset = WalkerSet<MEM>(mpi, rng, wlk_params, wfn.initial_guess(), nlocal);
  wset.resize_bp(nbp, nCV, 1);

  int const nhist = wset.HistoryBufferLength();
  int const tag0  = rank * nwalk + (rank > 0 ? 1 : 0);

  for(int s = 0; s < nbp; ++s) {
    nda::array<ComplexType, 2> slot(nlocal, nCV);
    for(int iw = 0; iw < nlocal; ++iw) {
      for(int c = 0; c < nCV; ++c) {
        slot(iw, c) = bp_record_entry(0, tag0 + iw, s, c);
      }
    }
    auto staged = memory::to_memory_space<MEM>(std::move(slot));
    wset.storeFields(s, staged);
  }
  {
    nda::array<ComplexType, 2> factors(nlocal, nhist);
    for(int iw = 0; iw < nlocal; ++iw) {
      for(int k = 0; k < nhist; ++k) {
        factors(iw, k) = bp_record_entry(1, tag0 + iw, 0, k);
      }
    }
    auto staged = memory::to_memory_space<MEM>(std::move(factors));
    auto weight_factors = wset.getWeightFactors();
    weight_factors() = staged();
  }
  {
    auto anchor = wset.SlaterMatricesN(Alpha);
    nda::array<ComplexType, 3> smn(anchor.shape());
    for(int iw = 0; iw < nlocal; ++iw) {
      for(int i = 0; i < smn.extent(1); ++i) {
        for(int j = 0; j < smn.extent(2); ++j) {
          smn(iw, i, j) = bp_record_entry(2, tag0 + iw, i, j);
        }
      }
    }
    auto staged = memory::to_memory_space<MEM>(std::move(smn));
    anchor() = staged();
  }

  skew_weights(wset);
  population_control(*mpi, *wset.getRNG(), wset);

  REQUIRE(wset.size() == nlocal);

  auto fields  = nda::to_host(wset.getFields());
  auto factors = nda::to_host(wset.getWeightFactors());
  auto anchor  = nda::to_host(wset.SlaterMatricesN(Alpha));

  // a branch that left every walker in its own slot would let a record that never moves pass,
  // so the run has to have moved at least one of them somewhere
  int moved = 0;
  for(int iw = 0; iw < nlocal; ++iw) {
    int const tag = bp_record_tag(fields(iw, 0, 0));
    if(tag != tag0 + iw) {
      ++moved;
    }

    for(int s = 0; s < nbp; ++s) {
      for(int c = 0; c < nCV; ++c) {
        CHECK_THAT(fields(iw, s, c), utils::Approx(bp_record_entry(0, tag, s, c)));
      }
    }
    for(int k = 0; k < nhist; ++k) {
      CHECK_THAT(factors(iw, k), utils::Approx(bp_record_entry(1, tag, 0, k)));
    }
    for(int i = 0; i < anchor.extent(1); ++i) {
      for(int j = 0; j < anchor.extent(2); ++j) {
        CHECK_THAT(anchor(iw, i, j), utils::Approx(bp_record_entry(2, tag, i, j)));
      }
    }
  }
  CHECK(mpi->comm.all_reduce_value(moved) > 0);
}

/// The back propagation window and the population control interval run on grids of their own,
/// so a window straddles branching events at a different offset every time it comes around.
/// The propagator runs at dt -> 0 here, where back propagation is the identity and the
/// estimator must reproduce the mixed estimate of the same step -- but only if it weights the
/// walkers the population control event inside its window left behind, and anchors on the step
/// its schedule says. The walkers are spread apart at a real timestep first, so that a weighted
/// average over them is not the same as any single one of them and a mixed-up weight cannot
/// cancel out.
template<MEMORY_SPACE MEM>
void estimators_bp_matches_mixed_across_population_control(
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

  PropagatorParameters prop_params{.name = "prop0"};
  apply_defaults(prop_params, ham.getHamType());
  // one propagator to spread the walkers apart before the measurement phase, and one to run
  // the measurement phase itself at a timestep small enough that replaying its fields
  // backwards leaves the references where they started.
  Propagator<MEM> spread{AFQMCBasePropagator<MEM>(prop_params, mpi, wfn, rng_dev, 0.01)};
  Propagator<MEM> flat{AFQMCBasePropagator<MEM>(prop_params, mpi, wfn, rng_dev, 1e-18)};

  estimators_bp_record_survives_population_control<MEM>(mpi, wfn, wlk_params, rng, nwalk);

  auto wset = WalkerSet<MEM>(mpi, rng, wlk_params, wfn.initial_guess(), nwalk);
  for(int i = 0; i < 4; ++i) {
    spread.Propagate(wset, 0.0);
  }
  spread.Orthogonalize(wset);

  constexpr int bp_window    = 3;
  // deliberately not a divisor of the back propagation window, and not divided by it either
  constexpr int pop_interval = 2;
  constexpr long nsteps      = 9;

  const ExecuteParameters exec{
      .estimators = EstimatorParameters{
          .energy   = std::nullopt,
          .mixed    = MixedEstimatorParameters{.wavefunction     = "wfn0",
                                               .hamiltonian      = "ham0",
                                               .measure_interval = 1,
                                               .onerdm           = OneRDMParameters{}},
          // path restoration undoes the cosine projection over the window, which is a factor
          // of its own and not what this compares; without it both estimators weight the
          // walkers with the weights the population control event left on them
          .backprop = BackPropEstimatorParameters{.wavefunction          = "wfn0",
                                                  .hamiltonian           = "ham0",
                                                  .propagation_steps     = std::vector<int>{bp_window},
                                                  .walker_ortho_interval = 1,
                                                  .path_restoration      = false,
                                                  .onerdm                = OneRDMParameters{}}},
      .population_control_interval = pop_interval,
      .num_walkers = nwalk * mpi->comm.size()};

  Estimators<MEM> estimators{
      mpi, 0, exec, wset, wfn, flat,
      [&](std::string const&, std::string const&) -> Wavefunction<MEM>& { return wfn; }};

  // the anchor the constructor took sits before step 0, so the windows close at steps 2, 5
  // and 8, while population control runs at steps 0, 2, 4, 6 and 8: both inside a window and
  // on its closing step, over the course of the run
  for(long step = 0; step < nsteps; ++step) {
    flat.Propagate(wset, 0.0);
    flat.Orthogonalize(wset);
    if(step % pop_interval == 0) {
      // re-skewing after a branch has equalized the weights makes the comb duplicate and drop
      // walkers again, so that later windows see a branching event too and not just the first one
      skew_weights(wset);
      population_control(*mpi, *wset.getRNG(), wset);
    }
    estimators.measure(*mpi, step, wset);
  }

  // only root records any bin, and Estimators::write is not guarded, so root alone writes
  utils::TemporaryDirectory tmpdir;
  if(mpi->comm.root()) {
    auto const results = tmpdir / "bp_vs_mixed.results.h5";
    estimators.write(results);

    h5::file out(results.string(), 'r');
    h5::group root(out);

    nda::array<ComplexType, 4> bp_bins, mixed_bins;
    h5::read(root, std::format("Measurements/Stage0/BackPropEstimator/Steps={}/OneRDM/bins", bp_window), bp_bins);
    h5::read(root, "Measurements/Stage0/MixedEstimator/OneRDM/bins", mixed_bins);

    REQUIRE(bp_bins.extent(0) == nsteps / bp_window);
    REQUIRE(mixed_bins.extent(0) == nsteps);

    for(long j = 0; j < bp_bins.extent(0); ++j) {
      // the back propagated bin j was measured at step bp_window*(j+1) - 1, which is also
      // its mixed bin: the mixed estimator measures every step, starting at step 0
      long const mixed = bp_window * (j + 1) - 1;
      CHECK_THAT(bp_bins(j, nda::ellipsis{}),
                 utils::Approx(mixed_bins(mixed, nda::ellipsis{}), 1e-7, 1e-7));
    }
  }
}

TEST_CASE("estimators: back propagation across population control", "[estimators]")
{
  auto& mpi = utils::make_unit_test_mpi_context();

  utils::catch_test_exceptions("estimators: back propagation across population control", [&] {
    estimators_bp_matches_mixed_across_population_control<HOST_MEMORY>(mpi);
#if defined(ENABLE_DEVICE)
    estimators_bp_matches_mixed_across_population_control<DEVICE_MEMORY>(mpi);
#endif
  });
}

} // namespace sfqmc
