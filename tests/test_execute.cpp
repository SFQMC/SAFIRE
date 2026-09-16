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
#include "AFQMC/parameters.hpp"
#include "AFQMC/parameter_defaults.hpp"
#include "utilities/Random.hpp"
#include "utilities/Timer.hpp"
#include "test_common.hpp"
#include "utilities/check.hpp"

#include <algorithm>
#include <format>
#include <optional>
#include <ranges>
#include <set>
#include <string>
#include <utility>
#include <variant>
#include <vector>
#include <complex>
#include <iomanip>

#include "nda/nda.hpp"
#include "nda/tensor.hpp"
#include "nda/h5.hpp"
#include "numerics/sparse/sparse.hpp"
  
#include "test_utils.hpp"
#include "AFQMC/Utilities/readWfn.h"

#include "AFQMC/execute.hpp"


extern std::string UTEST_HAMIL, UTEST_WFN;

namespace sfqmc
{
using namespace afqmc;

template<MEMORY_SPACE MEM>
void execute_build(std::shared_ptr<utils::mpi_context_t<boost::mpi3::communicator>> mpi,
             std::string hamil_file, std::string wfn_file,
             WALKER_TYPES walker_type = UNDEFINED_WALKER_TYPE, bool finiteT = false)
{
  WalkerSetParameters wlk_full{.name = "wlk0"};

  const WavefunctionParameters wfn_min{.filename = wfn_file};
  const HamiltonianParameters ham_min{.filename = hamil_file};
  WalkerSetParameters wlk_min{.max_weight = 4.0};

  // KE: Some special walker_types must match the wavefunction type;
  //     if an explicit walker type is provided to this test, use it!
  if(walker_type != UNDEFINED_WALKER_TYPE) {
    wlk_full.walker_type = walker_type;
    wlk_min.walker_type = walker_type;
  }

  const PropagatorParameters prop_min{.hybrid = true};

  const bool default_walker = (walker_type == UNDEFINED_WALKER_TYPE);
  // the scenarios that take the default walker type are ground state only, the rest follow the
  // walker type of the test files
  const DriverType driver = finiteT ? DriverType::ftafqmc : DriverType::afqmc;

  // The blocks an execute block can refer to by name, together with one execute block per
  // scenario. This is the shape of a real input, so the whole set is resolved at once, and the
  // scenarios run as consecutive stages of a single simulation.
  AFQMCParameters params{};
  params.driver       = driver;
  params.seed         = 463; // fix the seed so the test is reproducible
  params.hamiltonian  = {HamiltonianParameters{.name = "ham0", .filename = hamil_file}};
  params.wavefunction = {WavefunctionParameters{.name = "wfn0", .filename = wfn_file}};
  params.propagator   = {PropagatorParameters{.name = "prop0"}};
  params.walker_set   = {wlk_full};

  // label of each execute block, in the order they are added
  std::vector<std::string> scenarios;
  auto add = [&](std::string label, ExecuteParameters exec) {
    scenarios.push_back(std::move(label));
    params.execute.push_back(std::move(exec));
  };

  if(default_walker) {
    // wfn only - this is invalid unless wfn file and hamil file are the same
    if(hamil_file == wfn_file) {
      add("wfn only (inline)", ExecuteParameters{.wavefunction = wfn_min});
    }
    add("wfn+ham (inline)", ExecuteParameters{.wavefunction = wfn_min, .hamiltonian = ham_min});
    add("wfn+ham+prop (inline)",
        ExecuteParameters{.wavefunction = wfn_min, .hamiltonian = ham_min, .propagator = prop_min});
  }

  add("wfn+ham+prop+wlk (all inline)",
      ExecuteParameters{.walker_set   = wlk_min,
                        .wavefunction = wfn_min,
                        .hamiltonian  = ham_min,
                        .propagator   = prop_min});

  if(default_walker) {
    if(hamil_file == wfn_file) {
      add("wfn only (external)", ExecuteParameters{.wavefunction = std::string{"wfn0"}});
    }
    add("wfn+ham (external)",
        ExecuteParameters{.wavefunction = std::string{"wfn0"}, .hamiltonian = std::string{"ham0"}});
    add("wfn+ham+prop (external)",
        ExecuteParameters{.wavefunction = std::string{"wfn0"},
                          .hamiltonian  = std::string{"ham0"},
                          .propagator   = std::string{"prop0"}});
  }

  add("wfn+ham+prop+wlk (all external)",
      ExecuteParameters{.walker_set   = std::string{"wlk0"},
                        .wavefunction = std::string{"wfn0"},
                        .hamiltonian  = std::string{"ham0"},
                        .propagator   = std::string{"prop0"}});

  // mixed external internal
  if(hamil_file == wfn_file) {
    add("wfn(inline)+wlk(external)",
        ExecuteParameters{.walker_set = std::string{"wlk0"}, .wavefunction = wfn_min});
  }

  if(default_walker) {
    add("wfn(inline)+ham(external)",
        ExecuteParameters{.wavefunction = wfn_min, .hamiltonian = std::string{"ham0"}});
  }

  add("wfn(external)+ham(inline)+wlk(external)",
      ExecuteParameters{.walker_set   = std::string{"wlk0"},
                        .wavefunction = std::string{"wfn0"},
                        .hamiltonian  = ham_min});

  add("wfn(external)+ham(inline)+wlk(inline)",
      ExecuteParameters{.walker_set   = wlk_min,
                        .wavefunction = std::string{"wfn0"},
                        .hamiltonian  = ham_min});

  // many more possibilities (combinatorial...) Add any problematic ones if needed

  utils::TemporaryDirectory tmpdir;
  params.output_name = (tmpdir / "exec_test").string();

  resolve_defaults(params, *mpi);

  for(std::size_t i = 0; i < scenarios.size(); ++i) {
    app_log(0, "[execute] stage {}: {}; walker_type={}", i, scenarios[i], walkerTypeToString(walker_type));
  }

  // every stage of the run appends its own Stage<N> group to one results file
  execute_simulation<MEM>(mpi, params);

  if(mpi->comm.root()) {
    h5::file results(std::format("{}.results.h5", params.output_name), 'r');
    h5::group meas = h5::group(results).open_group("Measurements");
    for(std::size_t i = 0; i < scenarios.size(); ++i) {
      CHECK(meas.has_subgroup(std::format("Stage{}", i)));
    }
    CHECK(!meas.has_subgroup(std::format("Stage{}", scenarios.size())));
  }
  mpi->comm.barrier();
}

TEST_CASE("execute: build", "[execute]")
{
  auto& mpi = utils::make_unit_test_mpi_context();

  using namespace utils;

  run_test_with_files([&]<auto MEM>(std::string hamil_file, std::string wfn_file, WALKER_TYPES walker_type, bool finiteT) {
    execute_build<MEM>(mpi, hamil_file, wfn_file, walker_type, finiteT);
  }, UTEST_HAMIL, UTEST_WFN, TestFiles::RHF | TestFiles::UHF | TestFiles::GHF | TestFiles::NOMSD | TestFiles::FINITE_T | TestFiles::ALL_SYSTEMS);
}

/// The name an execute block refers a component by, checking that the reference was hoisted: it
/// holds a name, and that name belongs to exactly one block of the registry.
template<typename Params>
std::string resolved_name(const std::optional<utils::BlockRef<Params>>& ref, const std::vector<Params>& blocks)
{
  REQUIRE(ref.has_value());
  const auto* name = std::get_if<std::string>(&*ref);
  REQUIRE(name != nullptr);
  CHECK(!name->empty());
  CHECK(std::ranges::count_if(blocks, [&](const Params& block) { return block.name == *name; }) == 1);
  return *name;
}

template<typename Params>
const Params& block_named(const std::vector<Params>& blocks, const std::string& name)
{
  const auto block = std::ranges::find_if(blocks, [&](const Params& candidate) { return candidate.name == name; });
  REQUIRE(block != blocks.end());
  return *block;
}

/// Every block of a registry carries a name of its own.
template<typename Params>
void check_unique_names(const std::vector<Params>& blocks)
{
  std::set<std::string> names;
  for(const auto& block : blocks) {
    CHECK(!block.name.empty());
    CHECK(names.insert(block.name).second);
  }
}

// What resolve_defaults produces, without running a driver. The checks are on the behaviour it
// documents -- every block named, hoisted and resolved -- and not on the values of the defaults,
// which are free to change.
void parameter_defaults_resolution(std::shared_ptr<utils::mpi_context_t<boost::mpi3::communicator>> mpi,
                                   std::string hamil_file, std::string wfn_file)
{
  // the minimal input: one nameless wavefunction and nothing else. Its file has to hold the
  // hamiltonian too, because that is what the inherited hamiltonian is peeked from -- only the
  // hamiltonian file is read here, so hamil_file stands in for a single file holding both.
  {
    AFQMCParameters params{};
    params.execute = {ExecuteParameters{.wavefunction = WavefunctionParameters{.filename = hamil_file}}};
    resolve_defaults(params, *mpi);

    // the absent blocks are materialized, one of each, and the registries name them uniquely
    REQUIRE(params.wavefunction.size() == 1);
    REQUIRE(params.hamiltonian.size() == 1);
    REQUIRE(params.walker_set.size() == 1);
    REQUIRE(params.propagator.size() == 1);
    check_unique_names(params.wavefunction);
    check_unique_names(params.hamiltonian);
    check_unique_names(params.walker_set);
    check_unique_names(params.propagator);

    // the execute block refers to all of them by name
    const ExecuteParameters& exec = params.execute[0];
    const std::string wfn_name    = resolved_name(exec.wavefunction, params.wavefunction);
    const std::string ham_name    = resolved_name(exec.hamiltonian, params.hamiltonian);
    const std::string prop_name   = resolved_name(exec.propagator, params.propagator);
    resolved_name(exec.walker_set, params.walker_set); // nothing below needs the name itself

    // a hamiltonian without a file of its own takes the one of the wavefunction
    const WavefunctionParameters& wfn = block_named(params.wavefunction, wfn_name);
    CHECK(block_named(params.hamiltonian, ham_name).filename == wfn.filename);

    // whatever the hamiltonian type is, the defaults that depend on it are filled in, so that no
    // consumer ever sees an empty optional
    CHECK(wfn.algorithm.has_value());
    CHECK(wfn.dense_trial.has_value());
    const PropagatorParameters& prop = block_named(params.propagator, prop_name);
    CHECK(prop.vbias_bound.has_value());
    CHECK(prop.upper_cutoff_scale.has_value());
    CHECK(prop.lower_cutoff_scale.has_value());
    CHECK(prop.denseP2.has_value());
    CHECK(prop.symmetric_split.has_value());

    // the energy is the estimator that is present by default, and it is resolved to the
    // driver's own blocks; nothing else is measured unless the input asks for it
    const EstimatorParameters& estimators = exec.estimators;
    REQUIRE(estimators.energy.has_value());
    CHECK(estimators.energy->wavefunction == wfn_name);
    CHECK(estimators.energy->hamiltonian == ham_name);
    CHECK(estimators.energy->measure_interval_multiplier == exec.measure_interval_multiplier);
    CHECK(!estimators.mixed.has_value());
    CHECK(!estimators.backprop.has_value());
    CHECK(!estimators.time_evolved_bp.has_value());
  }

  // what the input sets is never replaced by a default, and a block declared inside an execute
  // block keeps its values when it is hoisted
  {
    AFQMCParameters params{};
    params.wavefunction = {WavefunctionParameters{.name = "estimator_wfn", .filename = wfn_file}};
    params.hamiltonian  = {HamiltonianParameters{.name = "estimator_ham", .filename = hamil_file}};
    params.execute      = {ExecuteParameters{
             .walker_set   = WalkerSetParameters{.min_weight = 0.125, .max_weight = 8.0},
             .wavefunction = WavefunctionParameters{.filename    = wfn_file,
                                                   .algorithm   = PHMSDEnergyAlgorithm::woodbury,
                                                   .dense_trial = false},
             .hamiltonian  = HamiltonianParameters{.filename = hamil_file},
             .propagator   = PropagatorParameters{.vbias_bound        = 12.5,
                                                  .upper_cutoff_scale = 3.5,
                                                  .lower_cutoff_scale = 0.25,
                                                  .denseP2            = false,
                                                  .symmetric_split    = false},
             .estimators = EstimatorParameters{
                 .energy = EnergyEstimatorParameters{.measure_interval_multiplier = 5},
                 .mixed  = MixedEstimatorParameters{.wavefunction = "estimator_wfn",
                                                    .hamiltonian  = "estimator_ham"},
                 .backprop =
                     BackPropEstimatorParameters{.measure_interval_multiplier = std::vector<int>{3}}},
             .measure_interval_multiplier = 7,
    }};
    resolve_defaults(params, *mpi);

    const ExecuteParameters& exec = params.execute[0];
    const std::string wlk_name    = resolved_name(exec.walker_set, params.walker_set);
    const std::string wfn_name    = resolved_name(exec.wavefunction, params.wavefunction);
    const std::string ham_name    = resolved_name(exec.hamiltonian, params.hamiltonian);
    const std::string prop_name   = resolved_name(exec.propagator, params.propagator);

    const WalkerSetParameters& wlk = block_named(params.walker_set, wlk_name);
    CHECK(wlk.min_weight == 0.125);
    CHECK(wlk.max_weight == 8.0);

    const WavefunctionParameters& wfn = block_named(params.wavefunction, wfn_name);
    CHECK(wfn.algorithm == PHMSDEnergyAlgorithm::woodbury);
    CHECK(wfn.dense_trial == false);

    const PropagatorParameters& prop = block_named(params.propagator, prop_name);
    CHECK(prop.vbias_bound == 12.5);
    CHECK(prop.upper_cutoff_scale == 3.5);
    CHECK(prop.lower_cutoff_scale == 0.25);
    CHECK(prop.denseP2 == false);
    CHECK(prop.symmetric_split == false);

    // a hamiltonian that names a file keeps it, rather than inheriting the one of the wavefunction
    CHECK(block_named(params.hamiltonian, ham_name).filename == hamil_file);

    // an estimator keeps the blocks and the intervals it brings itself, and inherits the ones
    // it leaves out from the execute block around it
    const EstimatorParameters& estimators = exec.estimators;
    REQUIRE(estimators.energy.has_value());
    CHECK(estimators.energy->measure_interval_multiplier == 5);

    REQUIRE(estimators.mixed.has_value());
    CHECK(estimators.mixed->wavefunction == "estimator_wfn");
    CHECK(estimators.mixed->hamiltonian == "estimator_ham");
    CHECK(estimators.mixed->measure_interval_multiplier == exec.measure_interval_multiplier);

    REQUIRE(estimators.backprop.has_value());
    CHECK(measure_interval_multipliers(*estimators.backprop) == std::vector<int>{3});
    CHECK(estimators.backprop->walker_ortho_interval == exec.walker_ortho_interval);
    CHECK(estimators.backprop->wavefunction == wfn_name);
    CHECK(estimators.backprop->hamiltonian == ham_name);
  }

  // only one back propagation estimator may be defined at once
  {
    AFQMCParameters params{};
    params.execute = {ExecuteParameters{
        .wavefunction = WavefunctionParameters{.filename = hamil_file},
        .estimators   = EstimatorParameters{.backprop        = BackPropEstimatorParameters{},
                                            .time_evolved_bp = BackPropEstimatorParameters{}},
    }};
    CHECK_THROWS_AS(resolve_defaults(params, *mpi), AppAbortException);
  }

  // a generated name never takes one the input uses, not even one that only appears further down
  {
    AFQMCParameters params{};
    params.wavefunction = {WavefunctionParameters{.name = "wavefunction_0", .filename = wfn_file}};
    params.execute      = {
        ExecuteParameters{.wavefunction = WavefunctionParameters{.filename = wfn_file},
                               .hamiltonian  = HamiltonianParameters{.filename = hamil_file}},
        ExecuteParameters{.wavefunction = std::string{"wavefunction_0"},
                               .hamiltonian  = HamiltonianParameters{.name     = "hamiltonian_0",
                                                                     .filename = hamil_file}},
    };
    resolve_defaults(params, *mpi);

    check_unique_names(params.wavefunction);
    check_unique_names(params.hamiltonian);

    // the nameless blocks of the first execute block are named around the input's names, and the
    // reference by name of the second one still points at the block it was written for
    CHECK(resolved_name(params.execute[0].wavefunction, params.wavefunction) != "wavefunction_0");
    CHECK(resolved_name(params.execute[0].hamiltonian, params.hamiltonian) != "hamiltonian_0");
    CHECK(resolved_name(params.execute[1].wavefunction, params.wavefunction) == "wavefunction_0");
    CHECK(resolved_name(params.execute[1].hamiltonian, params.hamiltonian) == "hamiltonian_0");
  }

  // two blocks of the same kind cannot share a name
  {
    AFQMCParameters params{};
    params.wavefunction = {WavefunctionParameters{.name = "wfn", .filename = wfn_file},
                           WavefunctionParameters{.name = "wfn", .filename = wfn_file}};
    params.execute      = {ExecuteParameters{.wavefunction = std::string{"wfn"}}};
    CHECK_THROWS_AS(resolve_defaults(params, *mpi), AppAbortException);
  }

  // an execute block cannot refer to a block that is not declared
  {
    AFQMCParameters params{};
    params.execute = {ExecuteParameters{.wavefunction = std::string{"nowhere"}}};
    CHECK_THROWS_AS(resolve_defaults(params, *mpi), AppAbortException);
  }
}

TEST_CASE("parameter_defaults: resolution", "[parameter_defaults]")
{
  auto& mpi = utils::make_unit_test_mpi_context();

  using namespace utils;

  // UHF covers all of the hamiltonian types the defaults switch on: RealDenseFactorized for the
  // molecules, ModelHamiltonian for the lattices, and KPFactorized/THC/KPTHC for the solids
  run_test_with_files([&]<auto MEM>(std::string hamil_file, std::string wfn_file, WALKER_TYPES, bool) {
    parameter_defaults_resolution(mpi, hamil_file, wfn_file);
  }, UTEST_HAMIL, UTEST_WFN, TestFiles::RHF | TestFiles::UHF | TestFiles::NOMSD | TestFiles::ALL_SYSTEMS);
}


} // namespace sfqmc
