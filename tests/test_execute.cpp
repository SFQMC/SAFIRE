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
  WalkerSetParameters wlk_min{};

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
  params.driver        = driver;
  params.seed          = 463; // fix the seed so the test is reproducible
  params.hamiltonians  = {HamiltonianParameters{.name = "ham0", .filename = hamil_file}};
  params.wavefunctions = {WavefunctionParameters{.name = "wfn0", .filename = wfn_file}};
  params.propagators   = {PropagatorParameters{.name = "prop0"}};
  params.walker_sets   = {wlk_full};

  // label of each execute block, in the order they are added
  std::vector<std::string> scenarios;
  auto add = [&](std::string label, ExecuteParameters exec) {
    exec.steps               = 10;
    exec.equilibration_steps = 0;
    scenarios.push_back(std::move(label));
    params.execute.push_back(std::move(exec));
  };

  if(default_walker) {
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
  add("wfn(inline)+ham(external)+wlk(external)",
      ExecuteParameters{.walker_set   = std::string{"wlk0"},
                        .wavefunction = wfn_min,
                        .hamiltonian  = std::string{"ham0"}});

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

/// A UHF trial whose walkers start from the RHF determinant rather than from the trial itself,
/// followed by a stage that carries the walkers over.
template<MEMORY_SPACE MEM>
void execute_from_other_wavefunction(std::shared_ptr<utils::mpi_context_t<boost::mpi3::communicator>> mpi)
{
  std::string const pre = utils::unit_test_base();

  AFQMCParameters params{};
  params.seed          = 463;
  params.hamiltonians  = {HamiltonianParameters{.name = "ham", .filename = pre + "BH/afqmc_H_rhf_collinear.h5"}};
  params.wavefunctions = {WavefunctionParameters{.name = "uhf", .filename = pre + "BH/afqmc_uhf_nomsd.h5"},
                          WavefunctionParameters{.name = "rhf", .filename = pre + "BH/afqmc_rhf_nomsd.h5"}};
  for(int stage = 0; stage < 2; ++stage) {
    params.execute.push_back(ExecuteParameters{.wavefunction = std::string{"uhf"},
                                               .hamiltonian  = std::string{"ham"},
                                               .steps = 10, .equilibration_steps = 0});
  }
  params.execute[0].walker_set =
      WalkerSetParameters{.walker_type = COLLINEAR, .from = WavefunctionSource{.wavefunction = std::string{"rhf"}}};

  utils::TemporaryDirectory tmpdir;
  params.output_name = (tmpdir / "exec_from_test").string();

  resolve_defaults(params, *mpi);
  REQUIRE(params.walker_sets.size() == 1);
  CHECK(block_name(std::get<WavefunctionSource>(*params.walker_sets[0].from).wavefunction, "wavefunction") == "rhf");

  execute_simulation<MEM>(mpi, params);

  if(mpi->comm.root()) {
    h5::file results(std::format("{}.results.h5", params.output_name), 'r');
    h5::group meas = h5::group(results).open_group("Measurements");
    CHECK(meas.has_subgroup("Stage0"));
    CHECK(meas.has_subgroup("Stage1"));
  }
  mpi->comm.barrier();
}

TEST_CASE("execute: walker set from another wavefunction", "[execute]")
{
  auto& mpi = utils::make_unit_test_mpi_context();

  execute_from_other_wavefunction<HOST_MEMORY>(mpi);
#if defined(ENABLE_DEVICE)
  execute_from_other_wavefunction<DEVICE_MEMORY>(mpi);
#endif
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

/// The name of the wavefunction a resolved walker set is initialized from.
std::string source_wavefunction(const WalkerSetParameters& walker_set)
{
  REQUIRE(walker_set.from.has_value());
  const auto* source = std::get_if<WavefunctionSource>(&*walker_set.from);
  REQUIRE(source != nullptr);
  const auto* name = std::get_if<std::string>(&source->wavefunction);
  REQUIRE(name != nullptr);
  return *name;
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
  // the minimal input: one nameless wavefunction, one nameless hamiltonian, and nothing else
  {
    AFQMCParameters params{};
    params.execute = {ExecuteParameters{.wavefunction = WavefunctionParameters{.filename = wfn_file},
                                        .hamiltonian  = HamiltonianParameters{.filename = hamil_file}}};
    resolve_defaults(params, *mpi);

    // the absent blocks are materialized, one of each, and the registries name them uniquely
    REQUIRE(params.wavefunctions.size() == 1);
    REQUIRE(params.hamiltonians.size() == 1);
    REQUIRE(params.walker_sets.size() == 1);
    REQUIRE(params.propagators.size() == 1);
    check_unique_names(params.wavefunctions);
    check_unique_names(params.hamiltonians);
    check_unique_names(params.walker_sets);
    check_unique_names(params.propagators);

    // the execute block refers to all of them by name
    const ExecuteParameters& exec = params.execute[0];
    const std::string wfn_name    = resolved_name(exec.wavefunction, params.wavefunctions);
    const std::string ham_name    = resolved_name(exec.hamiltonian, params.hamiltonians);
    const std::string prop_name   = resolved_name(exec.propagator, params.propagators);
    const std::string wlk_name    = resolved_name(exec.walker_set, params.walker_sets);

    // the default walker set starts from the wavefunction of the execute block
    CHECK(source_wavefunction(block_named(params.walker_sets, wlk_name)) == wfn_name);

    // whatever the hamiltonian type is, the defaults that depend on it are filled in, so that no
    // consumer ever sees an empty optional
    const WavefunctionParameters& wfn = block_named(params.wavefunctions, wfn_name);
    CHECK(wfn.algorithm.has_value());
    CHECK(wfn.dense_trial.has_value());
    const PropagatorParameters& prop = block_named(params.propagators, prop_name);
    CHECK(prop.vbias_bound.has_value());
    CHECK(prop.upper_cutoff_scale.has_value());
    CHECK(prop.lower_cutoff_scale.has_value());
    CHECK(prop.denseP2.has_value());

    // the energy is the estimator that is present by default, and it is resolved to the
    // driver's own blocks; nothing else is measured unless the input asks for it
    const EstimatorParameters& estimators = exec.estimators;
    REQUIRE(estimators.energy.has_value());
    CHECK(estimators.energy->wavefunction == wfn_name);
    CHECK(estimators.energy->hamiltonian == ham_name);
    CHECK(estimators.energy->measure_interval == exec.measure_interval);
    CHECK(exec.measure_interval == afqmc::DEFAULT_MEASURE_INTERVAL);
    CHECK(!estimators.mixed.has_value());
    CHECK(!estimators.backprop.has_value());
    CHECK(!estimators.time_evolved_bp.has_value());
  }

  // what the input sets is never replaced by a default, and a block declared inside an execute
  // block keeps its values when it is hoisted
  {
    AFQMCParameters params{};
    params.wavefunctions = {WavefunctionParameters{.name = "estimator_wfn", .filename = wfn_file}};
    params.hamiltonians  = {HamiltonianParameters{.name = "estimator_ham", .filename = hamil_file}};
    params.execute       = {ExecuteParameters{
             .walker_set   = WalkerSetParameters{.walker_type = CLOSED},
             .wavefunction = WavefunctionParameters{.filename    = wfn_file,
                                                   .algorithm   = PHMSDEnergyAlgorithm::woodbury,
                                                   .dense_trial = false},
             .hamiltonian  = HamiltonianParameters{.filename = hamil_file},
             .propagator   = PropagatorParameters{.vbias_bound        = 12.5,
                                                  .upper_cutoff_scale = 3.5,
                                                  .lower_cutoff_scale = 0.25,
                                                  .denseP2            = false},
             .estimators = EstimatorParameters{
                 .energy = EnergyEstimatorParameters{.measure_interval = 5},
                 .mixed  = MixedEstimatorParameters{.wavefunction = "estimator_wfn",
                                                    .hamiltonian  = "estimator_ham"},
                 .backprop =
                     BackPropEstimatorParameters{.propagation_steps = std::vector<int>{3}}},
             .measure_interval = 7,
    }};
    resolve_defaults(params, *mpi);

    const ExecuteParameters& exec = params.execute[0];
    const std::string wlk_name    = resolved_name(exec.walker_set, params.walker_sets);
    const std::string wfn_name    = resolved_name(exec.wavefunction, params.wavefunctions);
    const std::string ham_name    = resolved_name(exec.hamiltonian, params.hamiltonians);
    const std::string prop_name   = resolved_name(exec.propagator, params.propagators);

    const WalkerSetParameters& wlk = block_named(params.walker_sets, wlk_name);
    CHECK(wlk.walker_type == CLOSED);

    const WavefunctionParameters& wfn = block_named(params.wavefunctions, wfn_name);
    CHECK(wfn.algorithm == PHMSDEnergyAlgorithm::woodbury);
    CHECK(wfn.dense_trial == false);

    const PropagatorParameters& prop = block_named(params.propagators, prop_name);
    CHECK(prop.vbias_bound == 12.5);
    CHECK(prop.upper_cutoff_scale == 3.5);
    CHECK(prop.lower_cutoff_scale == 0.25);
    CHECK(prop.denseP2 == false);

    // an estimator keeps the blocks and the intervals it brings itself, and inherits the ones
    // it leaves out from the execute block around it
    const EstimatorParameters& estimators = exec.estimators;
    REQUIRE(estimators.energy.has_value());
    CHECK(estimators.energy->measure_interval == 5);

    REQUIRE(estimators.mixed.has_value());
    CHECK(estimators.mixed->wavefunction == "estimator_wfn");
    CHECK(estimators.mixed->hamiltonian == "estimator_ham");
    CHECK(estimators.mixed->measure_interval == exec.measure_interval);

    REQUIRE(estimators.backprop.has_value());
    CHECK(estimators.backprop->propagation_steps == std::vector<int>{3});
    CHECK(estimators.backprop->walker_ortho_interval == exec.walker_ortho_interval);
    CHECK(estimators.backprop->wavefunction == wfn_name);
    CHECK(estimators.backprop->hamiltonian == ham_name);
  }

  // only one back propagation estimator may be defined at once
  {
    AFQMCParameters params{};
    params.execute = {ExecuteParameters{
        .wavefunction = WavefunctionParameters{.filename = wfn_file},
        .hamiltonian  = HamiltonianParameters{.filename = hamil_file},
        .estimators   =EstimatorParameters{.backprop        = BackPropEstimatorParameters{},
                                            .time_evolved_bp = BackPropEstimatorParameters{}},
    }};
    CHECK_THROWS_AS(resolve_defaults(params, *mpi), AppAbortException);
  }

  // a back propagation length is never inherited, so the estimator has to bring one
  {
    AFQMCParameters params{};
    params.execute = {ExecuteParameters{
        .wavefunction = WavefunctionParameters{.filename = wfn_file},
        .hamiltonian  = HamiltonianParameters{.filename = hamil_file},
        .estimators   =EstimatorParameters{.backprop = BackPropEstimatorParameters{}},
    }};
    CHECK_THROWS_AS(resolve_defaults(params, *mpi), AppAbortException);
  }

  // an empty list of back propagation lengths is rejected here rather than at construction
  {
    AFQMCParameters params{};
    params.execute = {ExecuteParameters{
        .wavefunction = WavefunctionParameters{.filename = wfn_file},
        .hamiltonian  = HamiltonianParameters{.filename = hamil_file},
        .estimators   =EstimatorParameters{.time_evolved_bp = BackPropEstimatorParameters{
                                                .propagation_steps = std::vector<int>{}}},
    }};
    CHECK_THROWS_AS(resolve_defaults(params, *mpi), AppAbortException);
  }

  // a generated name never takes one the input uses, not even one that only appears further down
  {
    AFQMCParameters params{};
    params.wavefunctions = {WavefunctionParameters{.name = "wavefunction_0", .filename = wfn_file}};
    params.execute       = {
        ExecuteParameters{.wavefunction = WavefunctionParameters{.filename = wfn_file},
                               .hamiltonian  = HamiltonianParameters{.filename = hamil_file}},
        ExecuteParameters{.wavefunction = std::string{"wavefunction_0"},
                               .hamiltonian  = HamiltonianParameters{.name     = "hamiltonian_0",
                                                                     .filename = hamil_file}},
    };
    resolve_defaults(params, *mpi);

    check_unique_names(params.wavefunctions);
    check_unique_names(params.hamiltonians);

    // the nameless blocks of the first execute block are named around the input's names, and the
    // reference by name of the second one still points at the block it was written for
    CHECK(resolved_name(params.execute[0].wavefunction, params.wavefunctions) != "wavefunction_0");
    CHECK(resolved_name(params.execute[0].hamiltonian, params.hamiltonians) != "hamiltonian_0");
    CHECK(resolved_name(params.execute[1].wavefunction, params.wavefunctions) == "wavefunction_0");
    CHECK(resolved_name(params.execute[1].hamiltonian, params.hamiltonians) == "hamiltonian_0");
  }

  // a later execute block without a walker set carries over the one of the stage before it, which
  // keeps the source it was given by the stage that introduced it
  {
    AFQMCParameters params{};
    params.wavefunctions = {WavefunctionParameters{.name = "second_wfn", .filename = wfn_file}};
    params.hamiltonians  = {HamiltonianParameters{.name = "ham", .filename = hamil_file}};
    params.execute       = {
        ExecuteParameters{.wavefunction = WavefunctionParameters{.filename = wfn_file},
                          .hamiltonian  = std::string{"ham"}},
        ExecuteParameters{.wavefunction = std::string{"second_wfn"}, .hamiltonian = std::string{"ham"}},
        ExecuteParameters{.wavefunction = std::string{"second_wfn"}, .hamiltonian = std::string{"ham"}},
    };
    resolve_defaults(params, *mpi);

    REQUIRE(params.walker_sets.size() == 1);
    const std::string wlk_name = resolved_name(params.execute[0].walker_set, params.walker_sets);
    CHECK(resolved_name(params.execute[1].walker_set, params.walker_sets) == wlk_name);
    CHECK(resolved_name(params.execute[2].walker_set, params.walker_sets) == wlk_name);
    CHECK(source_wavefunction(block_named(params.walker_sets, wlk_name)) ==
          resolved_name(params.execute[0].wavefunction, params.wavefunctions));
  }

  // so does an explicit walker set, while a later explicit one starts a walker set of its own, from
  // the wavefunction of the stage that introduces it
  {
    AFQMCParameters params{};
    params.wavefunctions = {WavefunctionParameters{.name = "first_wfn", .filename = wfn_file},
                            WavefunctionParameters{.name = "second_wfn", .filename = wfn_file}};
    params.hamiltonians  = {HamiltonianParameters{.name = "ham", .filename = hamil_file}};
    params.execute       = {
        ExecuteParameters{.walker_set   = WalkerSetParameters{.name = "first"},
                          .wavefunction = std::string{"first_wfn"},
                          .hamiltonian  = std::string{"ham"}},
        ExecuteParameters{.wavefunction = std::string{"second_wfn"}, .hamiltonian = std::string{"ham"}},
        ExecuteParameters{.walker_set   = WalkerSetParameters{.name = "second"},
                          .wavefunction = std::string{"second_wfn"},
                          .hamiltonian  = std::string{"ham"}},
    };
    resolve_defaults(params, *mpi);

    REQUIRE(params.walker_sets.size() == 2);
    CHECK(resolved_name(params.execute[1].walker_set, params.walker_sets) == "first");
    CHECK(resolved_name(params.execute[2].walker_set, params.walker_sets) == "second");
    CHECK(source_wavefunction(block_named(params.walker_sets, "first")) == "first_wfn");
    CHECK(source_wavefunction(block_named(params.walker_sets, "second")) == "second_wfn");
  }

  // a walker set may start from a wavefunction other than the one of the execute block, named or
  // declared inline, and that wavefunction has its defaults resolved as well
  {
    AFQMCParameters params{};
    params.wavefunctions = {WavefunctionParameters{.name = "init_wfn", .filename = wfn_file}};
    params.hamiltonians  = {HamiltonianParameters{.name = "ham", .filename = hamil_file}};
    params.execute       = {
        ExecuteParameters{
            .walker_set   = WalkerSetParameters{.from = WavefunctionSource{.wavefunction = std::string{"init_wfn"}}},
            .wavefunction = WavefunctionParameters{.filename = wfn_file},
            .hamiltonian  = std::string{"ham"}},
        ExecuteParameters{
            .walker_set   = WalkerSetParameters{.from = WavefunctionSource{
                                                    .wavefunction = WavefunctionParameters{.filename = wfn_file}}},
            .wavefunction = std::string{"init_wfn"},
            .hamiltonian  = std::string{"ham"}},
    };
    resolve_defaults(params, *mpi);

    REQUIRE(params.walker_sets.size() == 2);
    REQUIRE(params.wavefunctions.size() == 3);
    check_unique_names(params.wavefunctions);

    const std::string first  = resolved_name(params.execute[0].walker_set, params.walker_sets);
    const std::string second = resolved_name(params.execute[1].walker_set, params.walker_sets);
    CHECK(source_wavefunction(block_named(params.walker_sets, first)) == "init_wfn");

    // the inline source is hoisted into the registry, as a block of its own
    const std::string inline_source = source_wavefunction(block_named(params.walker_sets, second));
    CHECK(inline_source != "init_wfn");
    CHECK(inline_source != resolved_name(params.execute[0].wavefunction, params.wavefunctions));
    const WavefunctionParameters& source_wfn = block_named(params.wavefunctions, inline_source);
    CHECK(source_wfn.algorithm.has_value());
    CHECK(source_wfn.dense_trial.has_value());
  }

  // a walker set source is an object with exactly one known key
  {
    CHECK(nlohmann::json::parse(R"({"wavefunction": "wfn"})").get<WalkerSetSourceParameters>().index() == 0);
    CHECK_THROWS_AS(nlohmann::json::parse(R"({})").get<WalkerSetSourceParameters>(), AppAbortException);
    CHECK_THROWS_AS(nlohmann::json::parse(R"({"checkpoint": "stage0"})").get<WalkerSetSourceParameters>(),
                    AppAbortException);
    CHECK_THROWS_AS(nlohmann::json::parse(R"("wfn")").get<WalkerSetSourceParameters>(), AppAbortException);
  }

  // two blocks of the same kind cannot share a name
  {
    AFQMCParameters params{};
    params.wavefunctions = {WavefunctionParameters{.name = "wfn", .filename = wfn_file},
                            WavefunctionParameters{.name = "wfn", .filename = wfn_file}};
    params.execute       = {ExecuteParameters{.wavefunction = std::string{"wfn"}}};
    CHECK_THROWS_AS(resolve_defaults(params, *mpi), AppAbortException);
  }

  // an execute block cannot refer to a block that is not declared
  {
    AFQMCParameters params{};
    params.execute = {ExecuteParameters{.wavefunction = std::string{"nowhere"}}};
    CHECK_THROWS_AS(resolve_defaults(params, *mpi), AppAbortException);
  }

  // the hamiltonian is never taken from the file of the wavefunction: an execute block has to
  // give one, and it has to name its file
  {
    AFQMCParameters params{};
    params.execute = {ExecuteParameters{.wavefunction = WavefunctionParameters{.filename = hamil_file}}};
    CHECK_THROWS_AS(resolve_defaults(params, *mpi), AppAbortException);
  }
  {
    AFQMCParameters params{};
    params.execute = {ExecuteParameters{.wavefunction = WavefunctionParameters{.filename = hamil_file},
                                        .hamiltonian  = HamiltonianParameters{}}};
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
