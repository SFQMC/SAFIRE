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
////////////////////////////////////////////////////////////////////////////////

#include <algorithm>
#include <concepts>
#include <filesystem>
#include <format>
#include <map>
#include <random>
#include <set>
#include <string>
#include <string_view>
#include <type_traits>
#include <utility>
#include <vector>

#include "AFQMC/parameter_defaults.hpp"
#include "AFQMC/Hamiltonians/hdf5_helpers.hpp"
#include "IO/app_loggers.h"
#include "nda/h5.hpp"
#include "utilities/check.hpp"

namespace sfqmc::afqmc {

namespace {

/// Names the blocks of one component and hoists the ones declared inline into the top level list,
/// so that afterwards every one of `refs` refers to its block by name and `blocks` is the complete
/// registry. The refs are resolved in order, which is the order generated names are handed out in.
///
/// A generated name is only used if the input does not contain it, so that an input is free to
/// name a block e.g. "propagator_0" itself.
template<typename Params>
void resolve_block_refs(std::string_view key, std::vector<Params>& blocks,
                        const std::vector<utils::BlockRef<Params>*>& refs) {
  // collect every name the input gives explicitly. This has to see all of them before the first
  // name is generated, because an explicit name may appear in a later ref than the nameless block
  // that would otherwise be given it.
  std::set<std::string> names;
  auto claim = [&](const std::string& name) {
    utils::check(names.insert(name).second, "There is more than one {} named \"{}\". Names have to be unique.", key,
                 name);
  };
  for(const auto& block : blocks) {
    utils::check(!block.name.empty(), "A {} block outside of an execute block has to be named, so that an "
                 "execute block can refer to it.", key);
    claim(block.name);
  }
  for(const auto* ref : refs) {
    if(const auto* block = std::get_if<Params>(ref); block && !block->name.empty()) {
      claim(block->name);
    }
  }

  int counter = 0;
  auto generate_name = [&] {
    std::string name;
    do {
      name = std::format("{}_{}", key, counter++);
    } while(names.contains(name));
    names.insert(name);
    return name;
  };

  for(auto* ref : refs) {
    if(const auto* name = std::get_if<std::string>(ref)) {
      utils::check(names.contains(*name), "The input refers to the {} \"{}\", which is not declared anywhere.", key,
                   *name);
      continue;
    }

    Params block = std::get<Params>(std::move(*ref));
    if(block.name.empty()) {
      block.name = generate_name();
    }
    *ref = block.name;
    blocks.push_back(std::move(block));
  }
}

/// The refs to one component from every execute block. An absent one is materialized as a default
/// constructed block that nothing else can refer to, unless the component is required.
template<typename Params>
std::vector<utils::BlockRef<Params>*> execute_refs(std::string_view key, std::vector<ExecuteParameters>& execute,
                                                   std::optional<utils::BlockRef<Params>> ExecuteParameters::*member,
                                                   bool required) {
  std::vector<utils::BlockRef<Params>*> refs;
  for(auto& exec : execute) {
    auto& ref = exec.*member;
    if(!ref) {
      utils::check(!required, "An execute block is missing its {}, which is required.", key);
      ref.emplace(Params{});
    }
    refs.push_back(&*ref);
  }
  return refs;
}

/// Resolves the walker set of every execute block. Only the first execute block falls back to a
/// default walker set; a later one that names none carries over the walker set of the stage before
/// it, so that consecutive stages continue the same random walk.
void resolve_walker_set_refs(std::vector<WalkerSetParameters>& blocks, std::vector<ExecuteParameters>& execute) {
  std::vector<utils::BlockRef<WalkerSetParameters>*> refs;
  for(std::size_t i = 0; i < execute.size(); ++i) {
    auto& ref = execute[i].walker_set;
    if(!ref && i == 0) {
      ref.emplace(WalkerSetParameters{});
    }
    if(ref) {
      refs.push_back(&*ref);
    }
  }
  resolve_block_refs("walker_set", blocks, refs);

  for(std::size_t i = 1; i < execute.size(); ++i) {
    if(!execute[i].walker_set) {
      execute[i].walker_set = execute[i - 1].walker_set;
    }
  }
}

/// Applies `body` to every estimator the input requests. The blocks have different types, so this
/// is the one place that has to enumerate them.
template<typename Estimators, typename Body>
void for_each_estimator(Estimators& estimators, Body&& body) {
  auto visit = [&](auto& block) {
    if(block) {
      body(*block);
    }
  };
  visit(estimators.energy);
  visit(estimators.mixed);
  visit(estimators.backprop);
  visit(estimators.time_evolved_bp);
}

/// Fills the defaults one estimator block inherits from the execute block containing it.
void apply_estimator_defaults(auto& params, const ExecuteParameters& exec) {
  // an estimator may use a different wavefunction and hamiltonian than the driver
  if(!params.wavefunction) {
    params.wavefunction = block_name(exec.wavefunction, "wavefunction");
  }
  if(!params.hamiltonian) {
    params.hamiltonian = block_name(exec.hamiltonian, "hamiltonian");
  }

  if constexpr(requires { params.onerdm; params.paircorr; }) {
    if(params.onerdm && params.onerdm->rotation) {
      params.onerdm->rotation->filename = std::filesystem::absolute(params.onerdm->rotation->filename).string();
    }
    if(params.paircorr) {
      params.paircorr->pairs.filename = std::filesystem::absolute(params.paircorr->pairs.filename).string();
    }
  }

  if constexpr(std::same_as<std::remove_reference_t<decltype(params)>, BackPropEstimatorParameters>) {
    utils::check(params.propagation_steps && !params.propagation_steps->empty(),
                 "A back-propagation estimator requires a non-empty \"propagation_steps\", the back "
                 "propagation lengths in steps.");

    // back propagation retraces the forward propagation, so it orthogonalizes as often
    if(!params.walker_ortho_interval) {
      params.walker_ortho_interval = exec.walker_ortho_interval;
    }
  } else {
    if(!params.measure_interval) {
      params.measure_interval = resolved(exec.measure_interval, "measure_interval");
    }
    // the estimators gate on `step % measure_interval`, so zero is not merely meaningless
    utils::check(*params.measure_interval > 0,
                 "'measure_interval' must be positive, but it is {}.", *params.measure_interval);
  }
}

} // namespace

HamiltonianType peek_hamiltonian_type(const HamiltonianParameters& params,
                                      utils::mpi_context_t<mpi3::communicator>& mpi) {
  utils::check(!params.filename.empty(), "The hamiltonian \"{}\" must contain a filename.", params.name);

  int htype{};
  if(mpi.comm.root()) {
    h5::file file(params.filename, 'r');
    h5::group grp(file);
    htype = int(peekHamType(grp, get_hamiltonian_format(grp)));
  }
  mpi.comm.broadcast_n(&htype, 1, 0);
  return HamiltonianType(htype);
}

void apply_defaults(WavefunctionParameters& params, HamiltonianType htype) {
  // sparse trial wavefunctions are only worth it on k-point runs
  if(!params.dense_trial) {
    params.dense_trial = htype != HamiltonianType::kp_factorized && htype != HamiltonianType::kpthc;
  }
  if(!params.algorithm) {
    params.algorithm = (htype == HamiltonianType::real_dense_factorized ? PHMSDEnergyAlgorithm::woodbury
                                                                        : PHMSDEnergyAlgorithm::reference);
  }
}

void apply_defaults(PropagatorParameters& params, HamiltonianType htype) {
  // some defaults take legacy values for model hamiltonians
  const bool model = htype == HamiltonianType::model_hamiltonian;
  if(!params.vbias_bound) {
    params.vbias_bound = model ? 100.0 : 50.0;
  }
  if(!params.upper_cutoff_scale) {
    params.upper_cutoff_scale = model ? 50.0 : 10.0;
  }
  if(!params.lower_cutoff_scale) {
    params.lower_cutoff_scale = model ? 50.0 : 1.0;
  }
  if(!params.denseP2) {
    params.denseP2 = !model;
  }
}

void apply_defaults(EstimatorParameters& params, const ExecuteParameters& exec) {
  utils::check(!params.backprop || !params.time_evolved_bp,
               "Only one back propagation estimator may be defined at once, but the input has both "
               "\"backprop\" and \"time_evolved_bp\".");

  // the two back propagation estimators share a parameter struct, but only the retracing one
  // re-orthogonalizes. Warn before apply_estimator_defaults fills the value in anyway.
  if(params.time_evolved_bp && params.time_evolved_bp->walker_ortho_interval) {
    app_warning("A \"time_evolved_bp\" estimator does not re-orthogonalize, so the "
                "\"walker_ortho_interval\" of {} given for it is ignored.",
                *params.time_evolved_bp->walker_ortho_interval);
  }

  for_each_estimator(params, [&](auto& estimator) { apply_estimator_defaults(estimator, exec); });
}

void apply_defaults(ExecuteParameters& exec, DriverType driver) {
  if(!exec.measure_interval) {
    // the finite temperature driver takes one sample per sweep, so measuring every sweep is
    // the only default that makes sense there
    exec.measure_interval = (driver == DriverType::ftafqmc) ? 1 : DEFAULT_MEASURE_INTERVAL;
  }
  if(!exec.Eshift_relaxation_factor) {
    // Eshift relaxes once per step now, so the factor is set from the number of equilibration
    // steps: the gap to the average energy decays with a time constant of a tenth of the
    // equilibration phase, leaving e^-10 of it by the end
    if(exec.equilibration_steps > 1) {
      exec.Eshift_relaxation_factor = 1 - std::exp(-10.0 / exec.equilibration_steps);
    } else {
      exec.Eshift_relaxation_factor = 1;
    }
  }
  apply_defaults(exec.estimators, exec);
}

void resolve_defaults(AFQMCParameters& params, utils::mpi_context_t<mpi3::communicator>& mpi) {
  utils::check(!params.execute.empty(), "The input contains no execute block, so there is nothing to run.");

  // the walkers are split over the ranks, so their number has no default that suits every launch
  for(auto const& exec : params.execute) {
    utils::check(exec.num_walkers.has_value(),
                 "Every execute block needs \"num_walkers\", the total number of walkers over all ranks.");
    utils::check(*exec.num_walkers >= mpi.comm.size(),
                 "\"num_walkers\" ({}) has to be at least the number of MPI ranks ({}), so that every rank has a walker.",
                 *exec.num_walkers, mpi.comm.size());
  }

  // 1. draw a seed unless the input gave one, so that print_parameters reports a value that
  //    reproduces the run
  if(!params.seed) {
    int drawn{};
    if(mpi.comm.root()) {
      // the top bit goes so that the value stays non-negative and can be pasted back into the
      // input unchanged
      drawn = int(std::random_device{}() >> 1);
    }
    mpi.comm.broadcast_n(&drawn, 1, 0);
    params.seed = drawn;
  }

  // 2. + 3. name every block and hoist the ones declared inline. The walker sets go first, since
  //    the wavefunction a walker set is initialized from may be declared inside of it.
  resolve_walker_set_refs(params.walker_sets, params.execute);

  auto wavefunction_refs = execute_refs("wavefunction", params.execute, &ExecuteParameters::wavefunction, true);
  for(auto& walker_set : params.walker_sets) {
    if(walker_set.from) {
      if(auto* source = std::get_if<WavefunctionSource>(&*walker_set.from)) {
        wavefunction_refs.push_back(&source->wavefunction);
      }
    }
  }
  resolve_block_refs("wavefunction", params.wavefunctions, wavefunction_refs);
  resolve_block_refs("hamiltonian", params.hamiltonians,
                     execute_refs("hamiltonian", params.execute, &ExecuteParameters::hamiltonian, true));
  resolve_block_refs("propagator", params.propagators,
                     execute_refs("propagator", params.execute, &ExecuteParameters::propagator, false));

  // the files are made absolute, so that the printed parameters say which ones the run reads
  for(auto& wfn : params.wavefunctions) {
    utils::check(!wfn.filename.empty(), "The wavefunction \"{}\" must contain a filename.", wfn.name);
    wfn.filename = std::filesystem::absolute(wfn.filename).string();
  }
  for(auto& ham : params.hamiltonians) {
    utils::check(!ham.filename.empty(), "The hamiltonian \"{}\" must contain a filename.", ham.name);
    ham.filename = std::filesystem::absolute(ham.filename).string();
  }

  // 4. resolve what a block inherits from a neighbouring block
  for(auto& exec : params.execute) {
    const std::string& wfn_name = block_name(exec.wavefunction, "wavefunction");

    // a walker set starts from the wavefunction of the stage that introduces it. The stages are
    // visited in order, so a walker set carried over from an earlier stage already has its source.
    WalkerSetParameters& walker_set =
        find_block(params.walker_sets, block_name(exec.walker_set, "walker_set"), "walker_set");
    if(!walker_set.from) {
      walker_set.from = WavefunctionSource{.wavefunction = wfn_name};
    }

    apply_defaults(exec, params.driver);
  }

  // 5. resolve the defaults that depend on the hamiltonian type. Only the hamiltonians that are
  //    actually used are peeked, and each of them only once.
  std::map<std::string, HamiltonianType> htypes;
  auto hamiltonian_type = [&](const std::string& name) {
    auto entry = htypes.find(name);
    if(entry == htypes.end()) {
      entry = htypes.emplace(name, peek_hamiltonian_type(find_block(params.hamiltonians, name, "hamiltonian"), mpi)).first;
    }
    return entry->second;
  };

  std::set<std::string> introduced_walker_sets;
  for(const auto& exec : params.execute) {
    const HamiltonianType htype = hamiltonian_type(block_name(exec.hamiltonian, "hamiltonian"));
    apply_defaults(find_block(params.wavefunctions, block_name(exec.wavefunction, "wavefunction"), "wavefunction"),
                   htype);
    apply_defaults(find_block(params.propagators, block_name(exec.propagator, "propagator"), "propagator"), htype);

    // the wavefunction a walker set starts from is built with the hamiltonian of the stage that
    // introduces the walker set
    const std::string& walker_set_name = block_name(exec.walker_set, "walker_set");
    if(introduced_walker_sets.insert(walker_set_name).second) {
      const auto& from = *find_block(params.walker_sets, walker_set_name, "walker_set").from;
      if(const auto* source = std::get_if<WavefunctionSource>(&from)) {
        apply_defaults(find_block(params.wavefunctions, block_name(source->wavefunction, "wavefunction"),
                                  "wavefunction"),
                       htype);
      }
    }

    // an estimator that brings its own wavefunction builds it from its own hamiltonian
    for_each_estimator(exec.estimators, [&](const auto& estimator) {
      const std::string& estimator_wfn = resolved(estimator.wavefunction, "wavefunction");
      const std::string& estimator_ham = resolved(estimator.hamiltonian, "hamiltonian");
      apply_defaults(find_block(params.wavefunctions, estimator_wfn, "wavefunction"),
                     hamiltonian_type(estimator_ham));
    });
  }
}

} // namespace sfqmc::afqmc
