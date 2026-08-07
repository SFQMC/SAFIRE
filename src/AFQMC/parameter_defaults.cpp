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
#include <format>
#include <map>
#include <set>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

#include "AFQMC/parameter_defaults.hpp"
#include "AFQMC/Hamiltonians/hdf5_helpers.hpp"
#include "nda/h5.hpp"
#include "utilities/check.hpp"

namespace sfqmc::afqmc {

namespace {

/// The name a resolved execute block refers a component by. Every reference holds a name once
/// resolve_block_refs has run.
template<typename Params>
const std::string& block_name(const std::optional<utils::BlockRef<Params>>& ref, std::string_view key) {
  utils::check(ref.has_value(), "The execute block has no {}.", key);
  const auto* name = std::get_if<std::string>(&*ref);
  utils::check(name != nullptr, "The {} of the execute block was not resolved to a name. Did resolve_defaults run?",
               key);
  return *name;
}

template<typename Params>
Params& find_block(std::vector<Params>& blocks, const std::string& name, std::string_view key) {
  const auto block = std::ranges::find_if(blocks, [&](const Params& candidate) { return candidate.name == name; });
  utils::check(block != blocks.end(), "There is no {} named \"{}\".", key, name);
  return *block;
}

/// Names the blocks of one component and hoists the ones declared inside an execute block into
/// the top level list, so that afterwards every execute block refers to its components by name
/// and `blocks` is the complete registry.
///
/// A generated name is only used if the input does not contain it, so that an input is free to
/// name a block e.g. "propagator_0" itself.
template<typename Params>
void resolve_block_refs(std::string_view key, std::vector<Params>& blocks, std::vector<ExecuteParameters>& execute,
                        std::optional<utils::BlockRef<Params>> ExecuteParameters::*member, bool required) {
  // collect every name the input gives explicitly. This has to see all of them before the first
  // name is generated, because an explicit name may appear in a later execute block than the
  // nameless block that would otherwise be given it.
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
  for(const auto& exec : execute) {
    if(const auto& ref = exec.*member; ref) {
      if(const auto* block = std::get_if<Params>(&*ref); block && !block->name.empty()) {
        claim(block->name);
      }
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

  for(auto& exec : execute) {
    auto& ref = exec.*member;
    if(ref) {
      if(const auto* name = std::get_if<std::string>(&*ref)) {
        utils::check(names.contains(*name), "An execute block refers to the {} \"{}\", which is not declared "
                     "anywhere.", key, *name);
        continue;
      }
    } else {
      utils::check(!required, "An execute block is missing its {}, which is required.", key);
    }

    // an absent block is a default constructed one that nothing else can refer to
    Params block = ref ? std::get<Params>(std::move(*ref)) : Params{};
    if(block.name.empty()) {
      block.name = generate_name();
    }
    ref = utils::BlockRef<Params>{block.name};
    blocks.push_back(std::move(block));
  }
}

} // namespace

HamiltonianTypes peek_hamiltonian_type(const HamiltonianParameters& params,
                                      utils::mpi_context_t<mpi3::communicator>& mpi) {
  utils::check(!params.filename.empty(), "The hamiltonian \"{}\" must contain a filename.", params.name);

  int htype = UNKNOWN;
  if(mpi.comm.root()) {
    h5::file file(params.filename, 'r');
    h5::group grp(file);
    htype = peekHamType(grp, get_hamiltonian_format(grp));
  }
  mpi.comm.broadcast_n(&htype, 1, 0);
  return HamiltonianTypes(htype);
}

void apply_defaults(WavefunctionParameters& params, HamiltonianTypes htype) {
  // sparse trial wavefunctions are only worth it on k-point runs
  if(!params.dense_trial) {
    params.dense_trial = htype != KPFactorized && htype != KPTHC;
  }
  if(!params.algorithm) {
    params.algorithm = (htype == RealDenseFactorized ? PHMSDEnergyAlgorithm::woodbury
                                                     : PHMSDEnergyAlgorithm::reference);
  }
}

void apply_defaults(PropagatorParameters& params, HamiltonianTypes htype) {
  // some defaults take legacy values for model hamiltonians
  const bool model = htype == ModelHamiltonian;
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
  if(!params.symmetric_split) {
    params.symmetric_split = !model;
  }
}

void apply_defaults(EstimatorParameters& params, const ExecuteParameters& exec) {
  if(!params.measure_interval_multiplier) {
    params.measure_interval_multiplier = std::vector<int>{exec.measure_interval_multiplier};
  }
  // an estimator may use a different wavefunction and hamiltonian than the driver
  if(params.wfn.empty()) {
    params.wfn = block_name(exec.wavefunction, "wavefunction");
  }
  if(params.ham.empty()) {
    params.ham = block_name(exec.hamiltonian, "hamiltonian");
  }
}

void apply_defaults(ExecuteParameters& exec) {
  for(const auto& estimator : exec.estimator) {
    utils::check(estimator.name != EstimatorType::undefined, "An estimator block requires a name.");
  }

  const auto is_basic = [](const EstimatorParameters& e) { return e.name == EstimatorType::basic; };
  const auto nbasic = std::ranges::count_if(exec.estimator, is_basic);
  utils::check(nbasic <= 1, "An execute block cannot contain more than one basic estimator block.");
  if(nbasic == 0) {
    // the basic estimator always runs, so its block is always present
    exec.estimator.insert(exec.estimator.begin(), EstimatorParameters{.name = EstimatorType::basic});
  }

  for(auto& estimator : exec.estimator) {
    apply_defaults(estimator, exec);
  }
}

void resolve_defaults(AFQMCParameters& params, utils::mpi_context_t<mpi3::communicator>& mpi) {
  utils::check(!params.execute.empty(), "The input contains no execute block, so there is nothing to run.");

  // 1. + 2. name every block and hoist the ones declared inside an execute block
  resolve_block_refs("wavefunction", params.wavefunction, params.execute, &ExecuteParameters::wavefunction, true);
  resolve_block_refs("hamiltonian", params.hamiltonian, params.execute, &ExecuteParameters::hamiltonian, false);
  resolve_block_refs("walker_set", params.walker_set, params.execute, &ExecuteParameters::walker_set, false);
  resolve_block_refs("propagator", params.propagator, params.execute, &ExecuteParameters::propagator, false);

  for(const auto& wfn : params.wavefunction) {
    utils::check(!wfn.filename.empty(), "The wavefunction \"{}\" must contain a filename.", wfn.name);
  }

  // 3. resolve what a block inherits from a neighbouring block
  for(auto& exec : params.execute) {
    const std::string& wfn_name = block_name(exec.wavefunction, "wavefunction");
    const std::string& ham_name = block_name(exec.hamiltonian, "hamiltonian");

    // a hamiltonian that does not name a file of its own uses the one of the wavefunction
    HamiltonianParameters& ham = find_block(params.hamiltonian, ham_name, "hamiltonian");
    if(ham.filename.empty()) {
      ham.filename = find_block(params.wavefunction, wfn_name, "wavefunction").filename;
    }

    apply_defaults(exec);
  }

  // 4. resolve the defaults that depend on the hamiltonian type. Only the hamiltonians that are
  //    actually used are peeked, and each of them only once.
  std::map<std::string, HamiltonianTypes> htypes;
  auto hamiltonian_type = [&](const std::string& name) {
    const auto [entry, inserted] = htypes.try_emplace(name, UNKNOWN);
    if(inserted) {
      entry->second = peek_hamiltonian_type(find_block(params.hamiltonian, name, "hamiltonian"), mpi);
    }
    return entry->second;
  };

  for(const auto& exec : params.execute) {
    const HamiltonianTypes htype = hamiltonian_type(block_name(exec.hamiltonian, "hamiltonian"));
    apply_defaults(find_block(params.wavefunction, block_name(exec.wavefunction, "wavefunction"), "wavefunction"),
                   htype);
    apply_defaults(find_block(params.propagator, block_name(exec.propagator, "propagator"), "propagator"), htype);

    // an estimator that brings its own wavefunction builds it from its own hamiltonian
    for(const auto& estimator : exec.estimator) {
      apply_defaults(find_block(params.wavefunction, estimator.wfn, "wavefunction"),
                     hamiltonian_type(estimator.ham));
    }
  }
}

} // namespace sfqmc::afqmc
