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

#pragma once

#include <algorithm>
#include <optional>
#include <string>
#include <string_view>
#include <vector>

#include "AFQMC/config.h"
#include "AFQMC/parameters.hpp"
#include "utilities/check.hpp"
#include "utilities/mpi_context.h"

namespace sfqmc::afqmc {

/// The name a resolved execute block refers a component by. Every reference holds a name once
/// resolve_defaults has run.
template<typename Params>
const std::string& block_name(const std::optional<utils::BlockRef<Params>>& ref, std::string_view key) {
  utils::check(ref.has_value(), "The execute block has no {}.", key);
  const auto* name = std::get_if<std::string>(&*ref);
  utils::check(name != nullptr, "The {} of the execute block was not resolved to a name. Did resolve_defaults run?",
               key);
  return *name;
}

/// The block of a registry that carries `name`. resolve_defaults has made the top level lists the
/// complete registry, so a name that is not in one is an error.
template<typename Params>
Params& find_block(std::vector<Params>& blocks, const std::string& name, std::string_view key) {
  const auto block = std::ranges::find_if(blocks, [&](const Params& candidate) { return candidate.name == name; });
  utils::check(block != blocks.end(), "There is no {} named \"{}\".", key, name);
  return *block;
}

template<typename Params>
const Params& find_block(const std::vector<Params>& blocks, const std::string& name, std::string_view key) {
  const auto block = std::ranges::find_if(blocks, [&](const Params& candidate) { return candidate.name == name; });
  utils::check(block != blocks.end(), "There is no {} named \"{}\".", key, name);
  return *block;
}

/// Reads the Hamiltonian type of an integral file without building anything. Collective: the
/// root opens the file and broadcasts the result.
HamiltonianType peek_hamiltonian_type(const HamiltonianParameters& params,
                                      utils::mpi_context_t<mpi3::communicator>& mpi);

/// Fills the defaults that depend on the Hamiltonian type. Members that the input set are
/// left alone.
void apply_defaults(WavefunctionParameters& params, HamiltonianType htype);
void apply_defaults(PropagatorParameters& params, HamiltonianType htype);

/// Fills the defaults that the estimators inherit from the execute block containing them, and
/// checks that the set of estimators is consistent. The blocks of `exec` have to be resolved to
/// names already, which is what resolve_defaults does before it calls this.
void apply_defaults(EstimatorParameters& params, const ExecuteParameters& exec);

/// Fills the defaults of every estimator the execute block requests. The component blocks of
/// `exec` have to be resolved to names already.
void apply_defaults(ExecuteParameters& exec);

/// Applies every default that cannot be expressed as a member initializer of the parameter
/// structs, so that the rest of the code only ever sees resolved values:
///
/// 1. Names every block. A block that an execute block leaves out entirely is materialized as
///    a default constructed one. Generated names never collide with the names in the input.
/// 2. Hoists the blocks declared inside an execute block into the top level lists, leaving the
///    execute block referring to them by name. Afterwards every reference in an execute block
///    is a name, and the top level lists are the complete registry of blocks.
/// 3. Resolves the defaults a block inherits from a neighbouring block.
/// 4. Peeks the type of every Hamiltonian and resolves the defaults that depend on it.
///
/// Collective, because of the peek in the last step.
void resolve_defaults(AFQMCParameters& params, utils::mpi_context_t<mpi3::communicator>& mpi);

} // namespace sfqmc::afqmc
