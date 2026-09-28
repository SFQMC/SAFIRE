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

/// The name a resolved reference refers a block by. Every reference holds a name once
/// resolve_defaults has run.
template<typename Params>
const std::string& block_name(const utils::BlockRef<Params>& ref, std::string_view key) {
  const auto* name = std::get_if<std::string>(&ref);
  utils::check(name != nullptr, "A {} was not resolved to a name. Did resolve_defaults run?", key);
  return *name;
}

template<typename Params>
const std::string& block_name(const std::optional<utils::BlockRef<Params>>& ref, std::string_view key) {
  utils::check(ref.has_value(), "The execute block has no {}.", key);
  return block_name(*ref, key);
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
/// `exec` have to be resolved to names already. `driver` selects the unit of
/// `measure_interval`, which counts steps for the ground state driver and sweeps for the
/// finite temperature one.
void apply_defaults(ExecuteParameters& exec, DriverType driver);

/// Applies every default that cannot be expressed as a member initializer of the parameter
/// structs, so that the rest of the code only ever sees resolved values:
///
/// 1. Draws a seed unless the input gave one, so that the run can be reproduced from the
///    parameters as they are printed.
/// 2. Names every block. A block that an execute block leaves out entirely is materialized as
///    a default constructed one, except for the walker set: only the first execute block gets a
///    default one, and every later execute block carries over the walker set of the one before
///    it. Generated names never collide with the names in the input.
/// 3. Hoists the blocks declared inline, in an execute block or in the source of a walker set,
///    into the top level lists, leaving a reference by name in their place. Afterwards every
///    reference is a name, and the top level lists are the complete registry of blocks.
/// 4. Resolves the defaults a block inherits from a neighbouring block. A walker set without a
///    source starts from the wavefunction of the execute block that introduces it.
/// 5. Peeks the type of every Hamiltonian and resolves the defaults that depend on it.
///
/// Collective, because of the seed broadcast in the first step and the peek in the last one.
void resolve_defaults(AFQMCParameters& params, utils::mpi_context_t<mpi3::communicator>& mpi);

} // namespace sfqmc::afqmc
