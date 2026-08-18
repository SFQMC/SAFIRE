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

#include "AFQMC/config.h"
#include "AFQMC/parameters.hpp"
#include "utilities/mpi_context.h"

namespace sfqmc::afqmc {

/// Reads the Hamiltonian type of an integral file without building anything. Collective: the
/// root opens the file and broadcasts the result.
HamiltonianTypes peek_hamiltonian_type(const HamiltonianParameters& params,
                                       utils::mpi_context_t<mpi3::communicator>& mpi);

/// Fills the defaults that depend on the Hamiltonian type. Members that the input set are
/// left alone.
void apply_defaults(WavefunctionParameters& params, HamiltonianTypes htype);
void apply_defaults(PropagatorParameters& params, HamiltonianTypes htype);

/// Fills the defaults that an estimator inherits from the execute block containing it. The
/// blocks of `exec` have to be resolved to names already, which is what resolve_defaults does
/// before it calls this.
void apply_defaults(EstimatorParameters& params, const ExecuteParameters& exec);

/// Adds the estimator blocks that are always present and fills the defaults of all of them.
/// The component blocks of `exec` have to be resolved to names already.
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
