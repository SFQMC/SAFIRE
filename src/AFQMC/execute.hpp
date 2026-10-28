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

#pragma once

#include <memory>

#include "AFQMC/config.h"
#include "AFQMC/parameters.hpp"
#include "utilities/mpi_context.h"

namespace sfqmc::afqmc {

/// Runs every execute block of the input in order, each appending its own Stage<N> group to the
/// results file.
///
/// The walker sets, wavefunctions and hamiltonians a stage names are built once per name and
/// cached for the whole run, so a stage that names the walker set of an earlier one picks up
/// where that one left off. Propagators carry no state worth keeping and are rebuilt per stage.
///
/// Defined in execute.cpp and explicitly instantiated there, so that a caller does not have to
/// compile the AFQMC template stack to start a simulation.
template<MEMORY_SPACE MEM>
void execute_simulation(std::shared_ptr<utils::mpi_context_t<boost::mpi3::communicator>> mpi,
                        AFQMCParameters const& params);

} // namespace sfqmc::afqmc
