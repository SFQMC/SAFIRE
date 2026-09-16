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

#pragma once

#include <string>

#include "AFQMC/config.h"
#include "AFQMC/Estimators/Estimators.hpp"
#include "AFQMC/parameters.hpp"
#include "AFQMC/Propagators/Propagator.hpp"
#include "AFQMC/Walkers/WalkerSet.hpp"
#include "utilities/mpi_context.h"

namespace sfqmc::afqmc {

/// Projects the walker population for `exec.steps` steps and writes the measurements to
/// `<output_name>.results.h5`.
///
/// A measure block is one population control interval. Nothing is measured during the first
/// `exec.equilibration_steps` steps; `Eshift` is damped towards the population average instead,
/// and stops moving once the measurements begin.
template<MEMORY_SPACE MEM>
void run_afqmc(utils::mpi_context_t<boost::mpi3::communicator>& mpi,
               std::string const& output_name,
               ExecuteParameters const& exec,
               RealType Eshift,
               WalkerSet<MEM>& wset,
               Propagator<MEM>& propagator,
               Estimators<MEM>& estimators);

} // namespace sfqmc::afqmc
