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
#include "AFQMC/Wavefunctions/Wavefunction.hpp"
#include "utilities/mpi_context.h"

namespace sfqmc::afqmc {

/// Runs `exec.sweeps` sweeps of `exec.steps` imaginary time slices, i.e. beta = exec.steps *
/// exec.timestep, and writes the measurements to `<output_name>.results.h5`.
///
/// One sweep is one measurement sample: the observables are only defined once the full path has
/// been constructed, so the measurement happens after the step loop rather than inside it. The
/// walker set and the wavefunction's log scale are reset at the end of every sweep.
template<MEMORY_SPACE MEM>
void run_ftafqmc(utils::mpi_context_t<boost::mpi3::communicator>& mpi,
                 std::string const& output_name,
                 ExecuteParameters const& exec,
                 RealType Eshift,
                 WalkerSet<MEM>& wset,
                 Wavefunction<MEM>& wavefunction,
                 Propagator<MEM>& propagator,
                 Estimators<MEM>& estimators);

} // namespace sfqmc::afqmc
