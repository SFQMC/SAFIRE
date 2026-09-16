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

#include <filesystem>
#include <format>
#include <map>
#include <memory>
#include <string>

#include "AFQMC/config.h"
#include "AFQMC/Drivers/run_afqmc.hpp"
#include "AFQMC/Drivers/run_ftafqmc.hpp"
#include "AFQMC/Estimators/Estimators.hpp"
#include "AFQMC/execute.hpp"
#include "AFQMC/Hamiltonians/Hamiltonian.hpp"
#include "AFQMC/parameter_defaults.hpp"
#include "AFQMC/Propagators/AFQMCBasePropagator.h"
#include "AFQMC/Propagators/Propagator.hpp"
#include "AFQMC/Utilities/AFQMCTimer.h"
#include "AFQMC/Walkers/WalkerSet.hpp"
#include "AFQMC/Wavefunctions/Wavefunction.hpp"
#include "IO/app_loggers.h"
#include "IO/banner.hpp"
#include "utilities/check.hpp"
#include "utilities/Random.hpp"

namespace sfqmc::afqmc {

namespace {

// assumes wfn.Energy(Wset) has been called
// Then prints the energy breakdown
template<typename WlkSet>
void print_initial_energy(WlkSet& wset) {
  app_log(1, "Local Energy of starting determinant ");
  app_log(1, "  - Total energy    : {:f}", wset[0].energy());
  app_log(1, "  - One-body energy : {:f}", wset[0].get_property(E1_));
  app_log(1, "  - Coulomb energy  : {:f}", wset[0].get_property(EJ_));
  app_log(1, "  - Exchange energy : {:f}", wset[0].get_property(EXX_));
}

/// The wavefunction named `wavefunction_name`, built on first use and cached afterwards, so that
/// a later stage or an estimator naming the same wavefunction shares this one.
///
/// The hamiltonian is only needed to build a wavefunction, so it is built here and released again
/// as soon as the wavefunction has taken what it needs from it.
template<MEMORY_SPACE MEM>
Wavefunction<MEM>& construct_wavefunction(std::shared_ptr<utils::mpi_context_t<boost::mpi3::communicator>> mpi,
                                          std::map<std::string, Wavefunction<MEM>>& wavefunctions,
                                          AFQMCParameters const& params,
                                          std::string const& wavefunction_name,
                                          std::string const& hamiltonian_name,
                                          WALKER_TYPES walker_type,
                                          bool finiteT,
                                          int nwalkers) {
  auto entry = wavefunctions.find(wavefunction_name);
  if(entry == wavefunctions.end()) {
    Hamiltonian hamiltonian =
        Hamiltonian::from_params(mpi, find_block(params.hamiltonian, hamiltonian_name, "hamiltonian"));

    entry = wavefunctions
                .emplace(wavefunction_name,
                         Wavefunction<MEM>::from_params(
                             mpi, find_block(params.wavefunction, wavefunction_name, "wavefunction"),
                             walker_type, finiteT, hamiltonian, nwalkers))
                .first;
  }

  // a cached wavefunction was built for the walker type of the stage that first asked for it, and
  // cannot be reinterpreted as another one
  utils::check(entry->second.getWalkerType() == walker_type,
               "The wavefunction \"{}\" was built for {} walkers, but is now used with {} ones.",
               wavefunction_name, walkerTypeToString(entry->second.getWalkerType()),
               walkerTypeToString(walker_type));

  return entry->second;
}

/// Eshift as the propagator wants it to start out: hybrid propagation takes what the input asked
/// for, or zero, while local energy importance sampling ignores the input and starts from the
/// local energy of the population.
template<MEMORY_SPACE MEM>
RealType initial_Eshift(ExecuteParameters const& exec, Propagator<MEM>& propagator, WalkerSet<MEM> const& wset) {
  if(propagator.hybrid_propagation()) {
    if(exec.initial_Eshift) {
      app_warning("user set expert-level parameter, \"initial_Eshift\" : Using user-provided initial Eshift = {}",
                  *exec.initial_Eshift);
      return *exec.initial_Eshift;
    }
    return 0.0;
  }

  if(exec.initial_Eshift) {
    app_log(1, "[Warning] : User set initial Eshift {} with local energy importance. This value is ignored.",
            *exec.initial_Eshift);
  }
  return real(ComplexType(wset[0].energy()));
}

} // namespace

template<MEMORY_SPACE MEM>
void execute_simulation(std::shared_ptr<utils::mpi_context_t<boost::mpi3::communicator>> mpi,
                        AFQMCParameters const& params) {
  // every execute block appends its own stage to one results file, so a stale file from a
  // previous run has to go before the first stage writes
  if(mpi->comm.root()) {
    std::filesystem::remove(std::format("{}.results.h5", params.output_name));
  }
  mpi->comm.barrier();

  bool const finiteT = params.driver == DriverType::ftafqmc;

  std::map<std::string, WalkerSet<MEM>> walker_sets;
  std::map<std::string, Wavefunction<MEM>> wavefunctions;

  int stage_index{};
  for(auto const& stage : params.execute) {
    timers.reset_all();
    app_log(1, banner("Beginning stage initialization"));

    std::string const& walker_set_name   = block_name(stage.walker_set, "walker_set");
    std::string const& wavefunction_name = block_name(stage.wavefunction, "wavefunction");
    std::string const& hamiltonian_name  = block_name(stage.hamiltonian, "hamiltonian");
    std::string const& propagator_name   = block_name(stage.propagator, "propagator");

    WalkerSetParameters const& walker_set_params = find_block(params.walker_set, walker_set_name, "walker_set");
    int const nwalkers = stage.n_walkers_per_mpi_task;

    // the walkers and the auxiliary fields draw from separate generators. Both are seeded from
    // the stage's seed, which makes them reproducible but not independent; without one they are
    // seeded independently.
    utils::SeedType seed = stage.seed ? utils::split_seed(*stage.seed, mpi->comm) : utils::make_seed(mpi->comm);
    auto walker_rng = std::make_shared<utils::RandomGenerator_t<HOST_MEMORY>>(seed);
    seed = stage.seed ? utils::split_seed(*stage.seed, mpi->comm) : utils::make_seed(mpi->comm);
    auto field_rng = std::make_shared<utils::RandomGenerator_t<MEM>>(seed);

    auto wavefunction_for = [&](std::string const& wfn_name, std::string const& ham_name) -> Wavefunction<MEM>& {
      return construct_wavefunction<MEM>(mpi, wavefunctions, params, wfn_name, ham_name,
                                         walker_set_params.walker_type, finiteT, nwalkers);
    };

    Wavefunction<MEM>& wavefunction = wavefunction_for(wavefunction_name, hamiltonian_name);

    Propagator<MEM> propagator{AFQMCBasePropagator<MEM>(
        find_block(params.propagator, propagator_name, "propagator"), mpi, wavefunction, field_rng)};

    WalkerSet<MEM>& walker_set = walker_sets
                                     .try_emplace(walker_set_name, mpi, walker_rng, walker_set_params,
                                                  wavefunction.initial_guess(), nwalkers)
                                     .first->second;
    if(finiteT) {
      walker_set.setTauStep(0); // time-slice initialized to 0
    }

    // perform runtime optimization; for finite temperature ntau is implicitly 0 here
    wavefunction.runtime_optimization(walker_set);
    wavefunction.Energy(walker_set);

    if(finiteT) {
      memory::buffered_array<MEM, ComplexType, 1> ovlp0(nwalkers, 0.0);
      walker_set.getProperty(OVLP, ovlp0);
      wavefunction.setLogPT0(ovlp0);
    }

    print_initial_energy(walker_set);

    RealType const Eshift = initial_Eshift(stage, propagator, walker_set);

    Estimators<MEM> estimators{mpi, stage_index, stage, walker_set, wavefunction, propagator, wavefunction_for};

    // free any shared windows that were abandoned during initialization
    mpi->shared_windows.collective_free_unused();

    if(finiteT) {
      run_ftafqmc<MEM>(*mpi, params.output_name, stage, Eshift, walker_set, wavefunction, propagator, estimators);
    } else {
      run_afqmc<MEM>(*mpi, params.output_name, stage, Eshift, walker_set, propagator, estimators);
    }

    ++stage_index;
  }
}

// Instantiate
#define __inst__(M)                                                                                \
  template void execute_simulation<M>(std::shared_ptr<utils::mpi_context_t<boost::mpi3::communicator>>, \
                                      AFQMCParameters const&);

__inst__(HOST_MEMORY)
#if defined(ENABLE_DEVICE)
__inst__(DEVICE_MEMORY)
#endif

} // namespace sfqmc::afqmc
