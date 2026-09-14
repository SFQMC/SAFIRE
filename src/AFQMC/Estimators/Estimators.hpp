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

#include <filesystem>
#include <format>
#include <memory>
#include <vector>

#include "AFQMC/parameters.hpp"
#include "IO/banner.hpp"
#include "utilities/check.hpp"

#include "AFQMC/Hamiltonians/HamiltonianFactory.h"
#include "AFQMC/Propagators/Propagator.hpp"
#include "AFQMC/Wavefunctions/WavefunctionFactory.h"

#include "EstimatorBase.h"
#include "EnergyEstimator.h"
#include "MixedEstimator.hpp"
#include "BackPropEstimator.hpp"
#include "TimeEvolvedBPEstimator.hpp"

namespace sfqmc::afqmc {

template<MEMORY_SPACE MEM>
class Estimators {
public:
  Estimators(std::shared_ptr<utils::mpi_context_t<boost::mpi3::communicator>> mpi,
             int stage,
             ExecuteParameters const& exec,
             WalkerSet<MEM>& wset,
             WavefunctionFactory<MEM>& wfnFac,
             Wavefunction<MEM>& wfn0,
             Propagator<MEM>& prop,
             HamiltonianFactory& hamFac)
      : measurements_{std::format("Stage{}", stage)} {
    app_log(1, section("Initializing Estimators"));

    // the driver counts measurement blocks in population control intervals, so every
    // estimator's measure_interval_multiplier is a multiple of that block count rather than
    // a number of steps. Only the back propagation estimators need the interval itself, to
    // turn a block count into a number of propagation steps.
    int const pop_control_interval = exec.population_control_interval;

    // an estimator may use a different hamiltonian and wavefunction than the driver.
    // resolve_defaults has set both names to the driver's in the common case.
    auto estimator_wavefunction = [&](auto const& params) -> Wavefunction<MEM>& {
      std::string const& wfn_name = resolved(params.wavefunction, "wavefunction");
      Hamiltonian* ham = nullptr;
      if(!wfnFac.is_constructed(wfn_name)) {
        ham = std::addressof(hamFac.getHamiltonian(mpi, resolved(params.hamiltonian, "hamiltonian")));
      }
      return wfnFac.getWavefunction(mpi, wfn_name, wfn0.getWalkerType(), wfn0.isFiniteTemperature(), ham,
                                    exec.n_walkers_per_mpi_task);
    };

    // apply_defaults has already rejected the combination of both back propagation estimators
    EstimatorParameters const& estimators = exec.estimators;

    if(estimators.energy) {
      Wavefunction<MEM>& energy_wfn = estimator_wavefunction(*estimators.energy);
      // the walkers only ever carry the local energy of the driver's own wavefunction
      bool const walkers_carry_energy =
          prop.stores_local_energy() && std::addressof(energy_wfn) == std::addressof(wfn0);
      estimators_.emplace_back(std::make_unique<EnergyEstimator<MEM>>(
          *estimators.energy, walkers_carry_energy, energy_wfn));
      app_log(1, "Energy estimator initialized");
    }

    if(estimators.mixed) {
      estimators_.emplace_back(std::make_unique<MixedEstimator<MEM>>(
          *mpi, *estimators.mixed, wset.getWalkerType(),
          estimator_wavefunction(*estimators.mixed)));
      app_log(1, "Mixed estimator initialized");
    }

    if(estimators.backprop) {
      estimators_.emplace_back(std::make_unique<BackPropEstimator<MEM>>(
          *mpi, *estimators.backprop, pop_control_interval, wset,
          estimator_wavefunction(*estimators.backprop), prop));
      app_log(1, "Back-propagation estimator initialized");
    }

    if(estimators.time_evolved_bp) {
      estimators_.emplace_back(std::make_unique<TimeEvolvedBPEstimator<MEM>>(
          *mpi, *estimators.time_evolved_bp, pop_control_interval, wset,
          estimator_wavefunction(*estimators.time_evolved_bp), prop));
      app_log(1, "Time-evolved back-propagation estimator initialized");
    }
  }

  // call this once per measure block (= population control interval)
  void measure(utils::mpi_context_t<boost::mpi3::communicator>& mpi, long measureBlock, WalkerSet<MEM> &wset) {
    for(auto const& estimator : estimators_) {
      estimator->measure(mpi, measureBlock, measurements_, wset);
    }
  }

  void write(std::filesystem::path const& path) {
    auto tmppath = path;
    tmppath += ".tmp";
    
    if(std::filesystem::exists(path)) {
      std::filesystem::copy(path, tmppath, std::filesystem::copy_options::overwrite_existing);
    }

    {
      h5::file out(tmppath, 'a');
      h5::group root(out);

      h5::group meas_group = utils::h5_open_or_create(root, "Measurements");
      measurements_.write(meas_group);
    }

    std::filesystem::rename(tmppath, path);
  }
private:
  Measurements measurements_;
  std::vector<std::unique_ptr<EstimatorBase<MEM>>> estimators_;
};

} // namespace sfqmc::afqmc
