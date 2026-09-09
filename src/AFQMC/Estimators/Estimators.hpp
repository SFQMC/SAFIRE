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
  /// defaultEnergyEstimator adds an EnergyEstimator that is not part of the input. It is
  /// needed when the propagator cannot supply the local energy by itself, i.e. for hybrid
  /// propagation.
  Estimators(std::shared_ptr<utils::mpi_context_t<boost::mpi3::communicator>> mpi,
             ExecuteParameters const& exec,
             WalkerSet<MEM>& wset,
             WavefunctionFactory<MEM>& wfnFac,
             Wavefunction<MEM>& wfn0,
             Propagator<MEM>& prop,
             HamiltonianFactory& hamFac,
             bool defaultEnergyEstimator) {
    app_log(1, section("Initializing Estimators"));

    // every measurement interval is a multiple of the population control interval
    int const pop_control_interval = exec.population_control_interval;
    int const measure_interval = exec.measure_interval_multiplier * pop_control_interval;

    // an estimator may use a different hamiltonian and wavefunction than the driver.
    // resolve_defaults has set both names to the driver's in the common case.
    auto estimator_wavefunction = [&](EstimatorParameters const& params) -> Wavefunction<MEM>& {
      Hamiltonian* ham = nullptr;
      if(!wfnFac.is_constructed(params.wfn)) {
        ham = std::addressof(hamFac.getHamiltonian(mpi, params.ham));
      }
      return wfnFac.getWavefunction(mpi, params.wfn, wfn0.getWalkerType(), wfn0.isFiniteTemperature(), ham,
                                    exec.n_walkers_per_mpi_task);
    };

    bool add_default_energy = defaultEnergyEstimator;
    for(auto const& params : exec.estimator) {
      if(params.name == EstimatorType::energy && (params.overwrite || params.remove)) {
        add_default_energy = false;
      }
    }
    if(add_default_energy) {
      EstimatorParameters const params{.name = EstimatorType::energy,
                                       .measure_interval_multiplier =
                                           std::vector{exec.measure_interval_multiplier}};
      estimators_.emplace_back(std::make_unique<EnergyEstimator<MEM>>(mpi, params, measure_interval, wfn0));
    }

    for(auto const& params : exec.estimator) {
      if(params.remove) {
        continue;
      }
      switch(params.name) {
      case EstimatorType::energy:
        estimators_.emplace_back(
            std::make_unique<EnergyEstimator<MEM>>(mpi, params, measure_interval, estimator_wavefunction(params)));
        app_log(1, "Energy estimator initialized");
        break;
      case EstimatorType::mixed:
        estimators_.emplace_back(std::make_unique<MixedEstimator<MEM>>(*mpi, params, wset.getWalkerType(),
                                                                      estimator_wavefunction(params)));
        app_log(1, "Mixed estimator initialized");
        break;
      case EstimatorType::back_propagation:
        estimators_.emplace_back(std::make_unique<BackPropEstimator<MEM>>(
            *mpi, params, pop_control_interval, wset, estimator_wavefunction(params), prop));
        app_log(1, "Back-propagation estimator initialized");
        break;
      case EstimatorType::time_evolved_operators:
        estimators_.emplace_back(std::make_unique<TimeEvolvedBPEstimator<MEM>>(
            *mpi, params, pop_control_interval, wset, estimator_wavefunction(params), prop));
        app_log(1, "Time-evolved back-propagation estimator initialized");
        break;
      default:
        APP_ABORT("undefined estimator");
      }
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
