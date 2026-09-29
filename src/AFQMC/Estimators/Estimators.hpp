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
#include <functional>
#include <memory>
#include <string>
#include <vector>

#include "AFQMC/parameters.hpp"
#include "AFQMC/Utilities/format_version.hpp"
#include "IO/banner.hpp"
#include "utilities/check.hpp"

#include "AFQMC/Propagators/Propagator.hpp"
#include "AFQMC/Wavefunctions/Wavefunction.hpp"

#include "EstimatorBase.h"
#include "EnergyEstimator.h"
#include "MixedEstimator.hpp"
#include "BackPropEstimator.hpp"
#include "TimeEvolvedBPEstimator.hpp"

namespace sfqmc::afqmc {

/// Resolves the wavefunction an estimator names, building it if it does not exist yet. An
/// estimator may use a different wavefunction and hamiltonian than the stage around it;
/// resolve_defaults has set both names to the stage's in the common case.
template<MEMORY_SPACE MEM>
using WavefunctionLookup = std::function<Wavefunction<MEM>&(std::string const& wavefunction_name,
                                                            std::string const& hamiltonian_name)>;

template<MEMORY_SPACE MEM>
class Estimators {
public:
  Estimators(std::shared_ptr<utils::mpi_context_t<boost::mpi3::communicator>> mpi,
             int stage,
             ExecuteParameters const& exec,
             WalkerSet<MEM>& wset,
             Wavefunction<MEM>& wfn0,
             Propagator<MEM>& prop,
             WavefunctionLookup<MEM> const& wavefunction_for)
      : measurements_{std::format("Stage{}", stage), exec.binsize} {
    app_log(1, section("Initializing Estimators"));

    auto estimator_wavefunction = [&](auto const& params) -> Wavefunction<MEM>& {
      return wavefunction_for(resolved(params.wavefunction, "wavefunction"),
                              resolved(params.hamiltonian, "hamiltonian"));
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
          *mpi, *estimators.backprop, wset,
          estimator_wavefunction(*estimators.backprop), prop));
      app_log(1, "Back-propagation estimator initialized");
    }

    if(estimators.time_evolved_bp) {
      estimators_.emplace_back(std::make_unique<TimeEvolvedBPEstimator<MEM>>(
          *mpi, *estimators.time_evolved_bp, wset,
          estimator_wavefunction(*estimators.time_evolved_bp), prop));
      app_log(1, "Time-evolved back-propagation estimator initialized");
    }
  }

  // call this once per propagation step; each estimator measures at an interval of its own
  void measure(utils::mpi_context_t<boost::mpi3::communicator>& mpi, long step, WalkerSet<MEM> &wset) {
    for(auto const& estimator : estimators_) {
      estimator->measure(mpi, step, measurements_, wset);
    }
  }

  // call this once, at the last equilibration step
  void equilibrated(long step, WalkerSet<MEM>& wset) {
    for(auto const& estimator : estimators_) {
      estimator->equilibrated(step, wset);
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

      // every stage appends to one file, which execute starts afresh, so the first stamps it
      if(!has_format_version(root)) {
        write_format_version(root);
      }
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
