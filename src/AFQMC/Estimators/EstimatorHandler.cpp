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

#include "AFQMC/Estimators/EstimatorHandler.h"

#include "AFQMC/Estimators/EnergyEstimator.h"
#include "AFQMC/Estimators/BasicEstimator.h"
#include "AFQMC/Estimators/MixedEstimator.hpp"
#include "AFQMC/Estimators/BackPropagatedEstimator.hpp"
#include "AFQMC/Estimators/BPWithTimeEvolvedOperators.hpp"
#include "AFQMC/Hamiltonians/HamiltonianFactory.h"
#include "AFQMC/Wavefunctions/WavefunctionFactory.h"
#include "AFQMC/Propagators/Propagator.hpp"

namespace sfqmc {
namespace afqmc {

template<MEMORY_SPACE MEM>
EstimatorHandler<MEM>::EstimatorHandler(
                   std::shared_ptr<utils::mpi_context_t<boost::mpi3::communicator>> _mpi,
                   std::string title,
                   const ExecuteParameters& exec,
                   WalkerSet<MEM>& wset,
                   WavefunctionFactory<MEM>& WfnFac,
                   Wavefunction<MEM>& wfn0,
                   Propagator<MEM>& prop0,
                   HamiltonianFactory& HamFac,
                   double dt,
                   bool defaultEnergyEstim,
                   bool impsamp)
    : mpi(_mpi), project_title(title), NMO(wfn0.getNMO()), dt(dt), hdf_output(false) {
  estimators.reserve(10);

  app_log(1, section("Initializing Estimators"));

  // every measurement interval is a multiple of the population control interval
  const int pop_control_interval = exec.population_control_interval;
  const int measure_interval = exec.measure_interval_multiplier * pop_control_interval;

  int est_index = 0;
  const EstimatorParameters* basic_params = nullptr;
  bool overwrite_default_energy=false;
  bool remove_default_energy=false;
  for(const auto& params : exec.estimator)
  {
    if (params.name == EstimatorType::basic)
    {
      basic_params = std::addressof(params);
    }
    else if (params.name == EstimatorType::energy)
    {
      overwrite_default_energy = params.overwrite;
      remove_default_energy = params.remove;
    }
  }
  utils::check(basic_params != nullptr, " Error: missing basic estimator block. Did resolve_defaults run? ");

  // the basic estimator always measures at the interval of the execute block
  estimators.emplace_back(
      static_cast<EstimPtr>(std::make_shared<BasicEstimator<MEM>>(mpi, title, *basic_params, measure_interval, impsamp)));
  measure_schedule[est_index] = estimators.back()->get_measurement_interval();
  est_index++;

  // add an EnergyEstimator if requested. It is not part of the input, so it takes the defaults.
  if (defaultEnergyEstim &&
          not(overwrite_default_energy or remove_default_energy)
      )
    {
      const EstimatorParameters energy_params{.name = EstimatorType::energy};
      estimators.emplace_back(
        std::make_shared<EnergyEstimator<MEM>>(mpi, energy_params, measure_interval, wfn0, impsamp));
      measure_schedule[est_index] = estimators.back()->get_measurement_interval();
      est_index++;
    }

  int bp_estimator(false);
  for(const auto& params : exec.estimator)
  {
    if (params.name == EstimatorType::basic)
    {
      // do nothing
      // first process estimators that do not need a wfn
      continue;
    }

    // now do those that do. An estimator may use a different ham & wfn from the driver;
    // resolve_defaults has set both names to the driver's in the common case.
    const int targetNW = exec.n_walkers_per_mpi_task;
    Wavefunction<MEM>* wfn = nullptr;
    if (WfnFac.is_constructed(params.wfn))
    {
      wfn = std::addressof(WfnFac.getWavefunction(mpi, params.wfn, wfn0.getWalkerType(),
                                                  wfn0.isFiniteTemperature(), nullptr, targetNW));
    }
    else
    {
      Hamiltonian& ham = HamFac.getHamiltonian(mpi, params.ham);
      wfn              = std::addressof(WfnFac.getWavefunction(mpi, params.wfn, wfn0.getWalkerType(),
                                                  wfn0.isFiniteTemperature(), std::addressof(ham), targetNW));
    }

    switch (params.name)
    {
      case EstimatorType::back_propagation:
      {
        utils::check(not bp_estimator, " Error: Only one back propagator estimator allowed. ");
        estimators.emplace_back(static_cast<EstimPtr>(
            std::make_shared<BackPropagatedEstimator<MEM>>(mpi, title, params,
                                                      pop_control_interval, wset, *wfn, prop0, impsamp)));
        hdf_output = true;
        bp_estimator = true;
        break;
      }
      case EstimatorType::time_evolved_operators:
      {
        utils::check(not bp_estimator, " Error: Only one back propagator estimator allowed. ");
        estimators.emplace_back(static_cast<EstimPtr>(
            std::make_shared<BPWithTimeEvolvedOperators<MEM>>(mpi, title, params,
                                                      pop_control_interval, wset, *wfn, prop0, impsamp)));
        hdf_output = true;
        bp_estimator = true;
        break;
      }
      case EstimatorType::mixed:
      {
        estimators.emplace_back(static_cast<EstimPtr>(
            std::make_shared<MixedEstimator<MEM>>(mpi, title, params,
                                             pop_control_interval, wset.getWalkerType(), *wfn)));
        hdf_output = true;
        break;
      }
      case EstimatorType::energy:
      {
        // NOTE: use the measurement interval of the execute block rather than the one of
        //       this estimator, to ensure synchronization with other estimators that print
        //       to the scalar data file
        if(not params.remove) {
          estimators.emplace_back(
              std::make_shared<EnergyEstimator<MEM>>(mpi, params, measure_interval, *wfn, impsamp));
        } else {
          continue;
        }
        break;
      }
      default:
        utils::check(false, " Error: unhandled estimator type. ");
    }
    measure_schedule[est_index] = estimators.back()->get_measurement_interval();
    est_index++;
  }

  check_synchronized(); // for Estimators that print to the same line of the scalar.dat file

  if (mpi->comm.rank() == 0)
  {
    std::string filename = project_title + ".scalar.dat";
    if (hdf_output)
    {
      hdf_file = project_title + ".stat.h5";
      h5::file file(hdf_file, 'w');
      write_hdf_metadata(file, wset.getWalkerType(), !impsamp);
    }
    out.open(filename.c_str());
    utils::check(not out.fail(), "Problems opening estimator output file: " + filename + ""); 
    out << "# block  time  ";
    for (auto it = estimators.begin(); it != estimators.end(); it++)
      (*it)->tags(out);
    out << "Eshift freeMemory ";
    estimators[0]->tags_timers(out);
    out << std::endl;
  }
}


template<MEMORY_SPACE MEM>
void EstimatorHandler<MEM>::check_synchronized() {
  int synchronized_interval = -1;
  for (auto& it : measure_schedule)
  { 
    if (it.first < estimators.size())
    {
      auto estimator = estimators[it.first];
      if (std::dynamic_pointer_cast<BasicEstimator<MEM>>(estimator) || 
          std::dynamic_pointer_cast<EnergyEstimator<MEM>>(estimator)) 
      {
        if (synchronized_interval == -1)
          synchronized_interval = it.second;
        else if (it.second != synchronized_interval)
          APP_ABORT(" Error: EnergyEstimator and BasicEstimator must have the same measurement interval.");
      }
    }
  }
}

template class EstimatorHandler<HOST_MEMORY>;
#if defined(ENABLE_DEVICE)
template class EstimatorHandler<DEVICE_MEMORY>;
#endif

} // namespace afqmc
} // namespace sfqmc
