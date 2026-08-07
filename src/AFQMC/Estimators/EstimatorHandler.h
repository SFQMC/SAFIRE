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

#include "AFQMC/config.h"
#include "IO/banner.hpp"
#include "utilities/mpi_context.h"
#include "utilities/freemem.h"

#include "AFQMC/Utilities/Utils.hpp"

#include "AFQMC/Estimators/EstimatorBase.h"
#include "AFQMC/Estimators/EnergyEstimator.h"
#include "AFQMC/Estimators/BasicEstimator.h"
#include "AFQMC/Estimators/MixedEstimator.hpp"
#include "AFQMC/Estimators/BackPropagatedEstimator.hpp"
#include "AFQMC/Estimators/BPWithTimeEvolvedOperators.hpp"
#include "AFQMC/Walkers/WalkerSet.hpp"
#include "AFQMC/Hamiltonians/HamiltonianFactory.h"
#include "AFQMC/Wavefunctions/WavefunctionFactory.h"
#include "AFQMC/Propagators/Propagator.hpp"

namespace sfqmc
{
namespace afqmc
{

/* 
 * Manager class for all estimators/observables.
 * This class contains and manages a list of estimator objects.
 * An arbitrary combination of estimators can be used simultaneously
 * during a simulation, including: 
 *   1) mixed distribution estimators,  
 *   2) back propagated estimators, 
 *   3) any number of 1),2), 
 *   4) each with independent wavefunctions.
 */
template<MEMORY_SPACE MEM>
class EstimatorHandler
{
  using EstimPtr     = std::shared_ptr<EstimatorBase<MEM>>;
  using communicator = boost::mpi3::communicator;

public:
  EstimatorHandler(std::shared_ptr<utils::mpi_context_t<boost::mpi3::communicator>> _mpi,
                   std::string title,
                   const ExecuteParameters& exec,
                   WalkerSet<MEM>& wset,
                   WavefunctionFactory<MEM>& WfnFac,
                   Wavefunction<MEM>& wfn0,
                   Propagator<MEM>& prop0,
                   HamiltonianFactory& HamFac,
                   double dt,
                   bool defaultEnergyEstim = false,
                   bool impsamp            = true)
      : mpi(_mpi), project_title(title), NMO(wfn0.getNMO()), dt(dt), hdf_output(false)
  {
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

  EstimatorHandler(EstimatorHandler const& other) = delete;
  EstimatorHandler& operator=(EstimatorHandler const& other) = delete;
  EstimatorHandler(EstimatorHandler&& other) = default;
  EstimatorHandler& operator=(EstimatorHandler&& other) = default;

  double getEloc() { return estimators[0]->getEloc(); }

  double getEloc_step() { return estimators[0]->getEloc_step(); }

  void print(int block, double time, double Es, WalkerSet<MEM>& wlks)
  {
    hdf_file = project_title + ".stat.h5";
    h5::file file;
    bool printed_row_prefix = false;
    if (hdf_output and mpi->comm.root())
      file = h5::file(hdf_file,'a');
   
    // print must follow the measure_schedule as well (otherwise the data may not be updated)
    long step = std::lround(time / dt);
    int estimator_index = 0;
    for (auto it = estimators.begin(); it != estimators.end(); it++, estimator_index++)
    {
      if (step % measure_schedule[estimator_index] == 0)
      { 
        if (estimator_index == 0 && not printed_row_prefix)
        {
          // print prefix
          out << block << " " << time << " ";
          printed_row_prefix = true;
        }
        (*it)->print(out, file, wlks);
      }
    }

    if (printed_row_prefix)
    { 
      // print suffix
      out << std::setprecision(12) << Es << "  " << utils::freemem() << " ";
      estimators[0]->print_timers(out);
      out << std::endl;
    }
    if ((block + 1) % 10 == 0)
      out.flush();
  }

  // 1) acumulates estimators over steps, and 2) reduces and accumulates substep estimators
  void accumulate_step(double time, WalkerSet<MEM>& wlks, std::vector<ComplexType>& curData)
  {
    for (auto it = estimators.begin(); it != estimators.end(); it++)
      (*it)->accumulate_step(time, wlks, curData);
  }

  // 1) acumulates estimators over steps, and 2) reduces and accumulates substep estimators
  /* Requests that each estimator measure. Will check the current time against the measurement schedule.*/
  void accumulate_block(double time, WalkerSet<MEM>& wlks)
  {
    long step = std::lround(time / dt); // tmp for debug
    int estimator_index = 0;
    for (auto it = estimators.begin(); it != estimators.end(); it++, estimator_index++)
    {
      if (step % measure_schedule[estimator_index] == 0)
        (*it)->accumulate_block(time, wlks);
    }
  }

  void write_hdf_metadata(h5::file &h5f, WALKER_TYPES wlk, bool free_projection)
  {
    h5::group grp(h5f);
    h5::group mgrp = grp.create_group("Metadata");
    h5::h5_write(mgrp,"NMO", NMO);
    int wlk_t_copy = wlk; // the actual data type of enum is implementation-defined. convert to int for file
    h5::h5_write(mgrp, "WalkerType", wlk_t_copy);
    h5::h5_write(mgrp, "FreeProjection", free_projection);
    h5::h5_write(mgrp, "Timestep", dt);
  }

  
  int get_max_common_interval()
  {
    return get_max_common_interval( std::vector<int> {});
  }

  /*
  We can compute the maximum interval which is commensurate with all estimators.
  It is simply the greatest common devisor among all measurement intervals.
  Any additional intervals to consider (for example the accumulate interval)
  can be passed in the intervals vector.

  The exception is if no gcd exists; in this case, we assume that the maximum interval is 1.

  Notes:
  - All EnergyEstimators and BasicEstimators need to have the same measurement interval since
    they print inline scalar data to the same output file.
  */
  int get_max_common_interval(std::vector<int>&& intervals)
  { 
    int max = 0;
    int max_common_interval = 1;
    bool all_equal = true;

    for (auto& it : measure_schedule)
      intervals.push_back(it.second);

    for (auto& it : intervals)
      if (it > max)
        max = it;

    max_common_interval = max;
    for (auto& it : intervals)
      {
        max_common_interval = std::gcd(max_common_interval, it);
        if (it != max)
          all_equal = false;
      }
    
    if (max_common_interval == max && !all_equal)
      max_common_interval = 1;
    
    return max_common_interval;
  }


  void display_measurement_intervals()
  {
    //TODO: get a descriptive name for each estimator
    app_log(1, section("Measurement Schedule"));
    for (auto& it : measure_schedule)
      app_log(1, "Estimator {} has measurement interval {}", it.first, it.second);
    app_log(1, hrule());
  }

private:
  std::shared_ptr<utils::mpi_context_t<boost::mpi3::communicator>> mpi;

  std::string project_title;

  int NMO{};

  std::vector<EstimPtr> estimators;
  std::vector<std::string> tags;

  std::map<int,int> measure_schedule;
  double dt; // timestep

  std::ofstream out;
  std::string hdf_file;
  bool hdf_output;

  /*
  Check that all Estimators that need to be synchronized are synchronized.
  This is important because some Estimators print to the same output file.

  The following set of Estimators need to be synchronized:
  - EnergyEstimator
  - BasicEstimator

  The following set of Estimators do not need to be synchronized:
  - BackPropagatedEstimator
  - BPWithTimeEvolvedOperators
  - MixedEstimator
  */
  void check_synchronized()
  {
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

};

} // namespace afqmc
} // namespace sfqmc

