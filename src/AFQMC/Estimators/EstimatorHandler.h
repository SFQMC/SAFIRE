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
                   ptree exec_pt,
                   WalkerSet<MEM>& wset,
                   WavefunctionFactory<MEM>& WfnFac,
                   Wavefunction<MEM>& wfn0,
                   Propagator<MEM>& prop0,
                   HamiltonianFactory& HamFac,
                   std::string ham0,
                   double dt,
                   bool defaultEnergyEstim = false,
                   bool impsamp            = true)
      : mpi(_mpi), project_title(title), NMO(wfn0.getNMO()), dt(dt), hdf_output(false)
  {
    estimators.reserve(10);

    app_log(1,"\n****************************************************");
    app_log(1,"               Initializing Estimators ");
    app_log(1,"\n****************************************************");
    app_log(2,"\n EstimatorHandler input:\n{}\n", io::to_string(exec_pt));
    ptree est_pt, basic_pt;
    int est_index = 0;
    // default measure_interval to DEFAULT_MEASURE_INTERVAL if not specified in exec_pt
    int measure_interval_multiplier = exec_pt.get<int>("measure_interval_multiplier", DEFAULT_MEASURE_INTERVAL_MULTIPLIER);
    int population_control_interval = exec_pt.get<int>("population_control_interval", DEFAULT_POPULATION_CONTROL_INTERVAL);

    bool overwrite_default_energy=false;
    bool remove_default_energy=false;
    for(auto& it : exec_pt)
    {
      std::string cname = it.first;
      if (cname == "estimator")
      {
        ptree child = it.second;
        std::string name = child.get<std::string>("name");
        if (name == "basic" || name == "Basic" || name == "standard")
        {
          basic_pt = child;
        }
        else if (name == "energy")
        {
          overwrite_default_energy = child.get<bool>("overwrite", false);
          remove_default_energy = child.get<bool>("remove", false);
        }
      }
    }

    basic_pt.put("measure_interval_multiplier", measure_interval_multiplier);
    if (remove_node_if_exists(basic_pt, "_population_control_interval"))
    {
      app_warning("'_population_control_interval' is set in a 'basic' estimator block. The value in the 'basic' estimator block will be ignored.");
    }
    basic_pt.put("_population_control_interval", population_control_interval); // to compute measure_interval
    est_pt.put("measure_interval_multiplier", measure_interval_multiplier);
    est_pt.put("_population_control_interval", population_control_interval); // to compute measure_interval

    estimators.emplace_back(
        static_cast<EstimPtr>(std::make_shared<BasicEstimator<MEM>>(mpi, title, basic_pt, impsamp)));
    measure_schedule[est_index] = estimators.back()->get_measurement_interval();
    est_index++;

    // add an EnergyEstimator if requested
    if (defaultEnergyEstim && 
            not(overwrite_default_energy or remove_default_energy) 
        )
      {
        estimators.emplace_back(
          std::make_shared<EnergyEstimator<MEM>>(mpi, est_pt, wfn0, impsamp));
        measure_schedule[est_index] = estimators.back()->get_measurement_interval();
        est_index++;
      }

    int bp_estimator(false);
    for(auto& it : exec_pt)
    {
      std::string cname = it.first;
      if (cname == "estimator")
      {
        est_pt = it.second;
        std::string name, wfn_name, ham_name;
        name = est_pt.get<std::string>("name");
        // Estimator can use different ham & wfn from Driver
        wfn_name = est_pt.get<std::string>("wfn", "");
        ham_name = est_pt.get<std::string>("ham", "");
        //  default is same ham & wfn from Driver
        
        int child_measure_interval_multiplier = est_pt.get<int>("measure_interval_multiplier", measure_interval_multiplier);
        if (remove_node_if_exists(est_pt, "_population_control_interval"))
        {
          app_warning("'_population_control_interval' is set in an estimator block. The value in the estimator block will be ignored!");
        }
        est_pt.put("_population_control_interval", population_control_interval); // to compute measure_interval


        if (name == "basic" || name == "Basic" || name == "standard")
        {
          // do nothing
          // first process estimators that do not need a wfn
        }
        else
        {
          // now do those that do
          Wavefunction<MEM>* wfn = &wfn0;
          if (wfn_name != "")
          { // wfn_name must produce a viable wfn object
            ptree wfn_pt = WfnFac.get_input(wfn_name);
            if (WfnFac.is_constructed(wfn_name))
            {
              wfn = std::addressof(
                  WfnFac.getWavefunction(mpi, wfn_name, wfn0.getWalkerType(), wfn0.isFiniteTemperature(), nullptr));
            }
            else if (ham_name != "")
            {
              Hamiltonian& ham = HamFac.getHamiltonian(mpi, ham_name);
              wfn              = std::addressof(WfnFac.getWavefunction(mpi, wfn_name,
                                                          wfn0.getWalkerType(), wfn0.isFiniteTemperature(), std::addressof(ham)));
            }
            else
            {
              Hamiltonian& ham = HamFac.getHamiltonian(mpi, ham0);
              wfn              = std::addressof(WfnFac.getWavefunction(mpi, wfn_name,
                                                          wfn0.getWalkerType(), wfn0.isFiniteTemperature(), std::addressof(ham)));
            }
            utils::check(wfn != nullptr, " Error: Problems generating wavefunction in DriverFactory::executeAFQMCDriver(). ");
          }

          if (name == "back_propagation")
          {
            utils::check(not bp_estimator, " Error: Only one back propagator estimator allowed. ");
            est_pt.put("measure_interval_multiplier", child_measure_interval_multiplier);
            estimators.emplace_back(static_cast<EstimPtr>(
                std::make_shared<BackPropagatedEstimator<MEM>>(mpi, title, est_pt, wset, *wfn,
                                                          prop0, impsamp)));
            measure_schedule[est_index] = estimators.back()->get_measurement_interval();
            est_index++;
            hdf_output = true;
            bp_estimator = true;
          }
          else if (name == "time_evolved_operators")
          {
            utils::check(not bp_estimator, " Error: Only one back propagator estimator allowed. ");
            est_pt.put("measure_interval_multiplier", child_measure_interval_multiplier);
            estimators.emplace_back(static_cast<EstimPtr>(
                std::make_shared<BPWithTimeEvolvedOperators<MEM>>(mpi, title,
                            est_pt, wset, *wfn, prop0, impsamp)));
            measure_schedule[est_index] = estimators.back()->get_measurement_interval();
            est_index++;
            hdf_output = true;
            bp_estimator = true;
          }
          else if (name == "mixed")
          {
            est_pt.put("measure_interval_multiplier", child_measure_interval_multiplier);
            estimators.emplace_back(static_cast<EstimPtr>(
                std::make_shared<MixedEstimator<MEM>>(mpi, title, est_pt, wset.getWalkerType(),
                                                 *wfn)));
            measure_schedule[est_index] = estimators.back()->get_measurement_interval();
            est_index++;
            hdf_output = true;
          }
          else if (name == "energy")
          {
            // NOTE: do not put child_measure_interval into est_pt "measure_interval_multiplier"
            //       to ensure synchronization with other estimators that print to the
            //       scalar data file
            est_pt.put("measure_interval_multiplier", measure_interval_multiplier);
            bool remove = est_pt.get<bool>("remove", false);
            if(not remove) {
              estimators.emplace_back(
                  std::make_shared<EnergyEstimator<MEM>>(mpi, est_pt, *wfn, impsamp));
              measure_schedule[est_index] = estimators.back()->get_measurement_interval();
              est_index++;
            }
          }
          else
          {
            app_log(1," Ignoring unknown estimator type: {}", name); 
          }
        }
      }
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

  ~EstimatorHandler() {}

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
    app_log(1, "\n\n======= Measurement Schedule: =========\n");
    for (auto& it : measure_schedule)
      app_log(1, "Estimator {} has measurement interval {}", it.first, it.second);
    app_log(1, "\n======================================\n\n");
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

