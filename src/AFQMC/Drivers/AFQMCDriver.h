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
#include "AFQMC/Propagators/Propagator.hpp"
#include "AFQMC/Wavefunctions/Wavefunction.hpp"
#include "AFQMC/Walkers/WalkerSet.hpp"
#include "AFQMC/Estimators/EstimatorHandler.h"

namespace sfqmc
{
namespace afqmc
{
template<MEMORY_SPACE MEM>
class AFQMCDriver
{
public:
  AFQMCDriver(std::shared_ptr<utils::mpi_context_t<boost::mpi3::communicator>> _mpi,
              std::string& title,
              int mser,
              int blk0,
              int stp0,
              double eshft_,
              ptree pt_in,
              Wavefunction<MEM>& wfn_,
              Propagator<MEM>& prpg_,
              EstimatorHandler<MEM>& estim_)
      : mpi(_mpi),
        m_series(mser),
        project_title(title),
        block0(blk0),
        step0(stp0),
        wfn0(wfn_),
        prop0(prpg_),
        estim0(estim_),
        weight_reset_period(0.0),
        Eshift(eshft_)
  {
    name = "AFQMCDriver";
    // convert user input to verbose input
    ptree pt = interpret_inputs(pt_in);
    app_log(2, "\nAFQMCDriver input:");
    app_log(2, "{}\n", io::to_string(pt));
    // initialize using verbose input
    hdf_write_restart = pt.get<std::string>("hdf_write_file");
    nStep = pt.get<int>("steps");
    measure_interval_multiplier = pt.get<int>("measure_interval_multiplier");
    nPopulation = pt.get<int>("population_control_interval");
    nStabilize = pt.get<int>("walker_ortho_interval");
    nCheckpoint = pt.get<int>("checkpoint_interval");
    samplePeriod = pt.get<int>("sample_interval");
    weight_reset_period = pt.get<double>("weight_reset"); // in units of time
    dt = pt.get<double>("timestep");
    dShift = pt.get<double>("dshift");  // Etrial shift scale

    // KE: to make sure that all Estimators are measured at their own desired intervals
    _measure_interval = estim0.get_max_common_interval();
    // current implementation assumes that population control is called just before accumulate_step()
    // forcing to be the same interval for now.
    nAccumulate = nPopulation;
    estim0.display_measurement_intervals();
  }

  static ptree interpret_inputs(const ptree pt0)
  {
    // read inputs with default options
    std::string hdf_write_file;
    int steps, measure_interval, measure_interval_multiplier, nPopulation, ortho, checkpoint;
    double weight_reset, timestep, dshift;
    hdf_write_file = pt0.get<std::string>("hdf_write_file", "");
    steps         = pt0.get<int>("steps", 1);
    nPopulation = pt0.get<int>("population_control_interval", DEFAULT_POPULATION_CONTROL_INTERVAL);
    measure_interval_multiplier = pt0.get<int>("measure_interval_multiplier",DEFAULT_MEASURE_INTERVAL_MULTIPLIER);
    ortho         = pt0.get<int>("walker_ortho_interval", DEFAULT_WALKER_ORTHO_INTERVAL);
    checkpoint    = pt0.get<int>("checkpoint_interval", -1);
    //sample_period = pt0.get<int>("sample_interval", -1); // KE: commented until relevant feature is implemented
    weight_reset = pt0.get<double>("weight_reset", 0.0);
    timestep     = pt0.get<double>("timestep", DEFAULT_TIME_STEP);
    dshift       = pt0.get<double>("dshift", 1.0);

    measure_interval = measure_interval_multiplier * nPopulation;
    // if steps and measure_interval are not commensurate, add steps so that
    //  the last block will be.
    if (steps % measure_interval != 0)
    {
      int scale = int(std::ceil((double)steps/(double)measure_interval));
      steps = scale*measure_interval;
      app_log(1, "Warning: 'steps' is not evenly divisible by 'measure_interval'. Setting 'steps' to {} steps \n", steps);
    }

    // create verbose internal inputs
    ptree pt1;
    pt1.put("hdf_write_file", hdf_write_file);
    pt1.put("steps", steps);
    pt1.put("population_control_interval", nPopulation);
    pt1.put("measure_interval_multiplier", measure_interval_multiplier);
    pt1.put("walker_ortho_interval", ortho);
    pt1.put("checkpoint_interval", checkpoint);
    pt1.put("sample_interval", -1); // KE: hardcoded until relevant feature is implemented
    pt1.put("weight_reset", weight_reset);
    pt1.put("timestep", timestep);
    pt1.put("dshift", dshift);
    // check for unknown input keys
    std::unordered_set<std::string> pass_through_keys = {
      "walker_set",
      "wavefunction",
      "propagator",
      "estimator",
      "hamiltonian",
      "seed",
      "n_walkers_per_mpi_task",
      "initial_Eshift"
    };
    io::compare_known_keys("AFQMC Driver",pt1, pt0, pass_through_keys);
    return pt1;
  }

  ~AFQMCDriver() {}

  bool run(WalkerSet<MEM>&);

  bool checkpoint(WalkerSet<MEM>&, int, int);

  bool clear() { return true; };

protected:
  std::shared_ptr<utils::mpi_context_t<boost::mpi3::communicator>> mpi;

  std::string name;

  int m_series;
  std::string project_title;

  std::string hdf_write_restart;

  int nStep;
  int nAccumulate;
  int measure_interval_multiplier;
  int _measure_interval; // determined as `_measure_interval = measure_interval_multiplier*nPopulation`
  int nPopulation;

  int nCheckpoint;
  int nStabilize;
  RealType dt;
  int block0, step0;

  Wavefunction<MEM>& wfn0;

  Propagator<MEM>& prop0;

  EstimatorHandler<MEM>& estim0;

  bool writeSamples(WalkerSet<MEM>&);

  int samplePeriod;

  double weight_reset_period;

  RealType dShift;
  RealType Eshift;
  RealType Etav;
};

} // namespace afqmc
} // namespace sfqmc

