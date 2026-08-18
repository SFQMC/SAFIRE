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
#include "AFQMC/parameters.hpp"
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
              const ExecuteParameters& exec,
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
    hdf_write_restart = exec.hdf_write_file;
    nStep = exec.steps;
    nPopulation = exec.population_control_interval;
    nStabilize = exec.walker_ortho_interval;
    nCheckpoint = exec.checkpoint_interval;
    samplePeriod = -1; // KE: hardcoded until relevant feature is implemented
    weight_reset_period = exec.weight_reset; // in units of time
    dt = exec.timestep;
    dShift = exec.dshift;  // Etrial shift scale

    // if steps and measure_interval are not commensurate, add steps so that
    //  the last block will be.
    const int measure_interval = exec.measure_interval_multiplier * nPopulation;
    if (nStep % measure_interval != 0)
    {
      nStep = measure_interval * int(std::ceil(double(nStep) / double(measure_interval)));
      app_log(1, "Warning: 'steps' is not evenly divisible by 'measure_interval'. Setting 'steps' to {} steps \n", nStep);
    }

    // KE: to make sure that all Estimators are measured at their own desired intervals
    _measure_interval = estim0.get_max_common_interval();
    // current implementation assumes that population control is called just before accumulate_step()
    // forcing to be the same interval for now.
    nAccumulate = nPopulation;
    estim0.display_measurement_intervals();
  }


  bool run(WalkerSet<MEM>&);

  bool checkpoint(WalkerSet<MEM>&, int, int);

  bool clear() { return true; };

protected:
  std::shared_ptr<utils::mpi_context_t<boost::mpi3::communicator>> mpi;

  int m_series;
  std::string project_title;

  std::string hdf_write_restart;

  int nStep;
  int nAccumulate;
  int _measure_interval; // the interval that is commensurate with every estimator
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

