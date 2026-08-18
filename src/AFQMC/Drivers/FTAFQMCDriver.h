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
class FTAFQMCDriver
{
public:
  FTAFQMCDriver(std::shared_ptr<utils::mpi_context_t<boost::mpi3::communicator>> _mpi,
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
        Eshift(eshft_),
        Eshift0(eshft_)
  {
    name = "FTAFQMCDriver";
    hdf_write_restart = exec.hdf_write_file;
    nStep = exec.steps;
    nSweep = exec.sweeps;
    measure_interval_multiplier = exec.measure_interval_multiplier;
    nPopulation = exec.population_control_interval;
    nStabilize = exec.walker_ortho_interval;
    nCheckpoint = exec.checkpoint_interval;
    samplePeriod = -1; // KE: hardcoded until relevant feature is implemented
    weight_reset_period = exec.weight_reset; // in units of time
    dt = exec.timestep;
    dShift = exec.dshift;  // Etrial shift scale
    print_sweep_step = exec.print_sweep_step;

    // the steps/measure_interval commensurability check that the ground state driver does is
    // not currently relevant for finite-T, but may be useful if backward sweeps are implemented

    // KE: to make sure that all Estimators are measured at their own desired intervals
    _measure_interval = estim0.get_max_common_interval();
    // current implementation assumes that population control is called just before accumulate_step()
    // forcing to be the same interval for now.
    nAccumulate = nPopulation;
    // measurements only happen after full path has been constructed (i.e. at nStep = L)
    //estim0.display_measurement_intervals();
  }

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
  int nSweep;
  int nAccumulate;
  int measure_interval_multiplier;
  int _measure_interval; // determined as `_measure_interval = measure_interval_multiplier*nPopulation`
  int nPopulation;

  int nCheckpoint;
  int nStabilize;
  RealType dt;
  RealType beta;

  int block0, step0;

  Wavefunction<MEM>& wfn0;

  Propagator<MEM>& prop0;

  EstimatorHandler<MEM>& estim0;

  bool writeSamples(WalkerSet<MEM>&);

  int samplePeriod;

  double weight_reset_period;

  RealType dShift;
  RealType Eshift;
  RealType Eshift0;
  RealType Etav;

  bool print_sweep_step;
};

} // namespace afqmc
} // namespace sfqmc

