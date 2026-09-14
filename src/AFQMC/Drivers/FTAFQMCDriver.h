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
#include "AFQMC/Estimators/Estimators.hpp"

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
              int blk0,
              int stp0,
              double eshft_,
              const ExecuteParameters& exec,
              Wavefunction<MEM>& wfn_,
              Propagator<MEM>& prpg_,
              Estimators<MEM>& estim_)
      : mpi(_mpi),
        project_title(title),
        hdf_write_restart{exec.hdf_write_file},
        nStep{exec.steps},
        nSweep{exec.sweeps},
        nPopulation{exec.population_control_interval},
        nCheckpoint{exec.checkpoint_interval},
        nStabilize{exec.walker_ortho_interval},
        dt{exec.timestep},
        block0(blk0),
        step0(stp0),
        wfn0(wfn_),
        prop0(prpg_),
        estimators_(estim_),
        dShift{exec.dshift}, // update factor for Eshift
        Eshift(eshft_),
        Eshift0(eshft_),
        print_sweep_step{exec.print_sweep_step}
  {
    // the steps/measure_interval commensurability check that the ground state driver does is
    // not currently relevant for finite-T, but may be useful if backward sweeps are implemented
    // measurements only happen after the full path has been constructed (i.e. at nStep = L)
  }

  bool run(WalkerSet<MEM>&);

  bool checkpoint(WalkerSet<MEM>&, int, int);

  bool clear() { return true; };

protected:
  std::shared_ptr<utils::mpi_context_t<boost::mpi3::communicator>> mpi;

  std::string project_title;

  std::string hdf_write_restart;

  int nStep;
  int nSweep;
  int nPopulation;

  int nCheckpoint;
  int nStabilize;
  RealType dt;
  RealType beta;

  int block0, step0;

  Wavefunction<MEM>& wfn0;

  Propagator<MEM>& prop0;

  Estimators<MEM>& estimators_;

  RealType dShift;
  RealType Eshift;
  RealType Eshift0;
  RealType Etav;

  bool print_sweep_step;
};

} // namespace afqmc
} // namespace sfqmc

