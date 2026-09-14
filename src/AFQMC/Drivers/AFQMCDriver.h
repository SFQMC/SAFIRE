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
class AFQMCDriver
{
public:
  AFQMCDriver(std::shared_ptr<utils::mpi_context_t<boost::mpi3::communicator>> mpi,
              std::string& title,
              int stp0,
              double eshft_,
              const ExecuteParameters& exec,
              Wavefunction<MEM>& wfn_,
              Propagator<MEM>& prpg_,
              Estimators<MEM>& estim_)
      : mpi_(mpi),
        output_name_(title),
        nStep{exec.steps},
        nPopulation{exec.population_control_interval},
        nEquilibration{exec.equilibration_steps},
        nCheckpoint{exec.checkpoint_interval},        
        nStabilize{exec.walker_ortho_interval},
        dt{exec.timestep},
        step0(stp0),
        wavefunction_(wfn_),
        propagator_(prpg_),
        estimators_(estim_),
        dShift{exec.dshift}, // update factor for Eshift
        Eshift(eshft_)
  {
  }


  bool run(WalkerSet<MEM>&);

  bool checkpoint(WalkerSet<MEM>&, int, int);

  bool clear() { return true; };

protected:
  std::shared_ptr<utils::mpi_context_t<boost::mpi3::communicator>> mpi_;

  std::string output_name_{};

  std::string hdf_write_restart;

  int nStep;
  int nPopulation;
  int nEquilibration{};

  int nCheckpoint;
  int nStabilize;
  RealType dt;
  int step0;

  Wavefunction<MEM>& wavefunction_;
  Propagator<MEM>& propagator_;
  Estimators<MEM>& estimators_;

  bool writeSamples(WalkerSet<MEM>&);

  RealType dShift;
  RealType Eshift;
  RealType Etav{};
};

} // namespace afqmc
} // namespace sfqmc

