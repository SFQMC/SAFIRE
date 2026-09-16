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

#include <algorithm>
#include <format>
#include <string>
#include <vector>

#include "config.h"
#include "utilities/check.hpp"
#include "utilities/memory_utils.hpp"

#include "AFQMC/config.h"
#include "AFQMC/Drivers/averageEloc.hpp"
#include "AFQMC/Drivers/run_ftafqmc.hpp"
#include "AFQMC/Utilities/AFQMCTimer.h"
#include "IO/app_loggers.h"
#include "IO/banner.hpp"

namespace sfqmc::afqmc {

template<MEMORY_SPACE MEM>
void run_ftafqmc(utils::mpi_context_t<boost::mpi3::communicator>& mpi,
                 std::string const& output_name,
                 ExecuteParameters const& exec,
                 RealType Eshift,
                 WalkerSet<MEM>& wset,
                 Wavefunction<MEM>& wavefunction,
                 Propagator<MEM>& propagator,
                 Estimators<MEM>& estimators) {
  app_log(1, banner("Beginning FT-AFQMC calculation"));

  std::vector<ComplexType> curData;

  RealType w0   = wset.GlobalWeight();
  int nwalk_ini = wset.GlobalPopulation();
  int nwalk_ini_per_mpi = nwalk_ini / mpi.comm.size();

  int print_interval = std::max(1, exec.steps / 20);

  app_log(1, "Initial weight and number of walkers: {}, {}", w0, nwalk_ini);
  app_log(1, "Initial Eshift: {} ", Eshift);

  // KE: the concept of a "block" is now implicitly defined by the measure_interval
  const RealType Eshift0 = Eshift;
  const RealType beta    = exec.timestep * exec.steps;

  app_log(1, "Executing {} sweeps, with Beta = {} ", exec.sweeps, beta);

  for(int iSweep = 0; iSweep < exec.sweeps; ++iSweep) {
    Eshift = Eshift0; // Eshift set to same value at the beginning of each sweep
    double total_time = 0.0;
    for(int iStep = 0; iStep < exec.steps; ++iStep) {
      auto step_time = timers.step.start();
      if(iStep % print_interval == 0 && exec.print_sweep_step) {
        app_log(1, "sweep {}, step {} ", iSweep, iStep);
      }
      // reset wset log(ovlp), read initial value
      // from memory after sweep 1, rather than re-computing
      if(iStep == 0 && iSweep > 0) {
        auto const& LogPT0 = wavefunction.getLogPT0();
        utils::check(LogPT0.size() == wset.size(),
                     "LogPT0 size ({}) does not match walker set size ({})",
                     LogPT0.size(), wset.size());
        wset.setProperty(OVLP, LogPT0);
        wset.setTauStep(0);
      }

      propagator.Propagate(wset, Eshift, iStep + 1);
      total_time += exec.timestep;

      if((iStep + 1) % exec.walker_ortho_interval == 0 && iStep != exec.steps - 1) {
        auto ortho_time = timers.ortho.start();
        propagator.Orthogonalize(wset);
        ortho_time.stop();
      }

      if(total_time < 1.0) {
        wset.processWalkerData(curData);
        Eshift = averageEloc(mpi, wset);
      }

      // KE: should there be a check for population control interval here?
      if((iStep + 1) % exec.population_control_interval == 0 || iStep == 0 || iStep == exec.steps - 1) {
        auto popcontrol_time = timers.popcontrol.start();
        wset.processWalkerData(curData);
        wset.popControl();
        popcontrol_time.stop();

        if(total_time >= 1.0) {
          Eshift += exec.dshift * (averageEloc(mpi, wset) - Eshift);
        }
      }

      // resize stack pointers to match maximum buffer use
      utils::resize_nda_static_allocator();
    }

    // one sweep is one measurement sample. The walker set is still at nt = nStep here,
    // i.e. the full path has been constructed.
    estimators.measure(mpi, iSweep, wset);

    wset.clean(); // reset walker buffer
    // reset weights, UR, DR, VR
    wset.reset(nwalk_ini_per_mpi);
    // reset logsclL, probably only necessary if backward sweeps are implemented
    wavefunction.resetLogScale();
  }

  // print timers
  if(mpi.comm.root()) {
    timers.print_all();
    estimators.write(std::format("{}.results.h5", output_name));
  }

  app_log(1, banner("Finished FT-AFQMC calculation"));
}

// Instantiate
#define __inst__(M)                                                                                  \
  template void run_ftafqmc<M>(utils::mpi_context_t<boost::mpi3::communicator>&, std::string const&,  \
                               ExecuteParameters const&, RealType, WalkerSet<M>&, Wavefunction<M>&,   \
                               Propagator<M>&, Estimators<M>&);

__inst__(HOST_MEMORY)
#if defined(ENABLE_DEVICE)
__inst__(DEVICE_MEMORY)
#endif

} // namespace sfqmc::afqmc
