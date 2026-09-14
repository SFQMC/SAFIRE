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

#include <format>
#include <tuple>
#include <map>
#include <string>
#include <iomanip>

#include "config.h"
#include "utilities/check.hpp"
#include "utilities/memory_utils.hpp"

#include "AFQMC/config.h"
#include "IO/app_loggers.h"
#include "IO/banner.hpp"
#include "AFQMC/Utilities/AFQMCTimer.h"
#include "FTAFQMCDriver.h"
#include "averageEloc.hpp"
#include "AFQMC/Walkers/WalkerIO.hpp"

namespace sfqmc
{
namespace afqmc
{
template<MEMORY_SPACE MEM>
bool FTAFQMCDriver<MEM>::run(WalkerSet<MEM>& wset)
{

  app_log(1, banner("Beginning FT-AFQMC calculation"));

  std::vector<ComplexType> curData;

  RealType w0   = wset.GlobalWeight();
  int nwalk_ini = wset.GlobalPopulation();
  int nwalk_ini_per_mpi = nwalk_ini/mpi->comm.size();

   int print_interval = std::max(1, nStep / 20);

  app_log(1, "Initial weight and number of walkers: {}, {}", w0 ,nwalk_ini);
  app_log(1, "Initial Eshift: {} ", Eshift);

  // problems with using step_tot to do ortho and load balance
  double total_time;
  // KE: the concept of a "block" is now implicitly defined by the measure_interval

  beta = dt*nStep;

  app_log(1, "Executing {} sweeps, with Beta = {} ", nSweep, beta);

  for(int iSweep = 0; iSweep < nSweep; ++iSweep) {
    auto block_time = timers.block.start();
    Eshift = Eshift0; // Eshift set to same value at the beginning of each sweep
    total_time = 0.0;
    for (int iStep = 0; iStep < nStep; ++iStep)
    {
      if(iStep % print_interval == 0 and print_sweep_step)
        app_log(1, "sweep {}, step {} ", iSweep, iStep);
      // reset wset log(ovlp), read initial value
      // from memory after sweep 1, rather than re-computing
      if(iStep==0 and iSweep>0){
        auto const& LogPT0 = wfn0.getLogPT0();
        utils::check(LogPT0.size() == wset.size(),
                    "LogPT0 size ({}) does not match walker set size ({})",
                    LogPT0.size(), wset.size());
        wset.setProperty(OVLP, LogPT0);
        wset.setTauStep(0);
      }

      prop0.Propagate(wset, Eshift, dt, iStep+1);
      total_time += dt;

      if ((iStep + 1) % nStabilize == 0 && iStep != nStep - 1 )
      {
        auto ortho_time = timers.ortho.start();
        prop0.Orthogonalize(wset);
        ortho_time.stop();
      }

      if (total_time < 1.0)
      {
        wset.processWalkerData(curData);
        Eshift = averageEloc(*mpi, wset);
      }

      // KE: should there be a check for population control interval here?
      if ((iStep + 1) % nPopulation == 0 || iStep == 0 || iStep == nStep-1)
      {
        auto popcontrol_time = timers.popcontrol.start();
        wset.processWalkerData(curData);
        wset.popControl();
        popcontrol_time.stop();

        if(total_time >= 1.0) {
          Eshift += dShift * (averageEloc(*mpi, wset) - Eshift);
        }
      }

      // resize stack pointers to match maximum buffer use
      utils::resize_nda_static_allocator();
    }

    block_time.stop();

    // one sweep is one measurement sample. The walker set is still at nt = nStep here,
    // i.e. the full path has been constructed.
    estimators_.measure(*mpi, iSweep, wset);

    //add finite-T checkpoint?

    wset.clean(); // reset walker buffer
    // reset weights, UR, DR, VR
    wset.reset(nwalk_ini_per_mpi);
    // reset logsclL, probably only necessary if backward sweeps are implemented
    wfn0.resetLogScale();

  }


  // print timers
  if(mpi->comm.root()) {
    timers.print_all();
    estimators_.write(std::format("{}.results.h5", project_title));
  }

  app_log(1, banner("Finished FT-AFQMC calculation"));

  return true;

}

// Instantiate                                        
template bool FTAFQMCDriver<HOST_MEMORY>::run(WalkerSet<HOST_MEMORY>& wset);
#if defined(ENABLE_DEVICE)
  template bool FTAFQMCDriver<DEVICE_MEMORY>::run(WalkerSet<DEVICE_MEMORY>& wset);
#endif

} // namespace afqmc

} // namespace sfqmc
