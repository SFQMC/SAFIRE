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

#include <tuple>
#include <map>
#include <string>
#include <iomanip>

#include "config.h"
#include "utilities/check.hpp"
#include "utilities/memory_utils.hpp"

#include "AFQMC/config.h"
#include "IO/app_loggers.h"
#include "AFQMC/Utilities/AFQMCTimer.h"
#include "FTAFQMCDriver.h"
#include "AFQMC/Walkers/WalkerIO.hpp"

namespace sfqmc
{
namespace afqmc
{
template<MEMORY_SPACE MEM>
bool FTAFQMCDriver<MEM>::run(WalkerSet<MEM>& wset)
{

  app_log(1,"****************************************************");
  app_log(1,"              Beginning FT-AFQMC calculation        ");
  app_log(1,"****************************************************");

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
    AFQMCTimer.start(block_timer);
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
        AFQMCTimer.start(ortho_timer);
        prop0.Orthogonalize(wset);
        AFQMCTimer.stop(ortho_timer);
      }

      if (total_time < weight_reset_period && !prop0.free_propagation()){
        wset.resetWeights();
      }

      if (total_time < 1.0) 
      {
        wset.processWalkerData(curData);
        estim0.accumulate_step(total_time, wset, curData);
      } 

      // KE: should there be a check for population control interval here?
      if ((iStep + 1) % nPopulation == 0 || iStep == 0 || iStep == nStep-1)
      {
        AFQMCTimer.start(popcont_timer);
        wset.processWalkerData(curData);
        wset.popControl();
        AFQMCTimer.stop(popcont_timer);
        estim0.accumulate_step(total_time,wset,curData);
      }

      //estim0.accumulate_step(total_time,wset,curData);

      if (total_time < 1.0)
      {
        Eshift = estim0.getEloc_step();
      }
      else if ((iStep + 1) % nAccumulate == 0)
        Eshift += dShift * (estim0.getEloc_step() - Eshift);

      //estim0.print_walker_info(iSweep+1, total_time);

      // resize stack pointers to match maximum buffer use
      //update_memory_managers();
      utils::resize_nda_static_allocator();
    }

    AFQMCTimer.stop(block_timer);
    
    // accumulate measurements
    // for measurement we need nt = nStep
    //wset.advanceTauStep();
    estim0.accumulate_block(double(nStep), wset);
    estim0.print(iSweep + 1, total_time, Eshift, wset);
    
    //add finite-T checkpoint?

    wset.clean(); // reset walker buffer
    // reset weights, UR, DR, VR
    wset.reset(nwalk_ini_per_mpi);
    // reset logsclL, probably only necessary if backward sweeps are implemented
    wfn0.resetLogScale();

  }


  // print timers
  if(mpi->comm.root()) AFQMCTimer.print_all();
  
  app_log(1,"****************************************************");
  app_log(1,"               Finished AFQMC calculation           ");
  app_log(1,"****************************************************");

  return true;

}

// Instantiate                                        
template bool FTAFQMCDriver<HOST_MEMORY>::run(WalkerSet<HOST_MEMORY>& wset);
#if defined(ENABLE_DEVICE)
  template bool FTAFQMCDriver<DEVICE_MEMORY>::run(WalkerSet<DEVICE_MEMORY>& wset);
#endif

} // namespace afqmc

} // namespace sfqmc