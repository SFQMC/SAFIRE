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
#include <optional>
#include <string>
#include <iomanip>

#include "config.h"
#include "utilities/check.hpp"
#include "utilities/memory_utils.hpp"

#include "AFQMC/config.h"
#include "IO/app_loggers.h"
#include "IO/banner.hpp"
#include "AFQMC/Utilities/AFQMCTimer.h"
#include "AFQMCDriver.h"
#include "AFQMC/Walkers/WalkerIO.hpp"

namespace sfqmc
{
namespace afqmc
{
template<MEMORY_SPACE MEM>
bool AFQMCDriver<MEM>::run(WalkerSet<MEM>& wset)
{

  app_log(1, banner("Beginning AFQMC calculation"));

  std::vector<ComplexType> curData;

  RealType w0   = wset.GlobalWeight();
  int nwalk_ini = wset.GlobalPopulation();

  app_log(1, "Initial weight and number of walkers: {}, {}", w0 ,nwalk_ini);
  app_log(1, "Initial Eshift: {} ", Eshift);

  // problems with using step_tot to do ortho and load balance
  double total_time = step0 * dt;
  int step_tot      = step0, iBlock;
  // KE: the concept of a "block" is now implicitly defined by the measure_interval
  iBlock = 0;

  // one timed region per measurement interval, covering every step of the block plus its measurement
  std::optional<utils::Timer::Handle> block_time;

  // KE: need to change the hard-coded 1.0 to an equilibration phase.
  for (int iStep = 0; iStep < nStep; ++iStep, ++step_tot) {
    if (iStep % _measure_interval == 0)
      block_time.emplace(timers.block);

    prop0.Propagate(wset, Eshift, dt);
    total_time += dt;

    if ((step_tot + 1) % nStabilize == 0)
    {
      auto ortho_time = timers.ortho.start();
      prop0.Orthogonalize(wset);
      ortho_time.stop();
    }

    if (total_time < weight_reset_period && !prop0.free_propagation())
      wset.resetWeights();

    if (total_time < 1.0)
    {
      wset.processWalkerData(curData);
      estim0.accumulate_step(total_time, wset, curData);
    }

    if ((iStep + 1) % nPopulation == 0 || iStep == 0)
    {
      auto popcontrol_time = timers.popcontrol.start();
      wset.processWalkerData(curData);
      wset.popControl(); // make this a call to actual pop control
      popcontrol_time.stop();
      estim0.accumulate_step(total_time, wset, curData);
    }

    if (total_time < 1.0)
    {
      Eshift = estim0.getEloc_step();
    }
    else if ((iStep + 1) % nAccumulate == 0)
      Eshift += dShift * (estim0.getEloc_step() - Eshift);


    // checkpoint
    if (nCheckpoint > 0 && (iStep + 1) % nCheckpoint == 0)
      if (!checkpoint(wset, iStep, step_tot))
      {
        app_error("Error in AFQMCDriver::checkpoint(). ");
        app_error_flush();
        return false;
      }

    // write samples
    if (samplePeriod > 0 && (iStep + 1) % samplePeriod == 0)
      if (!writeSamples(wset))
      {
        app_error("Error in AFQMCDriver::writeSamples(). ");
        app_error_flush();
        return false;
      }

    // resize stack pointers to match maximum buffer use
    // UPDATE size of dynamic_bucket inside fallback allocator!!!
    utils::resize_nda_static_allocator();

    if ((iStep + 1) % _measure_interval == 0 )
    {
      // quantities that are measured once per block
      estim0.accumulate_block(total_time, wset);
      estim0.print(iBlock + 1, total_time, Eshift, wset);
      iBlock++;
      block_time->stop();
      block_time.reset();
    }
  }
  // steps left over by an nStep that is not a multiple of the interval
  if(block_time) {
    block_time->stop();
    block_time.reset();
  }

  if (nCheckpoint > 0)
    checkpoint(wset, iBlock, step_tot);

  prop0.printBoundStatistics();
  // print timers
  if(mpi->comm.root()) timers.print_all();

  app_log(1, banner("Finished AFQMC calculation"));

  return true;
}

// writes checkpoint file
template<MEMORY_SPACE MEM>
bool AFQMCDriver<MEM>::checkpoint(WalkerSet<MEM>& wset, int block, int step)
{
return true;
  if (mpi->comm.rank() == 0)
  {
    std::string file;
    if (hdf_write_restart != std::string(""))
      file = hdf_write_restart;
    else
      file = project_title + std::string(".chk.h5");

    std::vector<RealType> Rdata(2);
    Rdata[0] = Eshift;
    Rdata[1] = Eshift;

    std::vector<IndexType> Idata(2);
    Idata[0] = block;
    Idata[1] = step;

    // always write driver data and walkers
    h5::file h5f(file,'a');
    h5::group grp(h5f);
    h5::group dgrp = ( grp.has_key("AFQMCDriver") ?
                       grp.open_group("AFQMCDriver") :
                       grp.create_group("AFQMCDriver") );
    h5::h5_write(dgrp,"DriverInts",Idata);
    h5::h5_write(dgrp,"DriverReals",Rdata);

    return dumpToHDF5(wset, h5f);
  } else {
    h5::file h5f; 
    return dumpToHDF5(wset, h5f);
  }
}

// writes samples
template<MEMORY_SPACE MEM>
bool AFQMCDriver<MEM>::writeSamples(WalkerSet<MEM>& wset)
{
return true;
  int nwtowrite = -1;
  if (mpi->comm.rank() == 0)
  {
    std::string file;
    file = project_title + std::string(".confg.h5");
    h5::file h5f(file,'a');
    return dumpSamplesHDF5(wset, h5f, nwtowrite);
  } else {
    h5::file h5f; 
    return dumpSamplesHDF5(wset, h5f, nwtowrite);
  }
}

// Instantiate
#define __inst__(M)                                            \
template bool AFQMCDriver<M>::run(WalkerSet<M>& wset);            \
template bool AFQMCDriver<M>::checkpoint(WalkerSet<M>&,int,int);  \
template bool AFQMCDriver<M>::writeSamples(WalkerSet<M>&);        

__inst__(HOST_MEMORY)
#if defined(ENABLE_DEVICE)
__inst__(DEVICE_MEMORY)
#endif

} // namespace afqmc

} // namespace sfqmc
