/*
 * This file is distributed under the Apache License, Version 2.0 License.
 * See LICENSE file in top directory for details.
 *
 * Copyright (c) 2021-2025 The Simons Foundation, Inc.
 *
 * You may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *      http://www.apache.org/licenses/LICENSE-2.0
 *
 */

#include <iostream>
#include <fstream>
#include <tuple>
#include <map>
#include <string>
#include <iomanip>

#include "config.h"
#include "IO/AppAbort.hpp"

#include "AFQMC/config.h"
#include "IO/app_loggers.h"
#include "IO/banner.hpp"
#include "AFQMC/Utilities/AFQMCTimer.h"
#include "CSAFQMCDriver.h"
#include "AFQMC/Walkers/WalkerIO.hpp"
#include "AFQMC/Walkers/WalkerControl.hpp"
#include "Memory/buffer_managers.h"

namespace sfqmc
{
namespace afqmc
{
bool CSAFQMCDriver::run(std::vector<std::reference_wrapper<WalkerSet>>& wset_ref)
{
  int nsys = prop_ref.size(); 
  if( wset_ref.size() != nsys )
    APP_ABORT("Error in CSAFQMCDriver::run wset_ref.size() != nsys");

  app_log(1, banner("Beginning AFQMC calculation"));

  std::vector<ComplexType> curData(7); 
  Matrix<ComplexType> wData({nsys,7});

  // problems with using step_tot to do ortho and load balance
  double total_time = step0 * nSubstep * dt;
  int step_tot      = step0, iBlock;
  for (iBlock = block0; iBlock < nBlock; ++iBlock)
  {
    AFQMCTimer.start(block_timer);
    for (int iStep = 0; iStep < nStep; ++iStep, ++step_tot)
    {
      // propagate nSubstep
      // save RNG state
      for( int s = 0; s<nsys; s++ ) 
      {
        prop_ref[s].get().Propagate(nSubstep, wset_ref[s].get(), 
				   Eshift[s], dt, fix_bias);
      }
      total_time += nSubstep * dt;

      if ((step_tot + 1) % nStabilize == 0)
      {
        AFQMCTimer.start(ortho_timer);
        for( int s = 0; s<nsys; s++ ) 
          prop_ref[s].get().Orthogonalize(wset_ref[s].get());
        AFQMCTimer.stop(ortho_timer);
      }

      for( int s = 0; s<nsys; s++ ) 
        if (total_time < weight_reset_period && !prop_ref[s].get().free_propagation())
          wset_ref[s].get().resetWeights();

      AFQMCTimer.start(popcont_timer);
      correlatedPopulationControl(wset_ref, wData, combine_type);
//      for( int s = 0; s<nsys; s++ ) 
//        wset_ref[s].get().popControl(curData);
      AFQMCTimer.stop(popcont_timer);
      for( int s = 0; s<nsys; s++ ) { 
        std::copy_n(wData[s].origin(), 7, curData.data());
        estimators[s].accumulate_step(total_time, wset_ref[s].get(), curData);
      } 

      for( int s = 0; s<nsys; s++ ) 
      {
        if (total_time < 1.0)
          Eshift[s] = estimators[s].getEloc_step();
        else
          Eshift[s] += dShift * (estimators[s].getEloc_step() - Eshift[s]);
      }
    }
   // checkpoint
    if (nCheckpoint > 0 && (iBlock + 1) % nCheckpoint == 0)
      for( int s = 0; s<nsys; s++ )
        if (!checkpoint(wset_ref[s].get(), iBlock, step_tot))
        {
          app_error(" Error in CSAFQMCDriver::checkpoint(). ");
	  app_error_flush();
          return false;
        }

    // write samples
    if (samplePeriod > 0 && (iBlock + 1) % samplePeriod == 0)
      for( int s = 0; s<nsys; s++ )
        if (!writeSamples(wset_ref[s].get()))
        {
          app_error(" Error in CSAFQMCDriver::writeSamples(). ");
     	  app_error_flush();
          return false;
        }

    // quantities that are measured once per block
    for( int s = 0; s<nsys; s++ )
      estimators[s].accumulate_block(total_time, wset_ref[s].get());

    // resize stack pointers to match maximum buffer use
    update_memory_managers();

    AFQMCTimer.stop(block_timer);
    for( int s = 0; s<nsys; s++ )
      estimators[s].print(iBlock + 1, total_time, Eshift[s], wset_ref[s].get());

    // reset all systems to system 0, e.g. re-initialize CS from this point 
    if (resetPeriod > 0 && (iBlock + 1) % resetPeriod == 0) {
      app_log(1,"Resetting CS.");
      int nw = wset_ref[0].get().size();
      DeviceBufferManager buffer_manager;
      using buffer_alloc_type     = DeviceBufferManager::template allocator_t<ComplexType>;
      StaticMatrix<ComplexType, buffer_alloc_type>  buff({5,nw},
                      buffer_manager.get_generator().template get_allocator<ComplexType>());      
      bool coll = (wset_ref[0].get().getWalkerType() == COLLINEAR);
      bool bprop = (wset_ref[0].get().single_walker_bp_size() > 0);
      // SM
      auto const&& SM0a = wset_ref[0].get().SlaterMatrices(Alpha);
      auto const&& SM0b = wset_ref[0].get().SlaterMatrices(coll?Beta:Alpha);
      // W
      wset_ref[0].get().getProperty(WEIGHT,buff[0]); 
      // pseudo_eloc
      wset_ref[0].get().getProperty(PSEUDO_ELOC_,buff[1]); 
      // Overlap
      wset_ref[0].get().getProperty(OVLP,buff[2]); 
      // If reweight, W[i] = W[i] * Ov_new[i] / Ov_old[i] 
      if(reset_reweight) {
	// buff[0][i] = W[i] / Ov_old[i]
	ma::elementwise(ma::TOp_DIV,ComplexType(1.0),buff[2],buff[0]);   
      }
      for( int s = 1; s<nsys; s++ ) {
        wset_ref[s].get().SlaterMatrices(Alpha) = SM0a;
        if(coll) wset_ref[s].get().SlaterMatrices(Beta) = SM0b; 
        wfn_ref[s].get().Overlap(wset_ref[s].get()); 
        wfn_ref[s].get().Energy(wset_ref[s].get()); 
        if(reset_reweight) {
          buff[5] = buff[0];
          wset_ref[s].get().getProperty(OVLP,buff[3]); 
          // buff[4][i] = W'[i] * Ov_new[i]  
	  ma::elementwise(ma::TOp_MUL,ComplexType(1.0),buff[3],buff[5]);   
          wset_ref[s].get().setProperty(WEIGHT,buff[5]);
	} else {
          wset_ref[s].get().setProperty(WEIGHT,buff[0]);
        }
        wset_ref[s].get().setProperty(PSEUDO_ELOC_,buff[1]);
        if( bprop ) { //getWeightFactors getWeightHistory
          wset_ref[s].get().getWeightFactors() = wset_ref[0].get().getWeightFactors();
          wset_ref[s].get().getWeightHistory() = wset_ref[0].get().getWeightHistory();
        } 
      }
    }

  }

  if (nCheckpoint > 0)
    for( int s = 0; s<nsys; s++ )
      checkpoint(wset_ref[s].get(), iBlock, step_tot);

  // print timers
  if(globalComm.root()) AFQMCTimer.print_all();
  
  app_log(1, banner("Finished AFQMC calculation"));

  return true;
}

// writes checkpoint file
bool CSAFQMCDriver::checkpoint([[maybe_unused]] WalkerSet& wset, 
                               [[maybe_unused]] int block,
                               [[maybe_unused]] int step)
{
/*
  // hack until hdf_archive works with mpi3
  hdf_archive dump(globalComm, false);
  if (globalComm.rank() == 0)
  {
    std::string file;
    if (hdf_write_restart != std::string(""))
      file = hdf_write_restart;
    else
      file = project_title + std::string(".chk.h5");

    if (!dump.create(file))
    {
      app_error(" Error opening checkpoint file for write. ");
      app_error_flush();
      return false;
    }

    std::vector<RealType> Rdata(2);
    Rdata[0] = Eshift;
    Rdata[1] = Eshift;

    std::vector<IndexType> Idata(2);
    Idata[0] = block;
    Idata[1] = step;

    // always write driver data and walkers
    dump.push("CSAFQMCDriver");
    dump.write(Idata, "DriverInts");
    dump.write(Rdata, "DriverReals");
    dump.pop();
  }

  if (!dumpToHDF5(wset, dump))
  {
    app_error(" Problems writing checkpoint file in Driver/CSAFQMCDriver::checkpoint(). ");
    app_error_flush();
    return false;
  }

  if (globalComm.rank() == 0)
  {
    dump.close();
  }

*/
  return true;
}


// writes samples
bool CSAFQMCDriver::writeSamples([[maybe_unused]] WalkerSet& wset)
{
/*
  // hack until hdf_archive works with mpi3
  hdf_archive dump(globalComm, false);
  if (globalComm.rank() == 0)
  {
    std::string file;
    file = project_title + std::string(".confg.h5");

    if (!dump.create(file))
    {
      app_error(" Error opening checkpoint file for write. ");
      app_error_flush();
      return false;
    }
  }

  int nwtowrite = -1;
  if (!dumpSamplesHDF5(wset, dump, nwtowrite))
  {
    app_error(" Problems writing checkpoint file in Driver/CSAFQMCDriver::writeSample(). ");
    app_error_flush();
    return false;
  }

  if (globalComm.rank() == 0)
  {
    dump.close();
  }
*/
  return true;
}

bool CSAFQMCDriver::clear() { return true; }

} // namespace afqmc

} // namespace sfqmc
