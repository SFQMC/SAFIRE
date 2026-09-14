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

#include <chrono>
#include <format>
#include <string>
#include <string_view>

#include "config.h"
#include "utilities/check.hpp"
#include "utilities/memory_utils.hpp"

#include "AFQMC/config.h"
#include "IO/app_loggers.h"
#include "IO/banner.hpp"
#include "AFQMC/Utilities/AFQMCTimer.h"
#include "AFQMCDriver.h"
#include "averageEloc.hpp"
#include "AFQMC/Walkers/WalkerIO.hpp"

namespace sfqmc::afqmc {

template<MEMORY_SPACE MEM>
bool AFQMCDriver<MEM>::run(WalkerSet<MEM>& wset) {
  app_log(1, banner("Beginning AFQMC calculation"));

  std::vector<ComplexType> curData;

  RealType w0   = wset.GlobalWeight();
  int nwalk_ini = wset.GlobalPopulation();

  app_log(1, "Initial weight and number of walkers: {}, {}", w0 ,nwalk_ini);
  app_log(1, "Initial Eshift: {} ", Eshift);

  // problems with using step_tot to do ortho and load balance
  double total_time = step0 * dt;
  int step_tot      = step0;

  propagator_.generateP1(dt, wset.getWalkerType());
  
  const int log_interval = std::max(1, nStep / 100);
  const int steps_total  = nStep + step0;
  const int step_format_width   = int(std::to_string(steps_total).size());

  // three equal columns tiling the full rule width, shared by the header and the rows
  constexpr int log_col = default_banner_width / 3;
  static_assert(3 * log_col == default_banner_width, "columns must tile the rule exactly");
  constexpr std::string_view log_row = "{:<{}}{:>{}}{:>{}}";

  app_log(2, hrule());
  app_log(2, log_row, "Wall clock", log_col, "Step", log_col, "Energy", log_col);
  app_log(2, hrule());

  // KE: need to change the hard-coded 1.0 to an equilibration phase.
  for (int iStep = 0; iStep < nStep; ++iStep, ++step_tot) {
    propagator_.Propagate(wset, Eshift, dt);
    total_time += dt;

    if ((step_tot + 1) % nStabilize == 0) {
      auto ortho_time = timers.ortho.start();
      propagator_.Orthogonalize(wset);
      ortho_time.stop();
    }

    if (total_time < 1.0) {
      wset.processWalkerData(curData);
      Eshift = averageEloc(*mpi_, wset);
    }

    if ((iStep + 1) % nPopulation == 0 || iStep == 0) {
      auto popcontrol_time = timers.popcontrol.start();
      wset.processWalkerData(curData);
      wset.popControl(); // make this a call to actual pop control
      popcontrol_time.stop();

      if(iStep >= nEquilibration) {
        estimators_.measure(*mpi_, iStep / nPopulation, wset);
      } else {
        Eshift += dShift * (averageEloc(*mpi_, wset) - Eshift);
      }   
    }

    // checkpoint
    if (nCheckpoint > 0 && (iStep + 1) % nCheckpoint == 0) {
      if (!checkpoint(wset, iStep, step_tot))
      {
        app_error("Error in AFQMCDriver::checkpoint(). ");
        app_error_flush();
        return false;
      }
    }

    if(iStep % log_interval == 0) {
      const double energy = averageEloc(*mpi_, wset);
      const auto now = std::chrono::current_zone()->to_local(
          std::chrono::floor<std::chrono::seconds>(std::chrono::system_clock::now()));

      // app_log formats through spdlog's bundled fmt, which has no chrono formatter here,
      // so the timestamp is rendered by std::format and passed on as a string
      app_log(2, log_row, std::format("{:%F %T}", now), log_col,
              std::format("{:>{}}/{}", step_tot + 1, step_format_width, steps_total), log_col,
              std::format("{:#.8g}", energy), log_col);
    }

    // resize stack pointers to match maximum buffer use
    utils::resize_nda_static_allocator();
  }
  app_log(2, hrule());

  // steps left over by an nStep that is not a multiple of the interval

  if (nCheckpoint > 0)
    checkpoint(wset, step_tot/nPopulation, step_tot);

  propagator_.printBoundStatistics();
  // print timers
  if(mpi_->comm.root()){
    timers.print_all();
    estimators_.write(std::format("{}.results.h5", project_title_));
  }

  app_log(1, banner("Finished AFQMC calculation"));

  return true;
}

// writes checkpoint file
template<MEMORY_SPACE MEM>
bool AFQMCDriver<MEM>::checkpoint(WalkerSet<MEM>& wset, int block, int step)
{
return true;
  if (mpi_->comm.rank() == 0)
  {
    std::string file;
    if (hdf_write_restart != std::string(""))
      file = hdf_write_restart;
    else
      file = project_title_ + std::string(".chk.h5");

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

// Instantiate
#define __inst__(M)                                            \
template bool AFQMCDriver<M>::run(WalkerSet<M>& wset);            \
template bool AFQMCDriver<M>::checkpoint(WalkerSet<M>&,int,int);

__inst__(HOST_MEMORY)
#if defined(ENABLE_DEVICE)
__inst__(DEVICE_MEMORY)
#endif

} // namespace sfqmc::afqmc
