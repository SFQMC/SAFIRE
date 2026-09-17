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
#include <chrono>
#include <format>
#include <string>
#include <string_view>

#include "config.h"
#include "utilities/memory_utils.hpp"

#include "AFQMC/config.h"
#include "AFQMC/Drivers/average_energy.hpp"
#include "AFQMC/Drivers/run_afqmc.hpp"
#include "AFQMC/Utilities/AFQMCTimer.h"
#include "IO/app_loggers.h"
#include "IO/banner.hpp"

namespace sfqmc::afqmc {

template<MEMORY_SPACE MEM>
void run_afqmc(utils::mpi_context_t<boost::mpi3::communicator>& mpi,
               std::string const& output_name,
               ExecuteParameters const& exec,
               RealType Eshift,
               WalkerSet<MEM>& wset,
               Propagator<MEM>& propagator,
               Estimators<MEM>& estimators) {
  app_log(1, banner("Beginning AFQMC calculation"));

  RealType w0   = wset.GlobalWeight();
  int nwalk_ini = wset.GlobalPopulation();

  app_log(1, "Initial weight and number of walkers: {}, {}", w0, nwalk_ini);
  app_log(1, "Initial Eshift: {} ", Eshift);

  const int log_interval = std::max(1, exec.steps / 100);
  const int step_format_width = int(std::to_string(exec.steps).size());

  const double Eshift_relaxation_factor = resolved(exec.Eshift_relaxation_factor, "Eshift_relaxation_factor");

  // three equal columns tiling the full rule width, shared by the header and the rows
  constexpr int log_col = default_banner_width / 3;
  static_assert(3 * log_col == default_banner_width, "columns must tile the rule exactly");
  constexpr std::string_view log_row = "{:<{}}{:>{}}{:>{}}";

  app_log(2, hrule());
  app_log(2, log_row, "Wall clock", log_col, "Step", log_col, "Energy", log_col);
  app_log(2, hrule());

  for(int iStep = 0; iStep < exec.steps; ++iStep) {
    auto step_time = timers.step.start();
    propagator.Propagate(wset, Eshift);

    if((iStep + 1) % exec.walker_ortho_interval == 0) {
      auto ortho_time = timers.ortho.start();
      propagator.Orthogonalize(wset);
      ortho_time.stop();
    }

    if((iStep + 1) % exec.population_control_interval == 0 || iStep == 0) {
      auto popcontrol_time = timers.popcontrol.start();
      wset.popControl();
      wset.rescale_total_weight();
      popcontrol_time.stop();

      if(iStep >= exec.equilibration_steps) {
        estimators.measure(mpi, iStep / exec.population_control_interval, wset);
      } else {
        Eshift += Eshift_relaxation_factor * (averagePseudoEnergy(mpi, wset) - Eshift);
      }
    }

    if(iStep % log_interval == 0) {
      const double energy = averagePseudoEnergy(mpi, wset);
      const auto now = std::chrono::current_zone()->to_local(
          std::chrono::floor<std::chrono::seconds>(std::chrono::system_clock::now()));

      // app_log formats through spdlog's bundled fmt, which has no chrono formatter here,
      // so the timestamp is rendered by std::format and passed on as a string
      app_log(2, log_row, std::format("{:%F %T}", now), log_col,
              std::format("{:>{}}/{}", iStep + 1, step_format_width, exec.steps), log_col,
              std::format("{:#.15g}", energy), log_col);
    }

    // resize stack pointers to match maximum buffer use
    utils::resize_nda_static_allocator();
  }
  app_log(2, hrule());

  if(mpi.comm.root()) {
    std::string results_filename = std::format("{}.results.h5", output_name);
    estimators.write(results_filename);
    app_log(1, "Results written to '{}'.", results_filename);
  }

  propagator.printBoundStatistics(Eshift);

  if(mpi.comm.root()) {
    timers.print_all();
  }

  app_log(1, banner("Finished AFQMC calculation"));
}

// Instantiate
#define __inst__(M)                                                                                \
  template void run_afqmc<M>(utils::mpi_context_t<boost::mpi3::communicator>&, std::string const&,  \
                             ExecuteParameters const&, RealType, WalkerSet<M>&, Propagator<M>&,     \
                             Estimators<M>&);

__inst__(HOST_MEMORY)
#if defined(ENABLE_DEVICE)
__inst__(DEVICE_MEMORY)
#endif

} // namespace sfqmc::afqmc
