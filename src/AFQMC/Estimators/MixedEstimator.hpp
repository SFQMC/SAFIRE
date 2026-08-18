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

#pragma once

#include "AFQMC/config.h"
#include <vector>
#include <string>
#include <iostream>

#include "AFQMC/parameters.hpp"
#include "utilities/check.hpp"
#include "utilities/mpi_context.h"
#include "nda/nda.hpp"
#include "nda/h5.hpp"

#include "AFQMC/Utilities/AFQMCTimer.h"
#include "AFQMC/Estimators/MixedObsHandler.hpp"
#include "AFQMC/Wavefunctions/Wavefunction.hpp"
#include "AFQMC/Walkers/WalkerSet.hpp"

namespace sfqmc
{
namespace afqmc
{
/*
 * Top class for mixed estimators. 
 * An instance of this class will manage a set of observables evaluated at the mixed distribution.
 */
template<MEMORY_SPACE MEM>
class MixedEstimator : public EstimatorBase<MEM>
{

public:
  /// The measurement and equilibration intervals of the input are multiples of
  /// population_control_interval, which is given in steps.
  MixedEstimator(std::shared_ptr<utils::mpi_context_t<boost::mpi3::communicator>> mpi,
                          std::string name,
                          const EstimatorParameters& params,
                          int population_control_interval,
                          WALKER_TYPES wlk,
                          Wavefunction<MEM>& wfn)
      : observ0(mpi, name, params, wlk, wfn.getNMO(), wfn)
  {
    const std::vector<int> multipliers = measure_interval_multipliers(params);
    utils::check(multipliers.size() == 1,
                 "Error in MixedEstimator user input: 'measure_interval_multiplier' has to be a single value");
    measure_interval = multipliers[0] * population_control_interval;
    const int equil_steps = params.equil_multiplier * population_control_interval;
    utils::check(equil_steps % measure_interval==0,"Error in MixedEstimator user input: 'equil_multiplier' must be evenly divisible by 'measure_interval_multiplier'");
    nblocks_skip = equil_steps / measure_interval;
    writer = (mpi->comm.rank() == 0);
  }

  void accumulate_step([[maybe_unused]] double time, 
                       [[maybe_unused]] WalkerSet<MEM>& wset,
                       [[maybe_unused]] std::vector<ComplexType>& curData) {}

  void accumulate_block([[maybe_unused]] double time, WalkerSet<MEM>& wset)
  {
    accumulated_in_last_block = false;

    // 0. skip if requested
    // MAM: problematic on restarts!!!
    if (iblock < nblocks_skip) { 
      iblock++;
      return;
    }

    auto mixed_estimator_time = timers.mixed_estimator.start();
    observ0.accumulate(wset);
    iblock++;
    accumulated_in_last_block = true;
    mixed_estimator_time.stop();
  }

  void tags([[maybe_unused]] std::ofstream& out)
  {
  }

  int get_measurement_interval() { 
    return measure_interval;
  }

  void print([[maybe_unused]] std::ofstream& out, h5::file& file, [[maybe_unused]] WalkerSet<MEM>& wset)
  {
    // print resets the counters for block average.
    if (accumulated_in_last_block)
    {
      if (writer)
      {
        h5::group grp(file);
        h5::group g1 = ( grp.has_key("Observables") ? grp.open_group("Observables") : 
                                                      grp.create_group("Observables") );
        h5::group g2 = ( g1.has_key("Mixed") ? g1.open_group("Mixed") : 
                                                g1.create_group("Mixed") );
        observ0.print(iblock, std::addressof(g2));
      } else { 
        h5::group *grp = nullptr;
        observ0.print(iblock, grp);
      }
      accumulated_in_last_block = false;
    }
  }

private:
  bool writer = false;
  bool accumulated_in_last_block = false;

  MixedObsHandler<MEM> observ0;

  // Blocking info 
  int iblock       = 0;
  int nblocks_skip = 0;

  int measure_interval = 1;
};
} // namespace afqmc
} // namespace sfqmc

