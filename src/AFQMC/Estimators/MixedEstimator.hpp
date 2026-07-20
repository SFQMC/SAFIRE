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

#include "IO/ptree/ptree_utilities.hpp"
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
  MixedEstimator(std::shared_ptr<utils::mpi_context_t<boost::mpi3::communicator>> mpi,
                          std::string name,
                          ptree pt,
                          WALKER_TYPES wlk,
                          Wavefunction<MEM>& wfn)
      : observ0(mpi, name, pt, wlk, wfn.getNMO(), wfn)
  {
    int _pop_control_interval, equil_multiplier;
    _pop_control_interval = pt.get<int>("_population_control_interval", DEFAULT_POPULATION_CONTROL_INTERVAL);
    equil_multiplier = pt.get<int>("equil_multiplier", 0); // units of population control interval
    measure_interval_multiplier = pt.get<int>("measure_interval_multiplier", DEFAULT_MEASURE_INTERVAL_MULTIPLIER); // units of population control interval
    measure_interval = measure_interval_multiplier * _pop_control_interval;
    utils::check(equil_multiplier % measure_interval_multiplier==0,"Error in MixedEstimator user input: 'equil_multiplier' must be evenly divisible by 'measure_interval_multiplier'");
    nblocks_skip = equil_multiplier / measure_interval_multiplier;
    writer = (mpi->comm.rank() == 0);
  }

  ~MixedEstimator() {}

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

    AFQMCTimer.start(mixed_estimator_timer);
    observ0.accumulate(wset);
    iblock++;
    accumulated_in_last_block = true;
    AFQMCTimer.stop(mixed_estimator_timer);
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
  int measure_interval_multiplier = 1;
};
} // namespace afqmc
} // namespace sfqmc

