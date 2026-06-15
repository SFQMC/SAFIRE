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


// -*- C++ -*-
/**@file AFQMCFactory.h
 * @brief Top level class for AFQMC. Parses input and performs setup of classes.
 */

#pragma once

#include <string>
#include <vector>
#include <map>
#include <queue>
#include <algorithm>
#include <cstdlib>

#include "config.h"
#include "IO/ptree/ptree_utilities.hpp"
#include "utilities/threading.h"
#include "utilities/mpi_context.h"

#include "AFQMC/Walkers/WalkerSetFactory.hpp"
#include "AFQMC/Hamiltonians/HamiltonianFactory.h"
#include "AFQMC/Wavefunctions/WavefunctionFactory.h"
#include "AFQMC/Propagators/PropagatorFactory.h"
#include "AFQMC/Drivers/DriverFactory.h"


namespace sfqmc
{
namespace afqmc
{
/**
 * @brief Factory class for AFQMC. Parses input, performs setup of classes, and executes the driver.
 *
 * @details The AFQMCFactory class is the top-level class for AFQMC. It parses the input file, performs the setup of classes, and executes the driver. 
 It contains instances of the following factories which are used to construct the objects used during AFQMC calculations:
  * - HamiltonianFactory HamFac
  * - WalkerSetFactory WSetFac
  * - WavefunctionFactory WfnFac
  * - PropagatorFactory PropFac
  * - DriverFactory DriverFac

 *
 * @param type std::string describing the type of Driver to be used. Valid choices are "afqmc", and "csafqmc".
  * @param pt boost::property_tree::ptree The property tree containing input file parameters
 */
template<MEMORY_SPACE MEM>
class AFQMCFactory
{
public:
  ///constructor
  AFQMCFactory(std::string type, 
               std::shared_ptr<utils::mpi_context_t<boost::mpi3::communicator>> _mpi, 
	       const ptree pt, int n_groups = 1)
     : m_series(pt.get<int>("project.series", 0)),
       project_title(pt.get<std::string>("project.id", "afqmc")),
       mpi(_mpi), 
       InfoMap(),
       HamFac(InfoMap),
       WSetFac(InfoMap),
       WfnFac(InfoMap),
       PropFac(InfoMap),
       DriverFac(mpi, InfoMap, WSetFac, PropFac, WfnFac, HamFac) 
  {
    utils::check(n_groups==1, "finish!!!");
    app_log(1, " AFQMCFactory Project settings: ");
    app_log(1, "    -- id             : {} ", project_title);
    app_log(1, "    -- series         : {} ", m_series);
    app_log(1, "    -- n_groups       : {} ", n_groups);
    app_log(1, "    -- MPI tasks/node : {} ", mpi->node_comm.size());
    app_log(1, "    -- MPI nodes      : {} ", mpi->internode_comm.size());
    app_log(1, "    -- MPI tasks      : {} ", mpi->comm.size());
    app_log(1, "    -- Compute Device    : {} ", (MEM==DEVICE_MEMORY?"gpu":"cpu")); 
    const char* tblis_env = std::getenv("TBLIS_NUM_THREADS");
    const char* omp_env = std::getenv("OMP_NUM_THREADS");
    app_log(1, "    -- TBLIS_NUM_THREADS : {} {}", tblis_env ? tblis_env : "(unset)",
            sfqmc::utils::tblis_threads_was_user_set() ? "(user-provided)" : "");
    app_log(1, "    -- OMP_NUM_THREADS   : {} {}\n\n", omp_env ? omp_env : "(unset)",
            sfqmc::utils::omp_threads_was_user_set() ? "(user-provided)" : "");
    // parse input
    utils::check(parse(pt), " Error in AFQMCFactory: Problems parsing the input file. ");

    // execute 
    utils::check(execute(type,pt), "Error in AFQMCFactory: Problems executing the input file. ");
  }

  ///destructor
  ~AFQMCFactory() = default;

private:

  int m_series;
  std::string project_title;

  std::shared_ptr<utils::mpi_context_t<boost::mpi3::communicator>> mpi;

  // container of AFQMCInfo objects
  std::map<std::string, AFQMCInfo> InfoMap;

  // Hamiltonian factory
  HamiltonianFactory HamFac;

  // WalkerHandler factory
  WalkerSetFactory<MEM> WSetFac;

  // Wavefunction factory
  WavefunctionFactory<MEM> WfnFac;

  // Propagator factory
  PropagatorFactory<MEM> PropFac;

  // driver factory
  DriverFactory<MEM> DriverFac;

  //
  //  Traverse input tree and creates all non-executable objects.
  //  Created objects (pointers actually) are stored in maps based on name in xml block.
  //  Executable sections (drivers) are created with objects already existing
  //  in the maps.
  //
  bool parse(const ptree pt);

  //
  //  Traverse input tree and creates executable sections, using objects created during parsing.
  //
  bool execute(std::string type, const ptree pt);
};
} // namespace afqmc
} // namespace sfqmc

