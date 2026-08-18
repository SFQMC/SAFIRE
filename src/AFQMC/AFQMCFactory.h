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
#include "AFQMC/parameters.hpp"
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

/// Registers the input blocks of one component with its factory. resolve_defaults has hoisted
/// every block declared inside an execute block into these lists, so this registers all of them
/// and the execute blocks only ever refer to them by name.
template<class Factory, class Params>
void push_blocks(Factory& fac, const std::vector<Params>& blocks)
{
  for(const auto& params : blocks) {
    fac.push(params.name, params);
  }
}

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
 * @param params AFQMCParameters The deserialized contents of the simulation block of the input file
 */
template<MEMORY_SPACE MEM>
class AFQMCFactory
{
public:
  ///constructor
  AFQMCFactory(const AFQMCParameters& params,
               std::shared_ptr<utils::mpi_context_t<boost::mpi3::communicator>> _mpi)
     : m_series(params.project.series),
       project_title(params.project.id),
       mpi(_mpi),
       HamFac(),
       WSetFac(),
       WfnFac{},
       PropFac(),
       DriverFac(mpi, WSetFac, PropFac, WfnFac, HamFac)
  {
    utils::check(params.project.n_groups==1, "finish!!!");

    // parse input
    utils::check(parse(params), "Error in AFQMCFactory: Problems parsing the input file.");

    // execute
    utils::check(execute(params), "Error in AFQMCFactory: Problems executing the input file.");
  }

private:

  int m_series;
  std::string project_title;

  std::shared_ptr<utils::mpi_context_t<boost::mpi3::communicator>> mpi;

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
  bool parse(const AFQMCParameters& params);

  //
  //  Traverse input tree and creates executable sections, using objects created during parsing.
  //
  bool execute(const AFQMCParameters& params);
};
} // namespace afqmc
} // namespace sfqmc

