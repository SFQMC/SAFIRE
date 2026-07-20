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
/**@file AFQMCFactory.cpp
 * @brief Top level class for AFQMC. Parses input and performs setup of classes.
 */

#include <iostream>
#include <fstream>
#include <string>
#include <vector>
#include <map>
#include <complex>
#include <tuple>
#include <queue>
#include <algorithm>
#include "config.h"
#include "utilities/mpi_context.h"
#include "utilities/check.hpp"

#include "IO/app_loggers.h"
#include "AFQMC/AFQMCFactory.h"
#include "AFQMC/Walkers/WalkerSetFactory.hpp"
#include "AFQMC/Hamiltonians/HamiltonianFactory.h"
#include "AFQMC/Propagators/PropagatorFactory.h"
#include "AFQMC/Wavefunctions/WavefunctionFactory.h"
#include "AFQMC/Drivers/DriverFactory.h"

namespace sfqmc
{

namespace afqmc
{

template<MEMORY_SPACE MEM>
bool AFQMCFactory<MEM>::parse(const ptree pt_in)
{
  // look for non-executable blocks
  for(auto& it : pt_in)
  {
    std::string cname = it.first;
    ptree pt = it.second;
    std::string oname = pt.get<std::string>("name", "");

    if (cname == "hamiltonian")
    {
      if (oname == "") APP_ABORT("hamiltonian outside execute block must be named");
      HamFac.push(oname, pt);
    }
    else if (cname == "wavefunction")
    {
      if (oname == "") APP_ABORT("wavefunction outside execute block must be named");
      WfnFac.push(oname, pt);
    }
    else if (cname == "walker_set")
    {
      if (oname == "") APP_ABORT("walker_set outside execute block must be named");
      WSetFac.push(oname, pt);
    }
    else if (cname == "propagator")
    {
      if (oname == "") APP_ABORT("propagator outside execute block must be named");
      PropFac.push(oname, pt);
    }
    else if(cname != "project" and cname != "execute") {
      app_warning("Ignoring unknown input block: {}", cname);
    }
  }

  return true;
}

template<MEMORY_SPACE MEM>
bool AFQMCFactory<MEM>::execute(std::string type, const ptree pt_in)
{
  for(auto& [cname, pt] : pt_in)
  {
    if (cname == "execute")
    {
      // execute driver
      if (!DriverFac.executeDriver(type, std::format("{}.s{:03d}", project_title, m_series), m_series, pt))
      {
        app_error("Error in DriverFactory::executeDriver::run()");
        app_error_flush();
        return false;
      }

      m_series++;
    }
  }

  return true;
}

template bool AFQMCFactory<HOST_MEMORY>::execute(std::string,const ptree);
template bool AFQMCFactory<HOST_MEMORY>::parse(const ptree);

#if defined(ENABLE_DEVICE)
template bool AFQMCFactory<DEVICE_MEMORY>::execute(std::string,const ptree);
template bool AFQMCFactory<DEVICE_MEMORY>::parse(const ptree);
#endif

} // namespace afqmc

} // namespace sfqmc
