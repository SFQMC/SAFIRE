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
bool AFQMCFactory<MEM>::parse(const AFQMCParameters& params)
{
  push_blocks(HamFac, params.hamiltonian);
  push_blocks(WfnFac, params.wavefunction);
  push_blocks(WSetFac, params.walker_set);
  push_blocks(PropFac, params.propagator);

  return true;
}

template<MEMORY_SPACE MEM>
bool AFQMCFactory<MEM>::execute(const AFQMCParameters& params)
{
  for(const auto& exec : params.execute)
  {
    // execute driver
    if (!DriverFac.executeDriver(params.driver, std::format("{}.s{:03d}", project_title, m_series), m_series, exec))
    {
      app_error("Error in DriverFactory::executeDriver::run()");
      app_error_flush();
      return false;
    }

    m_series++;
  }

  return true;
}

template bool AFQMCFactory<HOST_MEMORY>::execute(const AFQMCParameters&);
template bool AFQMCFactory<HOST_MEMORY>::parse(const AFQMCParameters&);

#if defined(ENABLE_DEVICE)
template bool AFQMCFactory<DEVICE_MEMORY>::execute(const AFQMCParameters&);
template bool AFQMCFactory<DEVICE_MEMORY>::parse(const AFQMCParameters&);
#endif

} // namespace afqmc

} // namespace sfqmc
