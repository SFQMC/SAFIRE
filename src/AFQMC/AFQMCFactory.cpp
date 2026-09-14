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

#include <filesystem>
#include <format>
#include <string>
#include "config.h"

#include "IO/app_loggers.h"
#include "AFQMC/AFQMCFactory.h"
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
bool AFQMCFactory<MEM>::execute(const AFQMCParameters& params) {
  // every execute block appends its own stage to one results file, so a stale file from a
  // previous run has to go before the first stage writes
  if(mpi_->comm.root()) {
    std::filesystem::remove(std::format("{}.results.h5", output_name_));
  }
  mpi_->comm.barrier();

  for(const auto& exec : params.execute) {
    if(!DriverFac.executeDriver(params.driver, output_name_, stage_index_, exec)) {
      app_error("Error in DriverFactory::executeDriver::run()");
      app_error_flush();
      return false;
    }

    stage_index_++;
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
