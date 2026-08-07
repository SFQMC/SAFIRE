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


#include <fstream>
#include <cmath>
#include <algorithm>
#include <random>

#include "config.h"
#include "utilities/check.hpp"

#include "AFQMC/config.h"
#include "PropagatorFactory.h"
#include "AFQMC/Wavefunctions/Wavefunction.hpp"

namespace sfqmc
{
namespace afqmc
{
template<MEMORY_SPACE MEM>
Propagator<MEM> PropagatorFactory<MEM>::buildAFQMCPropagator(std::shared_ptr<utils::mpi_context_t<boost::mpi3::communicator>> mpi, const PropagatorParameters& params, Wavefunction<MEM>& wfn, std::shared_ptr<utils::RandomGenerator_t<MEM>> rng)
{
  return Propagator<MEM>(AFQMCBasePropagator<MEM>(params, mpi, wfn, rng));
}

template Propagator<HOST_MEMORY> 
PropagatorFactory<HOST_MEMORY>::buildAFQMCPropagator(std::shared_ptr<utils::mpi_context_t<boost::mpi3::communicator>>,const PropagatorParameters&,Wavefunction<HOST_MEMORY>&,std::shared_ptr<utils::RandomGenerator_t<HOST_MEMORY>>);

#if defined(ENABLE_DEVICE)
template Propagator<DEVICE_MEMORY> 
PropagatorFactory<DEVICE_MEMORY>::buildAFQMCPropagator(std::shared_ptr<utils::mpi_context_t<boost::mpi3::communicator>>,const PropagatorParameters&,Wavefunction<DEVICE_MEMORY>&,std::shared_ptr<utils::RandomGenerator_t<DEVICE_MEMORY>>);
#endif


} // namespace afqmc

} // namespace sfqmc
