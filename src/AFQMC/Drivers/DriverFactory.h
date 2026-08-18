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


#pragma once

#include "utilities/mpi_context.h"

#include "AFQMC/parameters.hpp"

#include "AFQMC/Walkers/WalkerSetFactory.hpp"
#include "AFQMC/Hamiltonians/HamiltonianFactory.h"
#include "AFQMC/Wavefunctions/WavefunctionFactory.h"
#include "AFQMC/Propagators/PropagatorFactory.h"

namespace sfqmc
{
namespace afqmc
{
template<MEMORY_SPACE MEM>
class DriverFactory
{
  using communicator = boost::mpi3::communicator;

public:
  DriverFactory(std::shared_ptr<utils::mpi_context_t<boost::mpi3::communicator>> _mpi,
                WalkerSetFactory<MEM>& wsetfac_,
                PropagatorFactory<MEM>& pfac_,
                WavefunctionFactory<MEM>& wfnfac_,
                HamiltonianFactory& hfac)
      : mpi(_mpi),
        WSetFac(wsetfac_),
        PropFac(pfac_),
        HamFac(hfac),
        WfnFac(wfnfac_)
  { }

  bool executeDriver(DriverType type, std::string title, int m_series, const ExecuteParameters& exec);

private:
  bool executeAFQMCDriver(std::string title, int m_series, const ExecuteParameters& exec);
  bool executeFTAFQMCDriver(std::string title, int m_series, const ExecuteParameters& exec);
  bool executeCSAFQMCDriver(std::string title, int m_series, const ExecuteParameters& exec);

  std::shared_ptr<utils::mpi_context_t<boost::mpi3::communicator>> mpi;

  // WalkerHandler factory
  WalkerSetFactory<MEM>& WSetFac;

  // Propagator factory
  PropagatorFactory<MEM>& PropFac;

  // Hamiltonian factory
  HamiltonianFactory& HamFac;

  // Wavefunction factory
  WavefunctionFactory<MEM>& WfnFac;

  // The names of the hamiltonian, wavefunction, walker set and propagator of an execute block.
  // resolve_defaults has registered every block under a name, so the execute block only holds
  // references.
  std::tuple<std::string,std::string,std::string,std::string>
    get_component_ids(const ExecuteParameters& exec);

  // The wavefunction named wfn_name. The hamiltonian ham_name is only built if the wavefunction
  // does not exist yet, e.g. from a previous execute block.
  Wavefunction<MEM>& get_wavefunction(const std::string& wfn_name, const std::string& ham_name,
                                      WALKER_TYPES walker_type, bool finiteT, int nWalkers);

};

} // namespace afqmc

} // namespace sfqmc

