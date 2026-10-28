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

#include <fstream>
#include <memory>
#include <variant>

#include "AFQMC/config.h"
#include "AFQMC/parameters.hpp"
#include "utilities/mpi_context.h"

#include "AFQMC/Hamiltonians/ModelHamOpsGenerator.h"
#include "AFQMC/Hamiltonians/THCHamiltonian.h"
#include "AFQMC/Hamiltonians/KPTHCHamiltonian.h"
#include "AFQMC/Hamiltonians/KPFactorizedHamiltonian.h"
#include "AFQMC/Hamiltonians/RealDenseHamiltonian.h"
#include "AFQMC/HamiltonianOperations/HamiltonianOperations.h"

namespace sfqmc
{
namespace afqmc
{

class Hamiltonian
{

public:

  /// Reads the integral file named by `params` and builds the Hamiltonian its format and type
  /// call for. Collective: the type is peeked on the root and broadcast.
  static Hamiltonian from_params(std::shared_ptr<utils::mpi_context_t<mpi3::communicator>> mpi,
                                 const HamiltonianParameters& params);

  template<typename Ham>
  Hamiltonian(Ham&& other) : var(std::forward<Ham>(other)) {}

  template<typename Ham>
  Hamiltonian& operator=(Ham&& other) {
    var = std::forward<Ham>(other);
    return *this;
  }

  template<MEMORY_SPACE MEM, class... Args>
  HamiltonianOperations<MEM> getHamiltonianOperations(Args&&... args)
  {
      return std::visit([&](auto&& a) { 
	return a.template getHamiltonianOperations<MEM>(std::forward<Args>(args)...); }, var);
  }

  HamiltonianType getHamType() const
  {
    return std::visit([&](auto&& a) { return a.getHamType(); }, var);
  }

  private:

    std::variant<THCHamiltonian, 
                 KPTHCHamiltonian, 
                 ModelHamOpsGenerator, 
                 RealDenseHamiltonian, 
                 KPFactorizedHamiltonian 
                > var;

};

} // namespace afqmc

} // namespace sfqmc

