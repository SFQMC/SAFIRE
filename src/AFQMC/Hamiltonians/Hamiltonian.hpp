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
#include <variant>

#include "AFQMC/config.h"

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
  Hamiltonian() = default; 
  explicit Hamiltonian(THCHamiltonian&& other) : var(std::move(other)) {}
  explicit Hamiltonian(KPTHCHamiltonian&& other) : var(std::move(other)) {}
  explicit Hamiltonian(ModelHamOpsGenerator&& other) : var(std::move(other)) {}
  explicit Hamiltonian(KPFactorizedHamiltonian&& other) : var(std::move(other)) {}
  explicit Hamiltonian(RealDenseHamiltonian&& other) : var(std::move(other)) {}

  explicit Hamiltonian(THCHamiltonian const& other) : var(other) {}  
  explicit Hamiltonian(KPTHCHamiltonian const& other) : var(other) {}  
  explicit Hamiltonian(ModelHamOpsGenerator const& other) : var(other) {}
  explicit Hamiltonian(KPFactorizedHamiltonian const& other) : var(other) {}
  explicit Hamiltonian(RealDenseHamiltonian const& other) : var(other) {}

  Hamiltonian(Hamiltonian const& other) = default;
  Hamiltonian(Hamiltonian&& other)      = default;

  Hamiltonian& operator=(Hamiltonian const& other) = default;
  Hamiltonian& operator=(Hamiltonian&& other) = default;

  template<MEMORY_SPACE MEM, class... Args>
  HamiltonianOperations<MEM> getHamiltonianOperations(Args&&... args)
  {
      return std::visit([&](auto&& a) { 
	return a.template getHamiltonianOperations<MEM>(std::forward<Args>(args)...); }, var);
  }

  HamiltonianTypes getHamType() const
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

