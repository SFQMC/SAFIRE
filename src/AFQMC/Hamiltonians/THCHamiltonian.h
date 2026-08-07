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

#include <iostream>
#include <vector>
#include <map>
#include <fstream>

#include "IO/app_loggers.h"
#include "utilities/mpi_context.h"

#include "AFQMC/config.h"
#include "AFQMC/parameters.hpp"
#include "nda/h5.hpp"

#include "AFQMC/HamiltonianOperations/HamiltonianOperations.h"

namespace sfqmc
{
namespace afqmc
{
class THCHamiltonian
{
public:

  THCHamiltonian() = delete;

  THCHamiltonian(const HamiltonianParameters& params) : fileName(params.filename) {}

  HamiltonianTypes getHamType() const { return THC; }

  template<MEMORY_SPACE MEM>
  HamiltonianOperations<MEM> getHamiltonianOperations(WALKER_TYPES type,
                 std::shared_ptr<utils::mpi_context_t<mpi3::communicator>> mpi, 
                 nda::array<PsiT_Matrix<MEM>,2> const& PsiT);

protected:
  std::string fileName;

  template<MEMORY_SPACE MEM, bool REAL> 
  HamiltonianOperations<MEM> getHamiltonianOperations_impl(WALKER_TYPES type,
                 std::shared_ptr<utils::mpi_context_t<mpi3::communicator>> mpi, 
                 nda::array<PsiT_Matrix<MEM>,2> const& PsiT);

};

} // namespace afqmc

} // namespace sfqmc

