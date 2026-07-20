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

#include "IO/ptree/ptree_utilities.hpp"
#include "IO/app_loggers.h"
#include "utilities/mpi_context.h"

#include "AFQMC/config.h"

#include "AFQMC/HamiltonianOperations/HamiltonianOperations.h"

namespace sfqmc
{
namespace afqmc
{
class KPFactorizedHamiltonian
{
public:

  KPFactorizedHamiltonian(ptree pt_in)
      : fileName("")
  {
    // convert user input to verbose input
    ptree pt = interpret_inputs(pt_in);
    app_log(2,"\nKPFactorizedHamiltonian input:");
    app_log(2, "{}", io::to_string(pt));
    // initialize using verbose input
    fileName  = pt.get<std::string>("filename");
    buffer_size = pt.get<int>("buffer_size");
  }

  ~KPFactorizedHamiltonian() {}

  KPFactorizedHamiltonian(KPFactorizedHamiltonian const& other) = default;
  KPFactorizedHamiltonian(KPFactorizedHamiltonian&& other)      = default;
  KPFactorizedHamiltonian& operator=(KPFactorizedHamiltonian const& other) = default;
  KPFactorizedHamiltonian& operator=(KPFactorizedHamiltonian&& other) = default;

  HamiltonianTypes getHamType() const { return KPFactorized; }

  template<MEMORY_SPACE MEM>
  HamiltonianOperations<MEM> getHamiltonianOperations(WALKER_TYPES type,
                 std::shared_ptr<utils::mpi_context_t<mpi3::communicator>> mpi,
                 nda::array<PsiT_Matrix<MEM>,2> const& PsiT);

  static ptree interpret_inputs(const ptree pt0)
  {
    // read inputs with default options
    std::string name, filename;
    int bsize;
    filename  = pt0.get<std::string>("filename");
    name      = pt0.get<std::string>("name", "ham0");
    bsize = pt0.get<int>("buffer_size", 4096);
    // create verbose internal inputs
    ptree pt1;
    pt1.put("name", name);
    pt1.put("filename", filename);
    pt1.put("buffer_size",bsize);
    std::unordered_set<std::string> pass_through_keys = {};
    io::compare_known_keys("K-point Factorized Cholesky Hamiltonian",pt1, pt0,pass_through_keys);
    return pt1;
  }

protected:
  std::shared_ptr<utils::mpi_context_t<mpi3::communicator>> mpi;

  std::string fileName;

  int buffer_size = 4096; 

};

} // namespace afqmc
} // namespace sfqmc

