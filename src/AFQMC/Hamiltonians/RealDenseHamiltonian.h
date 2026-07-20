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
#include "nda/h5.hpp"

#include "AFQMC/HamiltonianOperations/HamiltonianOperations.h"

namespace sfqmc
{
namespace afqmc
{
class RealDenseHamiltonian
{
public:
  RealDenseHamiltonian(ptree pt_in)
      : fileName(""),
        max_memory_MB(2000)
  {
    // convert user input to verbose input
    ptree pt = interpret_inputs(pt_in);
    app_log(2,"\nReal Cholesky input:");
    app_log(2, "{}", io::to_string(pt));
    // initialize using verbose input
    fileName  = pt.get<std::string>("filename");
    max_memory_MB = pt.get<int>("max_memory");
  }

  ~RealDenseHamiltonian() = default;

  RealDenseHamiltonian(RealDenseHamiltonian const& other) = default;
  RealDenseHamiltonian(RealDenseHamiltonian&& other)      = default;
  RealDenseHamiltonian& operator=(RealDenseHamiltonian const& other) = default; 
  RealDenseHamiltonian& operator=(RealDenseHamiltonian&& other) = default; 

  HamiltonianTypes getHamType() const {return RealDenseFactorized; }

  template<MEMORY_SPACE MEM>
  HamiltonianOperations<MEM> getHamiltonianOperations(WALKER_TYPES type,
                 std::shared_ptr<utils::mpi_context_t<mpi3::communicator>> mpi,
                 nda::array<PsiT_Matrix<MEM>,2> const& PsiT);

  static ptree interpret_inputs(const ptree pt0)
  {
    // read inputs with default options
    std::string name, filename;
    int mmem_mb;
    name      = pt0.get<std::string>("name", "ham0");
    filename  = pt0.get<std::string>("filename");
    mmem_mb = pt0.get<int>("max_memory", 2000);
    // create verbose internal inputs
    ptree pt1;
    pt1.put("name", name);
    pt1.put("filename", filename);
    pt1.put("max_memory", mmem_mb);
    std::unordered_set<std::string> pass_through_keys = {};
    io::compare_known_keys("Dense, Real Factorized (Cholesky) Hamiltonian",pt1, pt0,pass_through_keys);
    return pt1;
  }

protected:
  std::string fileName;

  int max_memory_MB;
};

} // namespace afqmc
} // namespace sfqmc

