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

#include "IO/ptree/ptree_utilities.hpp"
#include "IO/app_loggers.h"
#include "utilities/mpi_context.h"

#include "AFQMC/config.h"

#include "AFQMC/HamiltonianOperations/HamiltonianOperations.h"

namespace sfqmc
{
namespace afqmc
{
class KPTHCHamiltonian
{
public:

  KPTHCHamiltonian(ptree pt_in)
      : fileName("")
  {
    // convert user input to verbose input
    ptree pt = interpret_inputs(pt_in);
    app_log(2,"\nKPTHCHamiltonian input:");
    app_log(2, "{}", io::to_string(pt));
    // initialize using verbose input
    fileName  = pt.get<std::string>("filename");
  }

  ~KPTHCHamiltonian() {}

  KPTHCHamiltonian(KPTHCHamiltonian const& other) = default;
  KPTHCHamiltonian(KPTHCHamiltonian&& other)      = default;
  KPTHCHamiltonian& operator=(KPTHCHamiltonian const& other) = default;
  KPTHCHamiltonian& operator=(KPTHCHamiltonian&& other) = default;

  HamiltonianTypes getHamType() const { return KPTHC; }

  template<MEMORY_SPACE MEM>
  HamiltonianOperations<MEM> getHamiltonianOperations(WALKER_TYPES type,
                 std::shared_ptr<utils::mpi_context_t<mpi3::communicator>> mpi, 
                 nda::array<PsiT_Matrix<MEM>,2> const& PsiT);

  static ptree interpret_inputs(const ptree pt0)
  {
    // read inputs with default options
    std::string name, filename;
    name      = pt0.get<std::string>("name", "ham0");
    filename  = pt0.get<std::string>("filename");
    // validate inputs
    // create verbose internal inputs
    ptree pt1;
    pt1.put("name", name);
    pt1.put("filename", filename);
    std::unordered_set<std::string> pass_through_keys = {};
    io::compare_known_keys("Tensor hyper-contraction (KPTHC) KP Hamiltonian", pt1, pt0,pass_through_keys);
    return pt1;
  }

protected:
  std::shared_ptr<utils::mpi_context_t<mpi3::communicator>> mpi;

  std::string fileName;

};

} // namespace afqmc

} // namespace sfqmc

