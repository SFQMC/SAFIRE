/*
 * This file is distributed under the Apache License, Version 2.0 License.
 * See LICENSE file in top directory for details.
 *
 * Copyright (c) 2021-2025 The Simons Foundation, Inc.
 *
 * You may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *      http://www.apache.org/licenses/LICENSE-2.0
 *
 */

#pragma once

#include <vector>

#include "IO/ptree/ptree_utilities.hpp"
#include "IO/app_loggers.h"
#include "utilities/mpi_context.h"
    
#include "AFQMC/config.h"

#include "AFQMC/HamiltonianOperations/HamiltonianOperations.h"

namespace sfqmc
{
namespace afqmc
{
class ModelHamOpsGenerator
{
public:
  ModelHamOpsGenerator(ptree pt_in)
  {
    // convert user input to verbose input
    ptree pt = interpret_inputs(pt_in);
    app_log(2,"\nModelHamiltonian input:");
    app_log(2, "{}", io::to_string(pt));
    // initialize using verbose input
    fileName  = pt.get<std::string>("filename");
    shift_1body = pt.get<bool>("shift_1body");
  }

  ~ModelHamOpsGenerator() {}

  ModelHamOpsGenerator(ModelHamOpsGenerator const& other) = default;
  ModelHamOpsGenerator(ModelHamOpsGenerator&& other)      = default;
  ModelHamOpsGenerator& operator=(ModelHamOpsGenerator const& other) = default;
  ModelHamOpsGenerator& operator=(ModelHamOpsGenerator&& other) = default;

  HamiltonianTypes getHamType() const { return ModelHamiltonian; }

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
    bool shift_1b = pt0.get<bool>("shift_1body", false);
    // create verbose internal inputs
    ptree pt1;
    pt1.put("name", name);
    pt1.put("filename", filename);
    pt1.put("shift_1body", shift_1b);
    std::unordered_set<std::string> pass_through_keys = {};
    io::compare_known_keys("Lattice Model",pt1, pt0,pass_through_keys);
    return pt1;
  }

protected:

  std::string fileName = "";
  bool shift_1body = false;

  template<MEMORY_SPACE MEM, bool REAL, typename ValueType>
  SparseEnergy<MEM,REAL> make_SparseEnergy(
      std::shared_ptr<utils::mpi_context_t<mpi3::communicator>> mpi,
      WALKER_TYPES type, math::sparse::csr_matrix<ValueType, HOST_MEMORY, int, int>& hij,
      math::sparse::csr_matrix<ValueType, HOST_MEMORY, int, int>& U,
      math::sparse::csr_matrix<ValueType, HOST_MEMORY, int, int>& J, ComplexType E0);

  template<typename ValueType>
  nda::array<long,1> find_occupied_pairs(WALKER_TYPES type, 
      std::vector<math::sparse::csr_matrix<ValueType, HOST_MEMORY, int, int>>& U, 
      std::vector<math::sparse::csr_matrix<ValueType, HOST_MEMORY, int, int>>& J);

  template<MEMORY_SPACE MEM, bool REAL, typename ValueType, class map_t>
  void addComponent( WALKER_TYPES wtype, PropagatorTypes ptype,
       std::shared_ptr<utils::mpi_context_t<mpi3::communicator>> mpi,
       math::sparse::csr_matrix<ValueType, HOST_MEMORY, int, int>& U,
       math::sparse::csr_matrix<ValueType, HOST_MEMORY, int, int>& J,
       std::vector<ModelComponent<MEM,REAL>>& Hams,
       nda::array<long,1>& n2IJ, map_t& IJ2n);

  template<MEMORY_SPACE MEM, bool REAL>
  HamiltonianOperations<MEM> getHamiltonianOperations_impl(WALKER_TYPES type,
                 std::shared_ptr<utils::mpi_context_t<mpi3::communicator>> mpi,
                 nda::array<PsiT_Matrix<MEM>,2> const& PsiT);

};

} // namespace afqmc
} // namespace sfqmc

#include "AFQMC/Hamiltonians/ModelHamOpsGenerator.icc"

