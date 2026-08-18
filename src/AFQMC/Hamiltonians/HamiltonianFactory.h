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
#include <variant>
#include <vector>
#include <map>
#include <fstream>
#include "IO/app_loggers.h"
#include "IO/banner.hpp"

#include "AFQMC/config.h"
#include "AFQMC/parameters.hpp"
#include "AFQMC/Hamiltonians/Hamiltonian.hpp"

namespace sfqmc
{
namespace afqmc
{

class HamiltonianFactory
{
public:
  bool is_constructed(const std::string& ID)
  {
    auto block = hamBlocks.find(ID);
    utils::check(block != hamBlocks.end(),
                 "Error in HamiltonianFactory::is_constructed(string&): Missing input block.");
    auto ham = hamiltonians.find(ID);
    if (ham == hamiltonians.end())
      return false;
    else
      return true;
  }

  // returns a pointer to the base Hamiltonian class associated with a given ID
  Hamiltonian& getHamiltonian(std::shared_ptr<utils::mpi_context_t<mpi3::communicator>> mpi,
                              const std::string& ID)
  {
    auto block = hamBlocks.find(ID);
    utils::check(block != hamBlocks.end(),
                 "Error in HamiltonianFactory::getHamiltonian(string&): Missing input block.");
    auto ham = hamiltonians.find(ID);
    if (ham == hamiltonians.end())
    {
      auto newham = hamiltonians.insert(std::make_pair(ID, buildHamiltonian(mpi, block->second)));
      if (!newham.second)
        APP_ABORT("Error: Problems inserting new hamiltonian in HamiltonianFactory::getHamiltonian.");
      return (newham.first)->second;
    }
    else
      return ham->second;
  }

  // adds an input block from which a Hamiltonian can be built
  void push(const std::string& ID, HamiltonianParameters params)
  {
    auto block = hamBlocks.find(ID);
    utils::check(block == hamBlocks.end(),
                 "Error: Repeated Hamiltonian block in HamiltonianFactory. Hamiltonian names must be unique.");
    hamBlocks.insert(std::make_pair(ID, std::move(params)));
  }

  const HamiltonianParameters& get_input(const std::string& ID) const
  {
    auto block = hamBlocks.find(ID);
    utils::check(block != hamBlocks.end(),"Error: failed to find Hamiltonian with above name.");
    return block->second;
  }

protected:
  // generates a new Hamiltonian and returns the pointer to the base class
  Hamiltonian buildHamiltonian(std::shared_ptr<utils::mpi_context_t<mpi3::communicator>> mpi,
                               const HamiltonianParameters& params)
  {
    app_log(1, section(std::format("Initializing Hamiltonian \"{}\"", params.name)));

    return fromHDF5(mpi, params);
  }

  Hamiltonian fromHDF5(std::shared_ptr<utils::mpi_context_t<mpi3::communicator>> mpi,
                       const HamiltonianParameters& params);

  std::map<std::string, HamiltonianParameters> hamBlocks;

  std::map<std::string, Hamiltonian> hamiltonians;

};

} // namespace afqmc
} // namespace sfqmc

