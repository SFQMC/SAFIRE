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
#include "IO/ptree/ptree_utilities.hpp"
#include "IO/app_loggers.h"

#include "AFQMC/config.h"
#include "AFQMC/Hamiltonians/Hamiltonian.hpp"

namespace sfqmc
{
namespace afqmc
{

class HamiltonianFactory
{
public:
  HamiltonianFactory(std::map<std::string, AFQMCInfo>& info) : 
	InfoMap(info)
  {}

  ~HamiltonianFactory()
  {
    // delete Hamiltonian objects
    //for (auto it = hamiltonians.begin(); it != hamiltonians.end(); ++it)
    //  delete it->second;
  }

  bool is_constructed(const std::string& ID)
  {
    auto xml = hamBlocks.find(ID);
    utils::check(xml != hamBlocks.end(),
                 " Error in WavefunctionFactory::is_constructed(string&): Missing xml block. ");
    auto ham = hamiltonians.find(ID);
    if (ham == hamiltonians.end())
      return false;
    else
      return true;
  }

  // returns true if an xml block has been registered for ID (whether or not the Hamiltonian object
  // has been built yet). Unlike is_constructed, does not assert the block exists -- safe to call as a
  // "should I push this block?" guard.
  bool has_input(const std::string& ID) const
  {
    return hamBlocks.find(ID) != hamBlocks.end();
  }

  // returns a pointer to the base Hamiltonian class associated with a given ID
  Hamiltonian& getHamiltonian(std::shared_ptr<utils::mpi_context_t<mpi3::communicator>> mpi, 
                              const std::string& ID)
  {
    auto xml = hamBlocks.find(ID);
    utils::check(xml != hamBlocks.end(),
                 " Error in WavefunctionFactory::is_constructed(string&): Missing xml block. "); 
    auto ham = hamiltonians.find(ID);
    if (ham == hamiltonians.end())
    {
      auto newham = hamiltonians.insert(std::make_pair(ID, buildHamiltonian(mpi, xml->second)));
      if (!newham.second)
        APP_ABORT(" Error: Problems inserting new hamiltonian in HamiltonianFactory::getHamiltonian. ");
      return (newham.first)->second;
    }
    else
      return ham->second;
  }

  // adds a xml block from which a Hamiltonian can be built
  void push(const std::string& ID, ptree pt)
  {
    auto xml = hamBlocks.find(ID);
    utils::check(xml == hamBlocks.end(),
                 "Error: Repeated Hamiltonian block in HamiltonianFactory. Hamiltonian names must be unique. ");
    hamBlocks.insert(std::make_pair(ID, pt));
  }

  ptree get_input(const std::string& ID) const
  {
    auto xml = hamBlocks.find(ID);
    utils::check(xml != hamBlocks.end(),"Error: failed to find Hamiltonian with above name.");
    return xml->second;
  }

  // this routine allows you to modify the input block associated with ID 
  ptree& get_input(const std::string& ID)
  {
    auto xml = hamBlocks.find(ID);
    utils::check(xml != hamBlocks.end(),"Error: failed to find Hamiltonian with above name.");
    return xml->second;
  }

protected:
  // reference to container of AFQMCInfo objects
  std::map<std::string, AFQMCInfo>& InfoMap;

  // generates a new Hamiltonian and returns the pointer to the base class
  Hamiltonian buildHamiltonian(std::shared_ptr<utils::mpi_context_t<mpi3::communicator>> mpi, 
                               ptree pt)
  {
    std::string fham_type;
    fham_type = pt.get<std::string>("filetype", "hdf5");

    app_log(1,"\n****************************************************");
    app_log(1,"               Initializing Hamiltonian ");
    app_log(1,"\n****************************************************\n");
    app_log(2, " Hamiltonian Factory input: ");
    app_log(2, "{}", io::to_string(pt));

    if (fham_type == "hdf5")
      return fromHDF5(mpi, pt);
    else
    {
      utils::check(false," Error: Unknown Hamiltonian filetype in HamiltonianFactory::buildHamiltonian(). ");
    }
    return Hamiltonian{};
  }

  Hamiltonian fromHDF5(std::shared_ptr<utils::mpi_context_t<mpi3::communicator>> mpi, ptree pt);

  std::map<std::string, ptree> hamBlocks;

  std::map<std::string, Hamiltonian> hamiltonians;

};

} // namespace afqmc
} // namespace sfqmc

