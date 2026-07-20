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
#include "utilities/Random.hpp"
#include "utilities/check.hpp"
#include "IO/app_loggers.h"

#include "AFQMC/config.h"
#include "AFQMC/Walkers/WalkerSet.hpp"
#include "AFQMC/Walkers/WalkerIO.hpp"

namespace sfqmc
{
namespace afqmc
{
template<MEMORY_SPACE MEM>
class WalkerSetFactory
{
public:
  WalkerSetFactory() {}

  ~WalkerSetFactory() {}

  bool is_constructed(const std::string& ID)
  {
    auto xml = wlkBlocks.find(ID);
    if (xml == wlkBlocks.end())
      utils::check(false," Error in WalkerSetFactory::is_constructed(string&): Missing xml block. ");
    auto wlk = handlers.find(ID);
    if (wlk == handlers.end())
      return false;
    else
      return true;
  }

  // Build (or return the cached) walker set, fully populated from the per-spin
  // initial guess. Dimensions are inferred from the guess.
  auto& getWalkerSet(std::shared_ptr<utils::mpi_context_t<boost::mpi3::communicator>> mpi,
                     const std::string& ID,
                     std::shared_ptr<utils::RandomGenerator_t<HOST_MEMORY>> rng,
                     WALKER_TYPES walker_type,
                     const std::vector<nda::matrix<ComplexType>>& guess,
                     int nWalkers)
  {
    auto xml = wlkBlocks.find(ID);
    if (xml == wlkBlocks.end())
      utils::check(false,"Error: Missing xml Block in WalkerSetFactory::getWalkerSet(string&). ");
    auto wlk = handlers.find(ID);
    if (wlk == handlers.end())
    {
      auto newwlk = handlers.insert(std::make_pair(ID,
          WalkerSet<MEM>(mpi, xml->second, rng, walker_type, guess, nWalkers)));
      if (!newwlk.second)
        utils::check(false," Error: Problems inserting new walker set. ");
      return (newwlk.first)->second;
    }
    else
      return wlk->second;
  }

  // Build (or return the cached) finite-temperature walker set, fully populated
  // from the rank-4 UDV initial guess {3, nspin, rows, naea}. Dimensions are
  // inferred from the guess and finite_temperature is set true by the ctor.
  auto& getWalkerSetFT(std::shared_ptr<utils::mpi_context_t<boost::mpi3::communicator>> mpi,
                       const std::string& ID,
                       std::shared_ptr<utils::RandomGenerator_t<HOST_MEMORY>> rng,
                       WALKER_TYPES walker_type,
                       nda::MemoryArrayOfRank<4> auto const& UDV,
                       int nWalkers)
  {
    auto xml = wlkBlocks.find(ID);
    if (xml == wlkBlocks.end())
      utils::check(false,"Error: Missing xml Block in WalkerSetFactory::getWalkerSetFT(string&). ");
    auto wlk = handlers.find(ID);
    if (wlk == handlers.end())
    {
      auto newwlk = handlers.insert(std::make_pair(ID,
          WalkerSet<MEM>(mpi, xml->second, rng, walker_type, UDV, nWalkers)));
      if (!newwlk.second)
        utils::check(false," Error: Problems inserting new walker set. ");
      return (newwlk.first)->second;
    }
    else
      return wlk->second;
  }

  // Build (or return the cached) walker set from an HDF5 restart file. Walker
  // dimensions and count come from the file; fh5 must be open read-only on all
  // ranks.
  auto& getWalkerSetFromHDF5(std::shared_ptr<utils::mpi_context_t<boost::mpi3::communicator>> mpi,
                             const std::string& ID,
                             std::shared_ptr<utils::RandomGenerator_t<HOST_MEMORY>> rng,
                             WALKER_TYPES walker_type,
                             h5::file& fh5,
                             int nWalkers,
                             bool set_to_target)
  {
    auto xml = wlkBlocks.find(ID);
    if (xml == wlkBlocks.end())
      utils::check(false,"Error: Missing xml Block in WalkerSetFactory::getWalkerSetFromHDF5(string&). ");
    auto wlk = handlers.find(ID);
    if (wlk != handlers.end())
      return wlk->second;

    auto newwlk = handlers.insert(std::make_pair(ID,
        readWalkersFromHDF5<WalkerSet<MEM>>(mpi, xml->second, rng, walker_type,
                                            fh5, nWalkers, set_to_target)));
    if (!newwlk.second)
      utils::check(false," Error: Problems inserting new walker set. ");
    return (newwlk.first)->second;
  }

  ptree get_input(const std::string& ID) const
  {
    auto xml = wlkBlocks.find(ID);
    if (xml == wlkBlocks.end())
    {
      app_log(1,"WlkFac cannot find {}",ID);
      utils::check(false,"Error: failed to find walker_set with above name.");
      return ptree{};
    }
    else
      return xml->second;
  }

  // Resolve the walker_type of a walker-set block without constructing it.
  WALKER_TYPES get_walker_type(const std::string& ID) const
  {
    return WalkerSet<MEM>::parse_walker_type(get_input(ID));
  }

  // this routine allows you to modify the input block associated with ID 
  ptree& get_input(const std::string& ID)
  {
    auto xml = wlkBlocks.find(ID);
    if (xml == wlkBlocks.end())
    { 
      app_log(1,"failed to find {}", ID);
      utils::check(false,"Error: failed to find walker_set with above name.");
    }
    return xml->second;
  }

  // adds a xml block from which a WalkerSet can be built
  void push(const std::string& ID, ptree pt)
  {
    auto xml = wlkBlocks.find(ID);
    if (xml != wlkBlocks.end())
      utils::check(false,"Error: Repeated WalkerSet block in WalkerSetFactory. WalkerSet names must be unique. ");
    wlkBlocks.insert(std::make_pair(ID, pt));
  }

protected:
  std::map<std::string, ptree> wlkBlocks;

  std::map<std::string, WalkerSet<MEM>> handlers;
};
} // namespace afqmc
} // namespace sfqmc

