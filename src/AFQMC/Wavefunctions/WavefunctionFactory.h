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

#include "AFQMC/config.h"
#include "IO/banner.hpp"
#include "AFQMC/parameters.hpp"
#include "AFQMC/Hamiltonians/Hamiltonian.hpp"
#include "AFQMC/Wavefunctions/Wavefunction.hpp"
#include "AFQMC/HamiltonianOperations/HamiltonianOperations.h"

namespace sfqmc
{
namespace afqmc
{

template<MEMORY_SPACE MEM>
class WavefunctionFactory
{
public:
  WavefunctionFactory() 
  {
    // initialize in fromHDF5
  }

  bool is_constructed(const std::string& ID)
  {
    auto block = wfnBlocks.find(ID);
    if (block == wfnBlocks.end())
    {
      app_log(1,"failed to find {}", ID);
      APP_ABORT(" Error in WavefunctionFactory::is_constructed(string&): Missing wfn block. ");
    }
    auto w0 = wavefunctions.find(ID);
    if (w0 == wavefunctions.end())
      return false;
    else
      return true;
  }

  // returns a pointer to the base Wavefunction class associated with a given ID
  auto& getWavefunction(std::shared_ptr<utils::mpi_context_t<boost::mpi3::communicator>> mpi,
                                const std::string& ID,
                                WALKER_TYPES walker_type,
                                bool finiteT,
                                Hamiltonian* h,
                                int targetNW   = 1)
  {
    auto block = wfnBlocks.find(ID);
    if (block == wfnBlocks.end())
    {
      app_log(1,"failed to find {}", ID);
      utils::check(false," Error in WavefunctionFactory::getWavefunction(string&): Missing wfn block. ");
    }
    auto w0 = wavefunctions.find(ID);
    if (w0 == wavefunctions.end())
    {
      auto neww = wavefunctions.insert(
          std::make_pair(ID, buildWavefunction(mpi,block->second, walker_type, finiteT, h, targetNW)));
      utils::check(neww.second," Error: Problems building new wavefunction in WavefunctionFactory::getWavefunction(string&). ");
      return (neww.first)->second;
    }
    else
      return w0->second;
  }

  // Use this routine to check if there is a wfn associated with a given ID
  const WavefunctionParameters& get_input(const std::string& ID) const
  {
    auto block = wfnBlocks.find(ID);
    if (block == wfnBlocks.end())
    {
      app_log(1,"failed to find {}", ID);
      utils::check(false,"Error: failed to find Wavefunction with above name.");
    }
    return block->second;
  }

  // returns the per-spin initial-guess matrices associated with ID
  const std::vector<nda::matrix<ComplexType>>& getInitialGuess(const std::string& ID)
  {
    auto mat = initial_guess.find(ID);
    if (mat == initial_guess.end())
    {
      APP_ABORT(" Error: Missing initial guess in WavefunctionFactory. ");
    }
    return mat->second;
  }

  // returns the per-spin initial-guess matrices associated with ID
  const std::vector<nda::matrix<ComplexType>>& getInitialGuess(const std::string& ID) const
  {
    auto mat = initial_guess.find(ID);
    if (mat == initial_guess.end())
    {
      APP_ABORT(" Error: Missing initial guess in WavefunctionFactory. ");
    }
    return mat->second;
  }

    // returns the finite-temperature initial guess associated with ID
  auto getInitialGuess_ft(const std::string& ID) 
  {
    auto mat = initial_guess_ft.find(ID);
    if (mat == initial_guess_ft.end())
    {
      APP_ABORT(" Error: Missing initial guess in WavefunctionFactory. ");
    }
    // return view
    return mat->second();
  }

  // returns the finite-temperature initial guess associated with ID
  auto getInitialGuess_ft(const std::string& ID) const
  {
    auto mat = initial_guess_ft.find(ID);
    if (mat == initial_guess_ft.end())
    {
      APP_ABORT(" Error: Missing initial guess in WavefunctionFactory. ");
    }
    // return view
    return mat->second();
  }

  // adds an input block from which a Wavefunction can be built
  void push(const std::string& ID, WavefunctionParameters params)
  {
    auto block = wfnBlocks.find(ID);
    if (block != wfnBlocks.end())
      APP_ABORT("Error: Repeated Wavefunction block in WavefunctionFactory. Wavefunction names must be unique. ");
    wfnBlocks.insert(std::make_pair(ID, std::move(params)));
  }

protected:
  // generates a new Wavefunction and returns the pointer to the base class
  Wavefunction<MEM> buildWavefunction(std::shared_ptr<utils::mpi_context_t<boost::mpi3::communicator>> mpi,
                                 const WavefunctionParameters& params,
                                 WALKER_TYPES walker_type,
                                 bool finiteT,
                                 Hamiltonian* h,
                                 int targetNW)
  {
    app_log(1, section(std::format("Initializing Wavefunction \"{}\"", params.name)));

    return fromHDF5(mpi, params, walker_type, finiteT, *h, targetNW);
  }

  Wavefunction<MEM> fromHDF5(std::shared_ptr<utils::mpi_context_t<boost::mpi3::communicator>> mpi,
                        const WavefunctionParameters& params,
                        WALKER_TYPES walker_type,
                        bool finiteT,
                        Hamiltonian& h,
                        int targetNW);

  void getInitialGuess(h5::group grp, const std::string& name, int NMO, int nup, int ndown, WALKER_TYPES walker_type);
  void getInitialGuess_ft(h5::group grp, utils::mpi_context_t<boost::mpi3::communicator>& mpi, const std::string& name, int NMO, WALKER_TYPES walker_type, bool finiteT);
/*
  int getExcitation(nda::MemoryVector& deti,
                    nda::MemoryVector& detj,
                    std::vector<int>& excit,
                    int& perm);
  void computeVariationalEnergyPHMSD(Hamiltonian& ham,
                                     nda::MemoryMatrix& occs,
                                     std::vector<ComplexType>& coeff,
                                     int ndets,
                                     int nup,
                                     int ndown,
                                     int NMO,
                                     bool recomputeCI);
  ComplexType slaterCondon0(Hamiltonian& ham, nda::MemoryVector auto& det, int NMO);
  ComplexType slaterCondon1(Hamiltonian& ham, std::vector<int>& excit, nda::MemoryVector auto& det, int NMO);
  ComplexType slaterCondon2(Hamiltonian& ham, std::vector<int>& excit, int NMO);
*/

  void build_PsiT_MO_phmsd(WALKER_TYPES walker_type, int npol, int NMO, int nup, 
	int ndown, int ndets, nda::array<ComplexType,1>& coeffs, 
        nda::array<int,2>& occs, nda::array<PsiT_Matrix<HOST_MEMORY>,1>& PsiT_MO);

  std::map<std::string, WavefunctionParameters> wfnBlocks;

  std::map<std::string, Wavefunction<MEM>> wavefunctions;

  // per-spin trial orbital matrices, sized to the walker (alpha: npol*NMO x naea,
  // beta: NMO x naeb for collinear). The true widths keep {rows, naea, naeb}
  // inferable by the walker set directly from the guess.
  std::map<std::string, std::vector<nda::matrix<ComplexType>>> initial_guess;

  std::map<std::string, memory::const_shared_array<HOST_MEMORY, ComplexType, 4>> initial_guess_ft;
};
} // namespace afqmc
} // namespace sfqmc

