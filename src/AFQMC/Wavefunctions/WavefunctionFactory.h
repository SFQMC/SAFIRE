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
#include <unordered_set>
#include <boost/optional.hpp>

#include "AFQMC/config.h"
#include "AFQMC/Hamiltonians/Hamiltonian.hpp"
#include "AFQMC/Hamiltonians/HamiltonianFactory.h"
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
  // Original constructor (no inner-Hamiltonian support): kept so every existing call site stays valid.
  // A stochastic trial that names `inner_hamiltonian` under a factory built this way aborts in fromHDF5
  // with a clear message.
  WavefunctionFactory(std::map<std::string, AFQMCInfo>& info) : InfoMap(info)
  {
    // initialize in fromHDF5
  }

  // Overload that additionally takes the HamiltonianFactory used to build the StochasticWfn inner
  // (Variational) Hamiltonian `Ĥ_var` on demand when a stochastic trial names one via
  // `inner_hamiltonian` (Phase 3b-var). WavefunctionFactory remains a Hamiltonian *consumer* — it does
  // not own Hamiltonians, it asks hamfac to build the second one. Non-stochastic and clone-path trials
  // never touch it.
  WavefunctionFactory(std::map<std::string, AFQMCInfo>& info, HamiltonianFactory& hamfac)
      : InfoMap(info), HamFac_(&hamfac)
  {
    // initialize in fromHDF5
  }

  static ptree interpret_inputs(const ptree pt0)
  {
    // check required fields exist
    if(not io::check_exists<std::string>(pt0,"name"))
      APP_ABORT("Error in WavefunctionFactory: missing required input: name \n");
    if(not io::check_exists<std::string>(pt0,"filename"))
      APP_ABORT("Error in WavefunctionFactory: missing required input: filename \n");
    if(not io::check_exists<std::string>(pt0,"system"))
      APP_ABORT("Error in WavefunctionFactory: missing required input: info \n");
    // read inputs with default options
    int ndets_to_read = pt0.get<int>("ndets_to_read", -1);
    std::string name          = pt0.get<std::string>("name");
    std::string info          = pt0.get<std::string>("system");
    std::string filename      = pt0.get<std::string>("filename");
//    std::string restart_file  = pt0.get<std::string>("restart_file", "");
    bool rediag        = pt0.get<bool>("rediag", false);
    // validate inputs
    // create verbose internal inputs
    ptree pt1;
    pt1.put("name", name);
    pt1.put("system", info);
    pt1.put("filename", filename);
//    pt1.put("restart_file", restart_file);
    pt1.put("rediag", rediag);
    pt1.put("ndets_to_read", ndets_to_read);
    // optional parameters 
    if( auto val = pt0.get_optional<int>("algorithm") )
      pt1.put("algorithm", *val);
    // set default later, since it depends on HamiltonianOperations type
    if( auto val = pt0.get_optional<bool>("dense_trial") )
      pt1.put("dense_trial", *val);
    bool stochastic    = pt0.get<bool>("stochastic", false);
    int inner_nwalkers = pt0.get<int>("inner_nwalkers", 1);
    if (inner_nwalkers < 1)
      APP_ABORT("Error in WavefunctionFactory::interpret_inputs: inner_nwalkers must be >= 1.");
    int inner_nsteps = pt0.get<int>("inner_nsteps", 0);
    bool inner_conditioning = pt0.get<bool>("inner_conditioning", false);
    int inner_seed   = pt0.get<int>("inner_seed", 777);
    auto inner_propagator_block = pt0.get_child_optional("inner_propagator");
    // inner_hamiltonian: optional block naming the second (Variational) Hamiltonian HDF5 file for the
    // stochastic inner stack. It is a factory-level key consumed by fromHDF5 (which builds the Ham via
    // HamFac_); it is deliberately NOT forwarded into pt1, so it never reaches the wavefunction's own
    // ptree (StochasticWfn::interpret_inputs does not know it). interpret_inputs only (a) rejects it
    // when stochastic is off and (b) lists it as a known pass-through key for compare_known_keys.
    for (auto const& key :
         {"inner_nwalkers", "inner_nsteps", "inner_conditioning", "inner_seed", "inner_propagator",
          "inner_hamiltonian"})
      if (not stochastic && pt0.get_child_optional(key))
        APP_ABORT("Error in WavefunctionFactory::interpret_inputs: " + std::string(key) +
                  " requires stochastic: true.");
    pt1.put("stochastic", stochastic);
    if (stochastic)
    {
      pt1.put("inner_nwalkers", inner_nwalkers);
      pt1.put("inner_nsteps", inner_nsteps);
      pt1.put("inner_conditioning", inner_conditioning);
      pt1.put("inner_seed", inner_seed);
      if (inner_propagator_block)
        pt1.put_child("inner_propagator", *inner_propagator_block);
    }
    std::unordered_set<std::string> pass_through_keys = {
      "system",
      "stochastic",
      "inner_nwalkers",
      "inner_nsteps",
      "inner_conditioning",
      "inner_seed",
      "inner_propagator",
      "inner_hamiltonian",
    };
    io::compare_known_keys("Wavefunction Factory",pt1, pt0,pass_through_keys);
    return pt1;
  }

  ~WavefunctionFactory() {}

  bool is_constructed(const std::string& ID)
  {
    auto xml = wfnBlocks.find(ID);
    if (xml == wfnBlocks.end())
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
                                Hamiltonian* h,
                                int targetNW   = 1)
  {
    auto xml = wfnBlocks.find(ID);
    if (xml == wfnBlocks.end())
    {
      app_log(1,"failed to find {}", ID);
      utils::check(false," Error in WavefunctionFactory::getWavefunction(string&): Missing wfn block. ");
    }
    auto w0 = wavefunctions.find(ID);
    if (w0 == wavefunctions.end())
    {
      auto neww = wavefunctions.insert(
          std::make_pair(ID, buildWavefunction(mpi,xml->second, walker_type, h, targetNW)));
      utils::check(neww.second," Error: Problems building new wavefunction in WavefunctionFactory::getWavefunction(string&). ");
      return (neww.first)->second;
    }
    else
      return w0->second;
  }

  void maybe_initialize_stochastic_inner_walkers(Wavefunction<MEM>& wfn,
                                                 const std::string& ID,
                                                 WALKER_TYPES walker_type,
                                                 ptree const& walker_pt)
  {
    auto xml = wfnBlocks.find(ID);
    if (xml == wfnBlocks.end())
      APP_ABORT(" Error in WavefunctionFactory::maybe_initialize_stochastic_inner_walkers: Missing wfn block. ");
    ptree pt = interpret_inputs(xml->second);
    if (not pt.get<bool>("stochastic"))
      return;
    if (not wfn.is_stochastic_wavefunction())
      return;
    if (wfn.stochastic_inner_walkers_initialized())
      return;
    std::string info = pt.get<std::string>("system");
    if (InfoMap.find(info) == InfoMap.end())
      APP_ABORT("ERROR: Undefined system in WavefunctionFactory::maybe_initialize_stochastic_inner_walkers.");
    int ndown = InfoMap[info].ndown;
    (void)walker_type;
    auto ig = initial_guess.find(ID);
    if (ig == initial_guess.end())
      APP_ABORT(" Error: Missing initial guess in WavefunctionFactory. ");
    wfn.initialize_stochastic_inner_walkers(walker_pt, ig->second, ndown);
  }

  // Use this routine to check if there is a wfn associated with a given ID
  ptree get_input(const std::string& ID) const
  {
    auto xml = wfnBlocks.find(ID);
    if (xml == wfnBlocks.end())
    {
      app_log(1,"failed to find {}", ID);
      utils::check(false,"Error: failed to find Wavefunction with above name.");
    }
    return xml->second;
  }

  // this routine allows you to modify the input block associated with ID 
  ptree& get_input(const std::string& ID) 
  {
    auto xml = wfnBlocks.find(ID);
    if (xml == wfnBlocks.end())
    {
      app_log(1,"failed to find {}", ID);
      utils::check(false,"Error: failed to find Wavefunction with above name.");
    }
    return xml->second;
  }

  // returns the xmlNodePtr associated with ID
  auto getInitialGuess(const std::string& ID) 
  {
    auto mat = initial_guess.find(ID);
    if (mat == initial_guess.end())
    {
      APP_ABORT(" Error: Missing initial guess in WavefunctionFactory. ");
    }
    // return view
    return mat->second();
  }

  // returns the xmlNodePtr associated with ID
  auto getInitialGuess(const std::string& ID) const
  {
    auto mat = initial_guess.find(ID);
    if (mat == initial_guess.end())
    {
      APP_ABORT(" Error: Missing initial guess in WavefunctionFactory. ");
    }
    // return view
    return mat->second();
  }

    // returns the xmlNodePtr associated with ID
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

  // returns the xmlNodePtr associated with ID
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

  // adds a xml block from which a Wavefunction can be built
  void push(const std::string& ID, ptree pt)
  {
    auto xml = wfnBlocks.find(ID);
    if (xml != wfnBlocks.end())
      APP_ABORT("Error: Repeated Wavefunction block in WavefunctionFactory. Wavefunction names must be unique. ");
    wfnBlocks.insert(std::make_pair(ID, pt));
  }

protected:
  // reference to container of AFQMCInfo objects
  std::map<std::string, AFQMCInfo>& InfoMap;

  // HamiltonianFactory used to build the inner (Variational) Hamiltonian on demand (Phase 3b-var).
  // Null when constructed via the original single-arg constructor; fromHDF5 aborts if a stochastic
  // trial then requests an inner_hamiltonian.
  HamiltonianFactory* HamFac_ = nullptr;

  // generates a new Wavefunction and returns the pointer to the base class
  Wavefunction<MEM> buildWavefunction(std::shared_ptr<utils::mpi_context_t<boost::mpi3::communicator>> mpi,
                                 ptree pt,
                                 WALKER_TYPES walker_type,
                                 Hamiltonian* h,
                                 int targetNW)
  {
    app_log(1,"\n****************************************************");
    app_log(1,"               Initializing Wavefunction ");
    app_log(1,"\n****************************************************");

    return fromHDF5(mpi, pt, walker_type, *h, targetNW);
  }

  Wavefunction<MEM> fromHDF5(std::shared_ptr<utils::mpi_context_t<boost::mpi3::communicator>> mpi,
                        ptree pt,
                        WALKER_TYPES walker_type,
                        Hamiltonian& h,
                        int targetNW);

  void getInitialGuess(h5::group grp, std::shared_ptr<utils::mpi_context_t<boost::mpi3::communicator>> mpi, std::string& name, int NMO, int nup, int ndown, WALKER_TYPES walker_type);
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

  std::map<std::string, ptree> wfnBlocks;

  std::map<std::string, Wavefunction<MEM>> wavefunctions;

  std::map<std::string, memory::const_shared_array<HOST_MEMORY, ComplexType, 3>> initial_guess;

  std::map<std::string, memory::const_shared_array<HOST_MEMORY, ComplexType, 4>> initial_guess_ft;
};
} // namespace afqmc
} // namespace sfqmc

