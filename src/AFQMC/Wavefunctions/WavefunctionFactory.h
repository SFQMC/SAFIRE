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
#include "AFQMC/Hamiltonians/hdf5_helpers.hpp"
#include "AFQMC/Wavefunctions/Wavefunction.hpp"
#include "AFQMC/HamiltonianOperations/HamiltonianOperations.h"
#include "IO/app_loggers.h"

namespace sfqmc
{
namespace afqmc
{

// First-class stochastic trial input via type: stochasticwfn.
inline bool is_stochastic_wavefunction_input(ptree const& pt0)
{
  if (auto type_opt = pt0.get_optional<std::string>("type"))
  {
    std::string const type = *type_opt;
    if (type == "stochasticwfn" || type == "stochastic_wfn")
      return true;
    if (type == "nomsd" || type == "phmsd")
      return false;
    APP_ABORT("Error in WavefunctionFactory: unknown wavefunction type: " + type);
  }
  return false;
}

template<MEMORY_SPACE MEM>
class WavefunctionFactory
{
public:
  WavefunctionFactory() = default;

  // Optional HamiltonianFactory for StochasticWfn `inner_hamiltonian`. Null when
  // default-constructed; fromHDF5 aborts if a stochastic trial then requests inner_hamiltonian.
  explicit WavefunctionFactory(HamiltonianFactory& hamfac) : HamFac_(&hamfac) {}

  static ptree interpret_inputs(const ptree pt0)
  {
    // check required fields exist
    if(not io::check_exists<std::string>(pt0,"name"))
      APP_ABORT("Error in WavefunctionFactory: missing required input: name \n");
    if(not io::check_exists<std::string>(pt0,"filename"))
      APP_ABORT("Error in WavefunctionFactory: missing required input: filename \n");
    // read inputs with default options
    int ndets_to_read = pt0.get<int>("ndets_to_read", -1);
    std::string name          = pt0.get<std::string>("name");
    std::string filename      = pt0.get<std::string>("filename");
//    std::string restart_file  = pt0.get<std::string>("restart_file", "");
    bool rediag        = pt0.get<bool>("rediag", false);
    // validate inputs
    // create verbose internal inputs
    ptree pt1;
    pt1.put("name", name);
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
    bool stochastic = is_stochastic_wavefunction_input(pt0);
    if (pt0.get<bool>("stochastic", false) && not pt0.get_child_optional("type"))
      APP_ABORT("Error in WavefunctionFactory: stochastic: true is no longer supported; use type: stochasticwfn.");
    int inner_nwalkers = pt0.get<int>("inner_nwalkers", 1);
    if (inner_nwalkers < 1)
      APP_ABORT("Error in WavefunctionFactory::interpret_inputs: inner_nwalkers must be >= 1.");
    int inner_nsteps = pt0.get<int>("inner_nsteps", 0);
    bool inner_conditioning = pt0.get<bool>("inner_conditioning", false);
    bool inner_leapfrog = pt0.get<bool>("inner_leapfrog", false);
    bool inner_persistence = pt0.get<bool>("inner_persistence", false);
    int inner_equil_steps = pt0.get<int>("inner_equil_steps", 1);
    int inner_pool_burn_in = pt0.get<int>("inner_pool_burn_in", 0);
    // Measurement-replica averaging (nm). Stride default tracks inner_equil_steps -- keep this in step
    // with StochasticWfn::interpret_inputs, which computes the same default.
    int inner_measure_replicas = pt0.get<int>("inner_measure_replicas", 1);
    int inner_measure_stride = pt0.get<int>("inner_measure_stride", inner_equil_steps);
    bool inner_measure_restore = pt0.get<bool>("inner_measure_restore", true);
    std::string inner_mcmc = pt0.get<std::string>("inner_mcmc", "pcn");
    // pcn default s = 1 (independence proposal): validated conditioned-path default (see StochasticWfn).
    double inner_mcmc_step = pt0.get<double>("inner_mcmc_step", inner_mcmc == "gaussian" ? 0.05 : 1.0);
    bool inner_log_aggregate = pt0.get<bool>("inner_log_aggregate", false);
    // Current-walker conditioning: advance persistent pool at end-of-step against phi_new.
    bool inner_condition_on_new = pt0.get<bool>("inner_condition_on_new", false);
    int inner_seed   = pt0.get<int>("inner_seed", 777);
    auto inner_propagator_block = pt0.get_child_optional("inner_propagator");
    // inner_hamiltonian: optional block naming the second (Variational) Hamiltonian HDF5 file for the
    // stochastic inner stack. It is a factory-level key consumed by fromHDF5 (which builds the Ham via
    // HamFac_); it is deliberately NOT forwarded into pt1, so it never reaches the wavefunction's own
    // ptree (StochasticWfn::interpret_inputs does not know it). interpret_inputs only (a) rejects it
    // when stochastic is off and (b) lists it as a known pass-through key for compare_known_keys.
    for (auto const& key :
         {"inner_nwalkers", "inner_nsteps", "inner_conditioning", "inner_leapfrog", "inner_persistence",
          "inner_equil_steps", "inner_pool_burn_in", "inner_measure_replicas", "inner_measure_stride",
          "inner_measure_restore", "inner_mcmc", "inner_mcmc_step",
          "inner_log_aggregate", "inner_condition_on_new", "inner_seed", "inner_propagator",
          "inner_hamiltonian"})
      if (not stochastic && pt0.get_child_optional(key))
        APP_ABORT("Error in WavefunctionFactory::interpret_inputs: " + std::string(key) +
                  " requires type: stochasticwfn.");
    if (stochastic)
    {
      pt1.put("inner_nwalkers", inner_nwalkers);
      pt1.put("inner_nsteps", inner_nsteps);
      pt1.put("inner_conditioning", inner_conditioning);
      pt1.put("inner_leapfrog", inner_leapfrog);
      pt1.put("inner_persistence", inner_persistence);
      pt1.put("inner_equil_steps", inner_equil_steps);
      pt1.put("inner_pool_burn_in", inner_pool_burn_in);
      pt1.put("inner_measure_replicas", inner_measure_replicas);
      pt1.put("inner_measure_stride", inner_measure_stride);
      pt1.put("inner_measure_restore", inner_measure_restore);
      pt1.put("inner_mcmc", inner_mcmc);
      pt1.put("inner_mcmc_step", inner_mcmc_step);
      pt1.put("inner_log_aggregate", inner_log_aggregate);
      pt1.put("inner_condition_on_new", inner_condition_on_new);
      pt1.put("inner_seed", inner_seed);
      if (inner_propagator_block)
        pt1.put_child("inner_propagator", *inner_propagator_block);
    }
    if( auto val = pt0.get_optional<int>("nwalk_block_size") )
      pt1.put("nwalk_block_size", *val);
    if( auto val = pt0.get_optional<int>("ndet_block_size") )
      pt1.put("ndet_block_size", *val);
    std::unordered_set<std::string> pass_through_keys = {
      "system",
      "type",
      "inner_nwalkers",
      "inner_nsteps",
      "inner_conditioning",
      "inner_leapfrog",
      "inner_persistence",
      "inner_equil_steps",
      "inner_pool_burn_in",
      "inner_measure_replicas",
      "inner_measure_stride",
      "inner_measure_restore",
      "inner_mcmc",
      "inner_mcmc_step",
      "inner_log_aggregate",
      "inner_condition_on_new",
      "inner_seed",
      "inner_propagator",
      "inner_hamiltonian",
    };
    io::compare_known_keys("Wavefunction Factory",pt1, pt0,pass_through_keys);
    return pt1;
  }

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
                                bool finiteT,
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
          std::make_pair(ID, buildWavefunction(mpi,xml->second, walker_type, finiteT, h, targetNW)));
      utils::check(neww.second," Error: Problems building new wavefunction in WavefunctionFactory::getWavefunction(string&). ");
      return (neww.first)->second;
    }
    else
      return w0->second;
  }

  // Backward-compatible overload for call sites that omit finiteT.
  auto& getWavefunction(std::shared_ptr<utils::mpi_context_t<boost::mpi3::communicator>> mpi,
                                const std::string& ID,
                                WALKER_TYPES walker_type,
                                Hamiltonian* h,
                                int targetNW   = 1)
  {
    return getWavefunction(mpi, ID, walker_type, false, h, targetNW);
  }

  void maybe_initialize_stochastic_inner_walkers(Wavefunction<MEM>& wfn,
                                                 const std::string& ID,
                                                 WALKER_TYPES walker_type,
                                                 ptree const& walker_pt)
  {
    auto xml = wfnBlocks.find(ID);
    if (xml == wfnBlocks.end())
      APP_ABORT(" Error in WavefunctionFactory::maybe_initialize_stochastic_inner_walkers: Missing wfn block. ");
    if (not wfn.is_stochastic_wavefunction())
      return;
    if (wfn.stochastic_inner_walkers_initialized())
      return;
    ptree pt = interpret_inputs(xml->second);
    int ndown = std::get<2>(read_info_from_wfn(pt.get<std::string>("filename"), "any"));
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
  HamiltonianFactory* HamFac_ = nullptr;

  // generates a new Wavefunction and returns the pointer to the base class
  Wavefunction<MEM> buildWavefunction(std::shared_ptr<utils::mpi_context_t<boost::mpi3::communicator>> mpi,
                                 ptree pt,
                                 WALKER_TYPES walker_type,
                                 bool finiteT,
                                 Hamiltonian* h,
                                 int targetNW)
  {
    app_log(1,"\n****************************************************");
    app_log(1,"               Initializing Wavefunction ");
    app_log(1,"\n****************************************************");

    return fromHDF5(mpi, pt, walker_type, finiteT, *h, targetNW);
  }

  Wavefunction<MEM> fromHDF5(std::shared_ptr<utils::mpi_context_t<boost::mpi3::communicator>> mpi,
                        ptree pt,
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

  std::map<std::string, ptree> wfnBlocks;

  std::map<std::string, Wavefunction<MEM>> wavefunctions;

  // per-spin trial orbital matrices, sized to the walker (alpha: npol*NMO x naea,
  // beta: NMO x naeb for collinear). The true widths keep {rows, naea, naeb}
  // inferable by the walker set directly from the guess.
  std::map<std::string, std::vector<nda::matrix<ComplexType>>> initial_guess;

  std::map<std::string, memory::const_shared_array<HOST_MEMORY, ComplexType, 4>> initial_guess_ft;
};
} // namespace afqmc
} // namespace sfqmc

