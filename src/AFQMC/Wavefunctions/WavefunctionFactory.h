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

#ifndef SFQMC_AFQMC_WAVEFUNCTIONFACTORY_H
#define SFQMC_AFQMC_WAVEFUNCTIONFACTORY_H

#include <iostream>
#include <vector>
#include <map>
#include <fstream>
#include <boost/optional.hpp>

#include "AFQMC/config.h"
#include "AFQMC/Utilities/taskgroup.h"
#include "AFQMC/Hamiltonians/Hamiltonian.hpp"
#include "AFQMC/Wavefunctions/Wavefunction.hpp"
#include "AFQMC/HamiltonianOperations/HamiltonianOperations.hpp"

namespace sfqmc
{
namespace afqmc
{
class WavefunctionFactory
{
public:
  WavefunctionFactory(std::map<std::string, AFQMCInfo>& info, bool prec) : 
	InfoMap(info), mixed_precision(prec) 
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
    std::string restart_file  = pt0.get<std::string>("restart_file", "");
    bool rediag        = pt0.get<bool>("rediag", false);
    bool stochastic    = pt0.get<bool>("stochastic", false);
    int inner_nwalkers = pt0.get<int>("inner_nwalkers", 1);
    if (inner_nwalkers < 1)
      APP_ABORT("Error in WavefunctionFactory::interpret_inputs: inner_nwalkers must be >= 1.");
    if (not stochastic && pt0.get_child_optional("inner_nwalkers"))
      APP_ABORT("Error in WavefunctionFactory::interpret_inputs: inner_nwalkers requires stochastic: true.");
    // validate inputs
    // create verbose internal inputs
    ptree pt1;
    pt1.put("name", name);
    pt1.put("system", info);
    pt1.put("filename", filename);
    pt1.put("restart_file", restart_file);
    pt1.put("rediag", rediag);
    pt1.put("ndets_to_read", ndets_to_read);
    pt1.put("stochastic", stochastic);
    if (stochastic)
      pt1.put("inner_nwalkers", inner_nwalkers);
    // optional parameters 
    if( auto val = pt0.get_optional<int>("algorithm") )
      pt1.put("algorithm", *val);
    // set default later, since it depends on HamiltonianOperations type
    if( auto val = pt0.get_optional<bool>("dense_trial") )
      pt1.put("dense_trial", *val);
    std::unordered_set<std::string> pass_through_keys = {
        "system",
        "stochastic",
        "inner_nwalkers",
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
  Wavefunction& getWavefunction(TaskGroup_& TGprop,
                                TaskGroup_& TGwfn,
                                const std::string& ID,
                                WALKER_TYPES walker_type,
                                Hamiltonian* h,
                                RealType cutvn = 1e-6,
                                int targetNW   = 1,
                                ptree const* walker_pt = nullptr)
  {
    auto xml = wfnBlocks.find(ID);
    if (xml == wfnBlocks.end())
    {
      app_log(1,"failed to find {}", ID);
      APP_ABORT(" Error in WavefunctionFactory::getWavefunction(string&): Missing wfn block. ");
    }
    auto w0 = wavefunctions.find(ID);
    if (w0 == wavefunctions.end())
    {
      auto neww = wavefunctions.insert(
          std::make_pair(ID, buildWavefunction(TGprop, TGwfn, xml->second, walker_type, h, cutvn, targetNW)));
      if (!neww.second)
        APP_ABORT(" Error: Problems building new wavefunction in WavefunctionFactory::getWavefunction(string&). ");
      if (walker_pt != nullptr)
        maybe_initialize_stochastic_inner_walkers((neww.first)->second, ID, walker_type, *walker_pt);
      return (neww.first)->second;
    }
    else
    {
      if (walker_pt != nullptr)
        maybe_initialize_stochastic_inner_walkers(w0->second, ID, walker_type, *walker_pt);
      return w0->second;
    }
  }

  // Phase 1a: initialize owned inner WalkerSet for stochastic trial wavefunctions.
  void maybe_initialize_stochastic_inner_walkers(Wavefunction& wfn,
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
    int NAEB = InfoMap[info].NAEB;
    (void)walker_type;
    wfn.initialize_stochastic_inner_walkers(walker_pt, getInitialGuess(ID), NAEB);
  }

  // Use this routine to check if there is a wfn associated with a given ID
  ptree get_input(const std::string& ID) const
  {
    auto xml = wfnBlocks.find(ID);
    if (xml == wfnBlocks.end())
    {
      app_log(1,"failed to find {}", ID);
      APP_ABORT("Error: failed to find Wavefunction with above name.");
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
      APP_ABORT("Error: failed to find Wavefunction with above name.");
    }
    return xml->second;
  }

  // returns the xmlNodePtr associated with ID
  boost::multi::array<ComplexType, 3>& getInitialGuess(const std::string& ID)
  {
    auto mat = initial_guess.find(ID);
    if (mat == initial_guess.end())
    {
      APP_ABORT(" Error: Missing initial guess in WavefunctionFactory. ");
    }
    return mat->second;
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

  // defines working precision
  bool mixed_precision;

  // generates a new Wavefunction and returns the pointer to the base class
  Wavefunction buildWavefunction(TaskGroup_& TGprop,
                                 TaskGroup_& TGwfn,
                                 ptree pt,
                                 WALKER_TYPES walker_type,
                                 Hamiltonian* h,
                                 RealType cutvn,
                                 int targetNW)
  {
    std::string fwf_type = pt.get<std::string>("filetype", "hdf5");

    app_log(1,"\n****************************************************");
    app_log(1,"               Initializing Wavefunction ");
    app_log(1,"\n****************************************************");

    if (fwf_type == "hdf5")
      return fromHDF5(TGprop, TGwfn, pt, walker_type, *h, cutvn, targetNW);
    else
    {
      app_error("Unknown Wavefunction filetype in WavefunctionFactory::buildWavefunction(): {}",
		    fwf_type);
      APP_ABORT(" Error: Unknown Wavefunction filetype in WavefunctionFactory::buildWavefunction(). ");
    }
    return Wavefunction{};
  }

  Wavefunction fromHDF5(TaskGroup_& TGprop,
                        TaskGroup_& TGwfn,
                        ptree pt,
                        WALKER_TYPES walker_type,
                        Hamiltonian& h,
                        RealType cutvn,
                        int targetNW);
  template<bool MP>
  HamiltonianOperations<MP> getHamOps(std::string const& restart_file,
                                  WALKER_TYPES type,
                                  int NMO,
                                  int NAEA,
                                  int NAEB,
                                  std::vector<PsiT_Matrix>& PsiT,
                                  TaskGroup_& TGprop,
                                  TaskGroup_& TGwfn,
                                  Hamiltonian& h);
  void getInitialGuess(hdf_archive& dump, std::string& name, int NMO, int NAEA, int NAEB, WALKER_TYPES walker_type);
  int getExcitation(boost::multi::array_ref<int, 1>& deti,
                    boost::multi::array_ref<int, 1>& detj,
                    std::vector<int>& excit,
                    int& perm);
  void computeVariationalEnergyPHMSD(TaskGroup_& TG,
                                     Hamiltonian& ham,
                                     boost::multi::array_ref<int, 2>& occs,
                                     std::vector<ComplexType>& coeff,
                                     int ndets,
                                     int NAEA,
                                     int NAEB,
                                     int NMO,
                                     bool recomputeCI);
  ComplexType slaterCondon0(Hamiltonian& ham, boost::multi::array_ref<int, 1>& det, int NMO);
  ComplexType slaterCondon1(Hamiltonian& ham, std::vector<int>& excit, boost::multi::array_ref<int, 1>& det, int NMO);
  ComplexType slaterCondon2(Hamiltonian& ham, std::vector<int>& excit, int NMO);

  void build_PsiT_MO_phmsd(TaskGroup_& TGwfn, WALKER_TYPES walker_type, int NPOL, int NMO, int NAEA, 
	int NAEB, int ndets, std::vector<ComplexType>& coeffs, 
        std::vector<int>& occbuff, std::vector<PsiT_Matrix>& PsiT_MO);

  std::map<std::string, ptree> wfnBlocks;

  std::map<std::string, Wavefunction> wavefunctions;

  std::map<std::string, boost::multi::array<ComplexType, 3>> initial_guess;
};
} // namespace afqmc
} // namespace sfqmc

#endif
