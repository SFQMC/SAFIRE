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

#include "utilities/mpi_context.h"

#include "AFQMC/Walkers/WalkerSetFactory.hpp"
#include "AFQMC/Hamiltonians/HamiltonianFactory.h"
#include "AFQMC/Wavefunctions/WavefunctionFactory.h"
#include "AFQMC/Propagators/PropagatorFactory.h"

namespace sfqmc
{
namespace afqmc
{
template<MEMORY_SPACE MEM>
class DriverFactory
{
  using communicator = boost::mpi3::communicator;

public:
  DriverFactory(std::shared_ptr<utils::mpi_context_t<boost::mpi3::communicator>> _mpi,
                std::map<std::string, AFQMCInfo>& info,
                WalkerSetFactory<MEM>& wsetfac_,
                PropagatorFactory<MEM>& pfac_,
                WavefunctionFactory<MEM>& wfnfac_,
                HamiltonianFactory& hfac)
      : mpi(_mpi), 
        InfoMap(info),
        WSetFac(wsetfac_),
        PropFac(pfac_),
        HamFac(hfac),
        WfnFac(wfnfac_)
  { }

  static ptree interpret_inputs_afqmc(const ptree pt0)
  {
    // "verbose" ptree 
    ptree pt1;

    // read inputs 
    // Rules (at least for afqmc):
    // 1. wavefunction must exist. It must be either a string referencing a previously defined wavefunction
    //    or a full input block.
    // 2. All other elements (walkerset, hamiltonian, propagator) can either be defined (by name or by 
    //    full specification) or not (in which case, default parameters are used).
    // 3. Unnamed and/or default blocks can not be referenced in future execute blocks.      

    if( auto wfn_pt = pt0.get_child_optional("wavefunction") ) {

      pt1.put_child("wavefunction",*wfn_pt);
		
      // empty value means default object
      for( auto& ss : {"system", "walker_set", "hamiltonian", "propagator"})
	      if( auto pt_child = pt0.get_child_optional(ss) ) 
          pt1.put_child(ss, *pt_child);

      auto hdf_read_file = pt0.get<std::string>("hdf_read_file", "");
      auto set_nwalker_to_target = pt0.get<bool>("set_nwalker_to_target", false);
      auto nWalkers = pt0.get<int>("n_walkers_per_mpi_task", 10);
      auto timestep = pt0.get<double>("timestep", 0.01);
      auto iseed = pt0.get<int>("seed", 0);
      // local energy importance sampling will ignore "initial_Eshift", if provided, with a warning
      auto Eshift = pt0.get<double>("initial_Eshift", 0.0); 
      pt1.put("hdf_read_file", hdf_read_file);
      pt1.put("set_nwalker_to_target", set_nwalker_to_target);
      pt1.put("n_walkers_per_mpi_task", nWalkers);
      pt1.put("timestep", timestep);
      pt1.put("seed", iseed);
      pt1.put("initial_Eshift", Eshift);
    } else 
      utils::check(false," wavefunction definition or declaration required in execution blocks.");
    // allow any keys that the execute block may use to pass through
    std::unordered_set<std::string> pass_through_keys = {
      "walker_set",
      "wavefunction",
      "propagator",
      "estimator",
      "hamiltonian",
      "hdf_write_file",
      "steps",
      "population_control_interval",
      "measure_interval_multiplier",
      "fix_bias",
      "walker_ortho_interval",
      "checkpoint_interval",
      "sample_interval",
      "weight_reset",
      "timestep",
      "dshift",
      "seed",
      "filename",
      "system",
      "ndets_to_read",
    };
    io::compare_known_keys("Driver factory" ,pt1, pt0, pass_through_keys);
    return pt1;
  }

    static ptree interpret_inputs_ftafqmc(const ptree pt0)
  {
    // "verbose" ptree 
    ptree pt1;

    // read inputs 
    // Rules (at least for afqmc):
    // 1. wavefunction must exist. It must be either a string referencing a previously defined wavefunction
    //    or a full input block.
    // 2. All other elements (walkerset, hamiltonian, propagator) can either be defined (by name or by 
    //    full specification) or not (in which case, default parameters are used).
    // 3. Unnamed and/or default blocks can not be referenced in future execute blocks.      

    if( auto wfn_pt = pt0.get_child_optional("wavefunction") ) {

      pt1.put_child("wavefunction",*wfn_pt);
		
      // empty value means default object
      for( auto& ss : {"system", "walker_set", "hamiltonian", "propagator"})
	      if( auto pt_child = pt0.get_child_optional(ss) ) 
          pt1.put_child(ss, *pt_child);

      auto hdf_read_file = pt0.get<std::string>("hdf_read_file", "");
      auto set_nwalker_to_target = pt0.get<bool>("set_nwalker_to_target", false);
      auto nWalkers = pt0.get<int>("n_walkers_per_mpi_task", 10);
      auto timestep = pt0.get<double>("timestep", 0.01);
      auto iseed = pt0.get<int>("seed", 0);
      // local energy importance sampling will ignore "initial_Eshift", if provided, with a warning
      auto Eshift = pt0.get<double>("initial_Eshift", 0.0); 
      pt1.put("hdf_read_file", hdf_read_file);
      pt1.put("set_nwalker_to_target", set_nwalker_to_target);
      pt1.put("n_walkers_per_mpi_task", nWalkers);
      pt1.put("timestep", timestep);
      pt1.put("seed", iseed);
      pt1.put("initial_Eshift", Eshift);
    } else 
      utils::check(false," wavefunction definition or declaration required in execution blocks.");
    // allow any keys that the execute block may use to pass through
    std::unordered_set<std::string> pass_through_keys = {
      "walker_set",
      "wavefunction",
      "propagator",
      "estimator",
      "hamiltonian",
      "hdf_write_file",
      "steps",
      "sweeps",
      "population_control_interval",
      "measure_interval_multiplier",
      "fix_bias",
      "walker_ortho_interval",
      "checkpoint_interval",
      "sample_interval",
      "weight_reset",
      "timestep",
      "dshift",
      "seed",
      "filename",
      "system",
      "ndets_to_read",
    };
    io::compare_known_keys("Driver factory" ,pt1, pt0, pass_through_keys);
    return pt1;
  }

  static ptree interpret_inputs_csafqmc(const ptree pt0)
  {
    ptree pt1;

    int n_systems = pt0.get<int>("n_systems", 0);
    
    utils::check(n_systems>0, "Error: n_systems < 1.");

    for(int i=0; i<n_systems; i++) {
      if( auto sys_pt = pt0.get_child_optional("cs_system_"+std::to_string(i)) ) 	
      {
        if( auto wfn_pt = sys_pt->get_child_optional("wavefunction") ) {
          pt1.put_child("cs_system_"+std::to_string(i),*sys_pt);
        } else {
          utils::check(false," wavefunction definition or declaration required in cs_system_N.");
        }
      } else {
        utils::check(false,"cs_system_N not found."); 
      }
      // check for unknown input keys
      std::unordered_set<std::string> pass_through_keys = {
        "walker_set",
        "wavefunction",
        "propagator",
        "estimator",
        "hamiltonian"
      };
      io::compare_known_keys("Driver factory, csafqmc",pt1, pt0, pass_through_keys);
    }

    auto hdf_read_file = pt0.get<std::string>("hdf_read_file", "");
    auto set_nwalker_to_target = pt0.get<bool>("set_nwalker_to_target", false);
    auto nWalkers = pt0.get<int>("n_walkers_per_mpi_task", 10);
    auto timestep = pt0.get<double>("timestep", 0.01);
    auto iseed = pt0.get<int>("seed", 0);
    pt1.put("n_systems", n_systems); 
    pt1.put("hdf_read_file", hdf_read_file);
    pt1.put("set_nwalker_to_target", set_nwalker_to_target);
    pt1.put("n_walkers_per_mpi_task", nWalkers);
    pt1.put("timestep", timestep);
    pt1.put("seed", iseed);

    return pt1;
  }

  ~DriverFactory() {}

  bool executeDriver(std::string type, std::string title, int m_series, ptree pt);

private:
  bool executeAFQMCDriver(std::string title, int m_seties, ptree pt);
  bool executeFTAFQMCDriver(std::string title, int m_seties, ptree pt);
  bool executeCSAFQMCDriver(std::string title, int m_series, ptree pt);

  std::shared_ptr<utils::mpi_context_t<boost::mpi3::communicator>> mpi;

  // container of AFQMCInfo objects
  std::map<std::string, AFQMCInfo>& InfoMap;

  // WalkerHandler factory
  WalkerSetFactory<MEM>& WSetFac;

  // Propagator factory
  PropagatorFactory<MEM>& PropFac;

  // Hamiltonian factory
  HamiltonianFactory& HamFac;

  // Wavefunction factory
  WavefunctionFactory<MEM>& WfnFac;

  int unique_id = 0;

  // Returns the name associated with the wavefunction input block in pt.
  // Can be either a name to a previously registered input block or a full 
  // (possibly nameless) declaration. 
  // After the successful return of this routine (e.g. wfn_name), we can assume that
  // WfnFac.get_input(wfn_name) exists and it contains a non-empty filename.
  std::string get_wavefunction_id(ptree pt);

  // similar to get_or_push, but customized for system
  std::string get_system_id(ptree pt, std::string wfn_name); 

  std::tuple<std::string,std::string,std::string,std::string,std::string>
    get_component_ids(ptree pt); 

  // this routine gets the node with key "key" from the property tree ptree. Then:
  // 1. If the node has a non-empty string as a value, it will check that the Factory
  //    provided has an input block defined with this identifier, otherwise the code aborts.
  //    In this case, the value of the node is returned. 
  // 2. If the node contains child ptrees, the node will be pushed into the provided Factory.
  //    If such node contains a "name", it is returned. If it doesn't, a unique name is made and returned.
  // 3. If the node contains an empty string and no child ptrees or no node is found, 
  //    then the provided (default) ptree is pushed into the factory with a unique name.        
  template<class Factory>
  std::string get_or_push(std::string key, ptree pt, Factory& fac, ptree default_ptree, std::string system)
  {
    std::string name("");
    if( auto pt_ = pt.get_child_optional(key) ) {
      if( pt_->size() > 0 ) {
        // assume declaration
        if( pt_->get<std::string>("system","") == "")
          pt_->put("system",system);
        auto val = pt_->get<std::string>("name","");
        if( val != "" ) {
          fac.push(val, *pt_);
          return val;
        } else {
          // unname block, set name and push
          name = key + std::string("_unique_id_") + std::to_string(++unique_id);
          pt_->put("name",name);
          fac.push(name, *pt_);
          return name;
        }
      } else if( auto val = pt_->get_value_optional<std::string>() ) {
        if(*val != "") {
          // a name is provided, retrieve it to make sure it exists 
          ptree dummy = fac.get_input(*val);
          if( dummy.get<std::string>("system","") == "" )
            fac.get_input(*val).put("system",system);
          return *val;
        }
      // if val=="", then construct further below
      } else
        utils::check(false," Error reading input: " + key);
    }
    // make name and push default
    if(name == "") {
      name = key + std::string("_unique_id_") + std::to_string(++unique_id);
      default_ptree.put("name",name);
      fac.push(name, default_ptree);
    }
    return name;
  }


};

} // namespace afqmc

} // namespace sfqmc

