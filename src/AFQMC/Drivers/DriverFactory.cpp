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

#include <iomanip>

#include "config.h"
#include "utilities/check.hpp"
#include "IO/ptree/ptree_utilities.hpp"

#include "utilities/mpi_context.h"
#include "nda/nda.hpp"
#include "nda/h5.hpp"

#include "AFQMC/Hamiltonians/hdf5_helpers.hpp"
#include "utilities/Random.hpp"
#include "AFQMC/Drivers/DriverFactory.h"
#include "AFQMC/Drivers/AFQMCDriver.h"
#include "AFQMC/Drivers/FTAFQMCDriver.h"
//#include "AFQMC/Drivers/CSAFQMCDriver.h"
#include "AFQMC/Walkers/WalkerIO.hpp"

#include "AFQMC/Walkers/WalkerSetFactory.hpp"
#include "AFQMC/Hamiltonians/HamiltonianFactory.h"
#include "AFQMC/Wavefunctions/WavefunctionFactory.h"
#include "AFQMC/Propagators/PropagatorFactory.h"

#include "AFQMC/Walkers/WalkerSet.hpp"
#include "AFQMC/Hamiltonians/Hamiltonian.hpp"
#include "AFQMC/Wavefunctions/Wavefunction.hpp"
#include "AFQMC/Propagators/Propagator.hpp"
#include "AFQMC/Estimators/EstimatorHandler.h"

namespace sfqmc
{
namespace afqmc
{

// assumes wfn.Energy(Wset) has been called
// Then prints the energy breakdown
template<typename WlkSet>
void print_initial_energy(WlkSet& wset){
  app_log(1," Local Energy of starting determinant ");
  //app_log(1," <psi_T|H|w_0>/<psi_T|w_0>: ");
  app_log(1,"  - Total energy    : {:f}", wset[0].energy());
  app_log(1,"  - One-body energy : {:f}", wset[0].get_property(E1_));
  app_log(1,"  - Coulomb energy  : {:f}", wset[0].get_property(EJ_));
  app_log(1,"  - Exchange energy : {:f}", wset[0].get_property(EXX_));
}

template<MEMORY_SPACE MEM>
bool DriverFactory<MEM>::executeDriver(std::string type, std::string title, 
				  int m_series, ptree pt)
{
  if (type == "afqmc")
  {
    return executeAFQMCDriver(title, m_series, pt);
  }
  else if(type == "ftafqmc")
  {
    return executeFTAFQMCDriver(title, m_series, pt);
  }
  else if(type == "csafqmc") 
  {
    return executeCSAFQMCDriver(title, m_series, pt);
  }
  else
  {
    app_error("Unknown execute driver: {}", type);
    utils::check(false," Unknown execute driver.  ");
    return false;
  }
}

// Returns the name associated with the wavefunction input block in pt.
// Can be either a name to a previously registered input block or a full 
// (possibly nameless) declaration. 
// After the successful return of this routine (e.g. wfn_name), we can assume that
// WfnFac.get_input(wfn_name) exists and it contains a non-empty filename.
template<MEMORY_SPACE MEM>
std::string DriverFactory<MEM>::get_wavefunction_id(ptree pt)
{
  std::string name("");
  if( auto pt_ = pt.get_child_optional("wavefunction") ) {
    if( pt_->size() > 0 ) {
      // found input block, push into Factory and return name
                 name = pt_->get<std::string>("name","");
      if(name == "") {
        name = std::string("sysid_") + std::to_string(++unique_id);
        pt_->put("name",name);
      }
      if(pt_->get<std::string>("filename","") == "")
        utils::check(false," Error: wavefunction must contain a filename. ");
      WfnFac.push(name,*pt_);
    } else if( auto val = pt_->get_value_optional<std::string>() ) {
      // found id
      if(*val != "") {
        // check that it exists in factory
        auto wfn_pt = WfnFac.get_input(*val);
        // check that it contains a non-empty filename
        if(wfn_pt.template get<std::string>("filename","") == "")
          utils::check(false," Error: wavefunction must contain a filename. ");
        return *val;
      } else
        utils::check(false," Error: Problems with wavefunction input. ");
    } else
        utils::check(false," Error: Problems reading wavefunction input. ");
    } else
    utils::check(false," Error: wavefunction input not found. It is required. ");
  return name;
}

// similar to get_or_push, but customized for AFQMCInfo
template<MEMORY_SPACE MEM>
std::string DriverFactory<MEM>::get_system_id(ptree pt, std::string wfn_name)
{
  std::string name("");
  auto wfn_pt = WfnFac.get_input(wfn_name);
  std::string filename = wfn_pt.template get<std::string>("filename","");
  const auto [nmo,nup,ndn] = read_info_from_wfn(filename,"any");
  if(ndn > nup)
    utils::check(false," Error  nup < ndown: Up spin must be the majority spin. nup: {}, ndown: {}",nup,ndn);
  if( auto pt_ = pt.get_child_optional("system") ) {
    if(pt_->size() > 0) {
      // input block provided, build ptree to reuse parse routine
      name = pt_->template get<std::string>("name","");
      if(name == "")
        name = std::string("sysid_") + std::to_string(++unique_id);
      if(InfoMap.find(name) == InfoMap.end()) {
        //note: for finiteT ntau = nup, ndn = 0
        AFQMCInfo info(name,nmo,nup,ndn,nup);
        InfoMap.insert(std::pair<std::string, AFQMCInfo>(info.name, info));
      }
    } else if( auto val = pt_->get_value_optional<std::string>() ) {
      // id of previously declared block provided
      if(*val != "") name = *val;
      else utils::check(false," Error: Found empty string in system tag.");
    } else
      utils::check(false," Error: Can't convert system value to string.");
  } else {
    // not found, build from wavefunction input 
    name = std::string("sysid_") + std::to_string(++unique_id);
    //note: for finiteT ntau = nup, ndn = 0
    AFQMCInfo info(name,nmo,nup,ndn,nup);
    InfoMap.insert(std::pair<std::string, AFQMCInfo>(info.name, info));
  }
  // check for compatibility
  utils::check(InfoMap.find(name) != InfoMap.end(), "Error: Unregistered system id:{}", name);
  {
    auto info = InfoMap.find(name)->second;
    utils::check(nmo == info.NMO," Error: Inconsistent definition of NMO between system and wavefunction.");
    utils::check(nup == info.nup," Error: Inconsistent definition of nup between system and wavefunction.");
    utils::check(ndn == info.ndown," Error: Inconsistent definition of ndown between system and wavefunction.");
  } 

  // add to WfnFac.get_input(wfn_name) if missing
  if(wfn_pt.template get<std::string>("system","") == "")
    WfnFac.get_input(wfn_name).put("system",name);
  return name;
}

template<MEMORY_SPACE MEM>
std::tuple<std::string,std::string,std::string,std::string,std::string>
    DriverFactory<MEM>::get_component_ids(ptree pt)
{
  // 1. get wavefunction id, push input block if necessary 
  std::string wfn_name = get_wavefunction_id(pt); //pt.get<std::string>("wavefunction");
  // 2. get or create system id. At this stage, 
  //    WfnFac.get_input(wfn_name) must exist 
  //    Adds system tag to wfn_pt if missing
  std::string system = get_system_id(pt,wfn_name);
  // default ptree in case input blocks are default constructed
  ptree pt_default;
  pt_default.put("system",system);
  // 3. get ids of hamiltonian, walker_set and propagator. Build later.
  std::string wset_name = get_or_push("walker_set",pt,WSetFac,pt_default,system);
  std::string prop_name = get_or_push("propagator",pt,PropFac,pt_default,system);
  // add filename from wavefunction fo default input for hamiltonian
  {
    auto wfn_pt = WfnFac.get_input(wfn_name);
    pt_default.put("filename", wfn_pt.template get<std::string>("filename"));
  }
  std::string ham_name = get_or_push("hamiltonian",pt,HamFac,pt_default,system);

  return std::make_tuple(system,ham_name,wfn_name,wset_name,prop_name);
}

template<MEMORY_SPACE MEM>
bool DriverFactory<MEM>::executeAFQMCDriver(std::string title, int m_series, ptree pt_in)
{
  // reset timers
  AFQMCTimer.reset_all();
  // convert user input to verbose input
  ptree pt = interpret_inputs_afqmc(pt_in);
  app_log(2,"\nDrvFac::executeAFQMCDriver input:\n{}\n",io::to_string(pt));
  // initialize using verbose input
  auto [system,ham_name,wfn_name,wset_name,prop_name] = get_component_ids(pt);

  if (InfoMap.find(system) == InfoMap.end())
  {
    app_error("ERROR: Undefined system in execute block. ");
    return false;
  }
  auto& AFinfo = InfoMap[system];

  std::string hdf_read_restart;
  bool set_nWalker_target;
  double dt;
  hdf_read_restart = pt.get<std::string>("hdf_read_file");
  set_nWalker_target = pt.get<bool>("set_nwalker_to_target");
  dt = pt.get<double>("timestep");
  int nWalkers = pt.get<int>("n_walkers_per_mpi_task");

  bool restarted = false;
  int step0      = 0;
  int block0     = 0;
  double Eshift =  pt.get<double>("initial_Eshift");

  utils::RandomGenerator_t<>::result_type iseed = ( (pt.get<int>("seed") == 0) ? 
					   utils::make_seed(mpi->comm) : 
					   utils::split_seed(pt.get<int>("seed"),mpi->comm));
  std::shared_ptr<utils::RandomGenerator_t<>> rng_wlk = std::make_shared<utils::RandomGenerator_t<>>(iseed);
  iseed = ( (pt.get<int>("seed") == 0) ? utils::make_seed(mpi->comm) : 
					 utils::split_seed(pt.get<int>("seed"),mpi->comm));
  std::shared_ptr<utils::RandomGenerator_t<MEM>> rng = std::make_shared<utils::RandomGenerator_t<MEM>>(utils::make_rng<MEM>(iseed));

  app_log(1,"\n****************************************************");
  app_log(1,"****************************************************");
  app_log(1,"****************************************************");
  app_log(1,"          Beginning Driver initialization.");
  app_log(1,"****************************************************");
  app_log(1,"****************************************************");
  app_log(1,"****************************************************\n");

  /*
   * Note: Hamiltonian is only needed to construct Wavefunction.
   *       If Wavefunction already exists in the factory (constructed in a previous exec block)
   *       there is no need to build Hamiltonian.
   */
  if (mpi->comm.root() == 0)
  {
    if (hdf_read_restart != std::string(""))
    {
      h5::file file(hdf_read_restart,'r');
      h5::group grp(file);
      if (not grp.has_key("AFQMCDriver")) return false;
      h5::group dgrp = grp.open_group("AFQMCDriver");
      
      std::vector<IndexType> Idata(2);
      std::vector<RealType> Rdata(2);

      h5::h5_read(dgrp,"DriverInts",Idata);
      h5::h5_read(dgrp,"DriverReals",Rdata);

      Eshift = Rdata[0];
      block0 = Idata[0];
      step0  = Idata[1];
      restarted = true;
    }
  }
  mpi->comm.broadcast_value(restarted);
  if (restarted)
  {
    app_log(1," Restarted from file. Block={}, step={}",block0,step0);
    app_log(1,"                      Eshift: {}", Eshift);
    mpi->comm.broadcast_value(Eshift);
    mpi->comm.broadcast_value(block0);
    mpi->comm.broadcast_value(step0);
  }

  /*
   * to do:
   *  - add logic for estimators, e.g. whether to evaluate energy, which wfn to use, etc.
   */

  // walker set and type
  auto& wset          = WSetFac.getWalkerSet(mpi, wset_name, rng_wlk);
  WALKER_TYPES walker_type = wset.getWalkerType();

  if (not WfnFac.is_constructed(wfn_name))
  {
    // hamiltonian
    Hamiltonian& ham0 = HamFac.getHamiltonian(mpi, ham_name);

    // build wavefunction
    WfnFac.getWavefunction(mpi, wfn_name, walker_type, &ham0, nWalkers);
  }

  // wfn builder should not use Hamiltonian pointer now
  auto& wfn0 = WfnFac.getWavefunction(mpi, wfn_name, walker_type, nullptr, nWalkers);
  WfnFac.maybe_initialize_stochastic_inner_walkers(wfn0, wfn_name, walker_type, WSetFac.get_input(wset_name));

  // propagator
  auto& prop0 = PropFac.getPropagator(mpi, prop_name, wfn0, rng);
  bool hybrid       = prop0.hybrid_propagation();
  // resize walker set
  if (restarted)
  {
    h5::file file(hdf_read_restart,'r');
    restartFromHDF5(wset, nWalkers, file, set_nWalker_target);
    // perform runtime optimization
    wfn0.runtime_optimization(wset);   
    wfn0.Energy(wset);
  }
  else
  {
    auto initial_guess = WfnFac.getInitialGuess(wfn_name);
    wset.resize(nWalkers, initial_guess()); 
    // perform runtime optimization
    wfn0.runtime_optimization(wset);   
    wfn0.Energy(wset);
    print_initial_energy(wset);
    if (hybrid)
    {
      // Eshift defaults to 0.0 if not provided in input
      //    otherwise, use the value from input with warning
      if (Eshift != 0.0)
      {
        app_warning("user set expert-level parameter, \"initial_Eshift\" : Using user-provided initial Eshift = {}", Eshift);
      }
    } else {
      if (Eshift != 0.0)
      {
        app_log(1, "[Warning] : User set initial Eshift {} with local energy importance. This value is ignored.", Eshift);
      }
      Eshift = real(ComplexType(wset[0].energy()));
    }
  }

  // is this run using importance sampling? 
// MAM: should be asking for importance sampling and not for free_propagation...
  bool free_proj = prop0.free_propagation();
  // if hybrid calculation, set to true
  bool addEnergyEstim = hybrid;

  // estimator setup
  auto estim0 = EstimatorHandler<MEM>(mpi, AFinfo, title, pt_in, wset, WfnFac, wfn0, 
         prop0, walker_type, HamFac, ham_name, dt, addEnergyEstim, !free_proj);

  app_log(1,"\n****************************************************");
  app_log(1,"****************************************************");
  app_log(1,"****************************************************");
  app_log(1,"          Finished Driver initialization.");
  app_log(1,"****************************************************");
  app_log(1,"****************************************************");
  app_log(1,"****************************************************\n");

  AFQMCDriver<MEM> driver(mpi, AFinfo, title, m_series, block0, step0, Eshift, pt_in, wfn0, prop0, estim0);

  // free any shared windows that were abandoned during initialization
  mpi->shared_windows.collective_free_unused();

  if (!driver.run(wset))
  {
    app_error(" Problems with AFQMCDriver::run().");
    return false;
  }

  if (!driver.clear())
  {
    app_error(" Problems with AFQMCDriver::clear().");
    return false;
  }

  return true;
}


template<MEMORY_SPACE MEM>
bool DriverFactory<MEM>::executeFTAFQMCDriver(std::string title, int m_series, ptree pt_in)
{
  // reset timers
  AFQMCTimer.reset_all();
  // convert user input to verbose input
  ptree pt = interpret_inputs_ftafqmc(pt_in);
  app_log(2,"\nDrvFac::executeFTAFQMCDriver input:\n{}\n",io::to_string(pt));
  // initialize using verbose input
  auto [system,ham_name,wfn_name,wset_name,prop_name] = get_component_ids(pt);

  if (InfoMap.find(system) == InfoMap.end())
  {
    app_error("ERROR: Undefined system in execute block. ");
    return false;
  }
  auto& AFinfo = InfoMap[system];
  //int NMO      = AFinfo.NMO;
  //int ndown     = AFinfo.ndown;

  std::string hdf_read_restart;
  bool set_nWalker_target;
  double dt;
  hdf_read_restart = pt.get<std::string>("hdf_read_file");
  set_nWalker_target = pt.get<bool>("set_nwalker_to_target");
  dt = pt.get<double>("timestep");
  int nWalkers = pt.get<int>("n_walkers_per_mpi_task");

  bool restarted = false;
  int step0      = 0;
  int block0     = 0;
  double Eshift =  pt.get<double>("initial_Eshift");

  utils::RandomGenerator_t<>::result_type iseed = ( (pt.get<int>("seed") == 0) ? 
					   utils::make_seed(mpi->comm) : 
					   utils::split_seed(pt.get<int>("seed"),mpi->comm));
  std::shared_ptr<utils::RandomGenerator_t<>> rng_wlk = std::make_shared<utils::RandomGenerator_t<>>(iseed);
  iseed = ( (pt.get<int>("seed") == 0) ? utils::make_seed(mpi->comm) : 
					 utils::split_seed(pt.get<int>("seed"),mpi->comm));
  std::shared_ptr<utils::RandomGenerator_t<MEM>> rng = std::make_shared<utils::RandomGenerator_t<MEM>>(utils::make_rng<MEM>(iseed));

  app_log(1,"\n****************************************************");
  app_log(1,"****************************************************");
  app_log(1,"****************************************************");
  app_log(1,"          Beginning Driver initialization.");
  app_log(1,"****************************************************");
  app_log(1,"****************************************************");
  app_log(1,"****************************************************\n");

  /*
   * Note: Hamiltonian is only needed to construct Wavefunction.
   *       If Wavefunction already exists in the factory (constructed in a previous exec block)
   *       there is no need to build Hamiltonian.
   */
  if (mpi->comm.root() == 0)
  {
    if (hdf_read_restart != std::string(""))
    {
      utils::check(false,"Restart not yet implemented for finite-T calculations");
      /*
      h5::file file(hdf_read_restart,'r');
      h5::group grp(file);
      if (not grp.has_key("AFQMCDriver")) return false;
      h5::group dgrp = grp.open_group("AFQMCDriver");
      
      std::vector<IndexType> Idata(2);
      std::vector<RealType> Rdata(2);

      h5::h5_read(dgrp,"DriverInts",Idata);
      h5::h5_read(dgrp,"DriverReals",Rdata);

      Eshift = Rdata[0];
      block0 = Idata[0];
      step0  = Idata[1];
      restarted = true;
      */
    }
  }
  mpi->comm.broadcast_value(restarted);
  if (restarted)
  {
    app_log(1," Restarted from file. Block={}, step={}",block0,step0);
    app_log(1,"                      Eshift: {}", Eshift);
    mpi->comm.broadcast_value(Eshift);
    mpi->comm.broadcast_value(block0);
    mpi->comm.broadcast_value(step0);
  }

  /*
   * to do:
   *  - add logic for estimators, e.g. whether to evaluate energy, which wfn to use, etc.
   */

  // walker set and type
  auto& wset          = WSetFac.getWalkerSet(mpi, wset_name, rng_wlk);
  WALKER_TYPES walker_type = wset.getWalkerType();
  wset.setTauStep(0); // time-slice initialized to 0
  if (not WfnFac.is_constructed(wfn_name))
  {
    // hamiltonian
    Hamiltonian& ham0 = HamFac.getHamiltonian(mpi, ham_name);

    // build wavefunction
    [[maybe_unused]] auto& wfn0 = WfnFac.getWavefunction(mpi, wfn_name, walker_type, &ham0, nWalkers);
  }

  // wfn builder should not use Hamiltonian pointer now
  auto& wfn0 = WfnFac.getWavefunction(mpi, wfn_name, walker_type, nullptr, nWalkers);

  // propagator
  auto& prop0 = PropFac.getPropagator(mpi, prop_name, wfn0, rng);
  bool hybrid       = prop0.hybrid_propagation();
  // resize walker set
  if (restarted)
  {
    h5::file file(hdf_read_restart,'r');
    restartFromHDF5(wset, nWalkers, file, set_nWalker_target);
    // perform runtime optimization
    wfn0.runtime_optimization(wset);  
    wfn0.Energy(wset);
  }
  else
  {
    auto initial_guess = WfnFac.getInitialGuess_ft(wfn_name);
    wset.resize(nWalkers, initial_guess()); 
    // perform runtime optimization
    wfn0.runtime_optimization(wset);  
    // ntau implicitly set to 0 here    
    wfn0.Energy(wset);
    memory::buffered_array<MEM,ComplexType,1> ovlp0(nWalkers,ComplexType(0.0));
    wset.getProperty(OVLP,ovlp0);
    wfn0.setLogPT0(ovlp0);
    print_initial_energy(wset);
    if (hybrid)
    {
      // Eshift defaults to 0.0 if not provided in input
      //    otherwise, use the value from input with warning
      if (Eshift != 0.0)
      {
        app_warning("user set expert-level parameter, \"initial_Eshift\" : Using user-provided initial Eshift = {}", Eshift);
      }
    } else {
      if (Eshift != 0.0)
      {
        app_log(1, "[Warning] : User set initial Eshift {} with local energy importance. This value is ignored.", Eshift);
      }
      Eshift = real(ComplexType(wset[0].energy()));
    }
  }

  // is this run using importance sampling? 
// MAM: should be asking for importance sampling and not for free_propagation...
  bool free_proj = prop0.free_propagation();
  // if hybrid calculation, set to true
  bool addEnergyEstim = hybrid;

  // estimator setup
  auto estim0 = EstimatorHandler<MEM>(mpi, AFinfo, title, pt_in, wset, WfnFac, wfn0, 
         prop0, walker_type, HamFac, ham_name, dt, addEnergyEstim, !free_proj);

  app_log(1,"\n****************************************************");
  app_log(1,"****************************************************");
  app_log(1,"****************************************************");
  app_log(1,"          Finished Driver initialization.");
  app_log(1,"****************************************************");
  app_log(1,"****************************************************");
  app_log(1,"****************************************************\n");

  FTAFQMCDriver<MEM> driver(mpi, AFinfo, title, m_series, block0, step0, Eshift, pt_in, wfn0, prop0, estim0);

  if (!driver.run(wset))
  {
    app_error(" Problems with FTAFQMCDriver::run().");
    return false;
  }

  if (!driver.clear())
  {
    app_error(" Problems with FTAFQMCDriver::clear().");
    return false;
  }

  return true;
}

template<MEMORY_SPACE MEM>
bool DriverFactory<MEM>::executeCSAFQMCDriver(std::string title, int m_series, ptree pt_in)
{
/*
  // convert user input to verbose input
  ptree pt = interpret_inputs_csafqmc(pt_in);
  app_log(2,"\nDrvFac::executeCSAFQMCDriver input:\n{}\n",io::to_string(pt));

  // total number of systems in correlated sampling  
  int n_systems = pt.get<int>("n_systems");
  int n_groups = gTG.getNumberOfGroups();
  if(n_systems < n_groups)
    utils::check(false,"Error: n_systems < n_groups.");
  if(n_systems%n_groups != 0)
    utils::check(false,"Error: n_systems%n_groups != 0.");
  int ng = gTG.World().rank()/gTG.Global().size();
  auto [ns0, ns1] = FairDivideBoundary(ng,n_systems,n_groups);
  int nsys = ns1-ns0;

  std::string hdf_read_restart;
  bool set_nWalker_target;
  double dt;
  hdf_read_restart = pt.get<std::string>("hdf_read_file");
  set_nWalker_target = pt.get<bool>("set_nwalker_to_target");
  dt = pt.get<double>("timestep");
  int nWalkers = pt.get<int>("n_walkers_per_mpi_task");

  bool restarted = false;
  int step0      = 0;
  int block0     = 0;
  std::vector<double> E0(n_systems,0.0);
  std::vector<double> Eshift(nsys,0.0);

  utils::RandomGenerator_t<>::result_type iseed = ( (pt.get<int>("seed") == 0) ? 
                                           utils::make_seed(gTG.Global()) : 
                                           pt.get<int>("seed"));
  utils::RandomGenerator_t<> rng_wlk(iseed);
  // All systems share the same seed, which is a function of your
  // rank in Global (NOT WORLD!)
  // Each system needs a separate, synchronized RNG. 
  iseed = ( (pt.get<int>("seed") == 0) ? utils::make_seed(gTG.Global()) :
                                         pt.get<int>("seed"));
  std::vector<utils::RandomGenerator_t<MEM>> rngs;
  rngs.reserve(nsys);
  for(int i=0; i<nsys; i++)
    rngs.emplace_back(utils::make_device_rng(iseed));

  app_log(1,"\n****************************************************");
  app_log(1,"****************************************************");
  app_log(1,"****************************************************");
  app_log(1,"          Beginning Driver initialization.");
  app_log(1,"****************************************************");
  app_log(1,"****************************************************");
  app_log(1,"****************************************************\n");

  / *
   * Note: Hamiltonian is only needed to construct Wavefunction.
   *       If Wavefunction already exists in the factory (constructed in a previous exec block)
   *       or it is being initialized from hdf5, there is no need to build Hamiltonian.
   * /
  hdf_archive read(gTG.Global());
  if (gTG.Global().rank() == 0)
  {
    if (hdf_read_restart != std::string(""))
    {
      if (read.open(hdf_read_restart, H5F_ACC_RDONLY))
      {
        // always write driver data and walkers
        if (!read.push("AFQMCDriver", false)) {
	  app_error("Error: Could not find restart information.");
          return false;
	}

        std::vector<IndexType> Idata(2);

        if (!read.readEntry(Idata, "DriverInts"))
          return false;
        if (!read.readEntry(E0, "DriverReals"))
          return false;

        block0 = Idata[0];
        step0  = Idata[1];

        read.pop();
        restarted = true;
        read.close();
      }

      if (!restarted)
      {
        app_log(1," WARNING: Problems restarting simulation. Starting from default settings. ");
      }
    }
  } else {
  }
  gTG.Global().broadcast_value(restarted);
  if (restarted)
  {
    app_log(1," Restarted from file. Block={}, step={}",block0,step0);
    gTG.Global().broadcast_n(E0.data(),n_systems);
    gTG.Global().broadcast_value(block0);
    gTG.Global().broadcast_value(step0);
    for(int i=ns0; i<ns1; i++)
      Eshift[i-ns0] = E0[i];
  }

  std::vector<std::reference_wrapper<AFQMCInfo>> AFinfo_ref;
  std::vector<std::reference_wrapper<WalkerSet>> wset_ref;
  std::vector<std::reference_wrapper<Wavefunction>> wfn_ref;
  std::vector<std::reference_wrapper<Propagator>> prop_ref;
  std::vector<EstimatorHandler> estimators;
  
  AFinfo_ref.reserve(nsys);
  wset_ref.reserve(nsys);
  wfn_ref.reserve(nsys);
  prop_ref.reserve(nsys);
  estimators.reserve(nsys);
  for(int sys=ns0; sys<ns1; sys++) {

    app_log(1," Initializing cs_system_{}. ",sys);

    auto sys_pt = pt.get_child("cs_system_"+std::to_string(sys));
    auto [system,ham_name,wfn_name,wset_name,prop_name] = 
						get_component_ids(sys_pt);

    // !!!! HACK get parameter from verbose input later
    //int nnodes_propg = 1;
    //int nnodes_wfn = 1;
    //RealType cutvn = 1e-6;
    std::string wfn_restart = "";
    //// get factory parameters
    int nnodes_propg = std::max(1, get_parameter<int>(PropFac, prop_name, "nnodes", 1));
    int nnodes_wfn   = std::max(1, get_parameter<int>(WfnFac, wfn_name, "nnodes", 1));
    RealType cutvn   = get_parameter<RealType>(PropFac, prop_name, "cutoff", 1e-6);
    //// Wavefunction and Hamiltonian Operations already stored in restart file.
    //std::string wfn_restart = get_parameter<std::string>(WfnFac, wfn_name, "restart_file", "");

    // setup task groups
    auto& TGprop = TGHandler.getTG(nnodes_propg);
    auto& TGwfn  = TGHandler.getTG(nnodes_wfn);

    if (InfoMap.find(system) == InfoMap.end())
    {
      app_error("ERROR: Undefined system in execute block. ");
      return false;
    }
    auto& AFinfo = InfoMap[system];
    int NMO      = AFinfo.NMO;
    int ndown     = AFinfo.ndown;
    AFinfo_ref.emplace_back(std::ref(AFinfo));

    // walker set and type
    auto& wset          = WSetFac.getWalkerSet(mpi, wset_name, rng_wlk);
    WALKER_TYPES walker_type = wset.getWalkerType();
    wset_ref.emplace_back(std::ref(wset));

    if (not WfnFac.is_constructed(wfn_name) && wfn_restart == "")
    {
      // hamiltonian
      Hamiltonian& ham0 = HamFac.getHamiltonian(mpi, ham_name);

      // build wavefunction
      [[maybe_unused]] Wavefunction& wfn0 = WfnFac.getWavefunction(mpi, wfn_name, walker_type, &ham0, nWalkers);
    }

    // wfn builder should not use Hamiltonian pointer now
    Wavefunction& wfn0 = WfnFac.getWavefunction(mpi, wfn_name, walker_type, nullptr, nWalkers);
    wfn_ref.emplace_back(std::ref(wfn0));

    // propagator
    Propagator& prop0 = PropFac.getPropagator(mpi, prop_name, wfn0, rngs[sys]);
    prop_ref.emplace_back(std::ref(prop0));
    bool hybrid       = prop0.hybrid_propagation();

    // resize walker set
    if (restarted)
    {
      restartFromHDF5(wset, nWalkers, hdf_read_restart, read, set_nWalker_target);
      wfn0.Energy(wset);
    }
    else
    {
      auto initial_guess = WfnFac.getInitialGuess(wfn_name);
      wset.resize(nWalkers, initial_guess);
      wfn0.Energy(wset);
      print_initial_energy(wset);
      if (hybrid)
        Eshift[sys] = 0.0;
      else
        Eshift[sys] = real(ComplexType(wset[0].energy()));
    }

    if (gTG.Global().rank() == 0)
      read.close();

    // is this run using importance sampling? 
    bool free_proj = prop0.free_propagation();
    // if hybrid calculation, set to true
    bool addEnergyEstim = hybrid;

    // estimator setup
    char buf[256];
    snprintf(buf, sizeof(buf), "%s.g%03d", title.c_str(), sys);
    std::string tag(buf);
    estimators.emplace_back(EstimatorHandler(TGHandler, AFinfo, 
		tag, pt_in, wset, 
		WfnFac, wfn0, prop0, walker_type, HamFac, ham_name, dt,
                addEnergyEstim, !free_proj));
  }

  // now that all propagators and hamops are constructed, set rng block size
  int max_nCV=0;
  for(auto& v : wfn_ref) max_nCV = std::max(max_nCV, v.get().local_number_of_cholesky_vectors()); 
  for(auto& v : prop_ref) v.get().set_rng_block_size(max_nCV);

  app_log(1,"\n****************************************************");
  app_log(1,"****************************************************");
  app_log(1,"****************************************************");
  app_log(1,"          Finished Driver initialization.");
  app_log(1,"****************************************************");
  app_log(1,"****************************************************");
  app_log(1,"****************************************************\n");

  gTG.Global().barrier();
/ *
  CSAFQMCDriver driver(gTG.Global(), title, m_series, block0, step0, 
		       std::move(Eshift), pt_in, std::move(AFinfo_ref), 
                       std::move(wfn_ref), std::move(prop_ref), 
                       std::move(estimators));

  if (!driver.run(wset_ref))
  {
    app_error(" Problems with CSAFQMCDriver::run().");
    return false;
  }

  if (!driver.clear())
  {
    app_error(" Problems with CSAFQMCDriver::clear().");
    return false;
  }
* /
*/
  return true;
}

// Instantiate
#define __inst__(M)                                                                  \
template bool DriverFactory<M>::executeDriver(std::string,std::string,int,ptree);    \
template std::string DriverFactory<M>::get_wavefunction_id(ptree);                   \
template std::string DriverFactory<M>::get_system_id(ptree,std::string);             \
template std::tuple<std::string,std::string,std::string,std::string,std::string>     \
  DriverFactory<M>::get_component_ids(ptree);                                        \
template bool DriverFactory<M>::executeAFQMCDriver(std::string,int,ptree);           \
template bool DriverFactory<M>::executeFTAFQMCDriver(std::string,int,ptree);           \
template bool DriverFactory<M>::executeCSAFQMCDriver(std::string,int,ptree);

__inst__(HOST_MEMORY)
#if defined(ENABLE_DEVICE)
__inst__(DEVICE_MEMORY)
#endif

} // namespace afqmc
} // namespace sfqmc
