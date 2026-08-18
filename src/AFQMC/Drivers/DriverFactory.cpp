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
#include "IO/banner.hpp"
#include "utilities/check.hpp"

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
  app_log(1,"Local Energy of starting determinant ");
  //app_log(1," <psi_T|H|w_0>/<psi_T|w_0>: ");
  app_log(1,"  - Total energy    : {:f}", wset[0].energy());
  app_log(1,"  - One-body energy : {:f}", wset[0].get_property(E1_));
  app_log(1,"  - Coulomb energy  : {:f}", wset[0].get_property(EJ_));
  app_log(1,"  - Exchange energy : {:f}", wset[0].get_property(EXX_));
}

template<MEMORY_SPACE MEM>
bool DriverFactory<MEM>::executeDriver(DriverType type, std::string title,
				  int m_series, const ExecuteParameters& exec)
{
  switch(type)
  {
    case DriverType::afqmc:
      return executeAFQMCDriver(title, m_series, exec);
    case DriverType::ftafqmc:
      return executeFTAFQMCDriver(title, m_series, exec);
    case DriverType::csafqmc:
      return executeCSAFQMCDriver(title, m_series, exec);
  }
  utils::check(false,"Unknown execute driver.  ");
  return false;
}

template<MEMORY_SPACE MEM>
std::tuple<std::string,std::string,std::string,std::string>
    DriverFactory<MEM>::get_component_ids(const ExecuteParameters& exec)
{
  auto name_of = [](const auto& block, std::string_view key) -> const std::string& {
    utils::check(block.has_value(), " Error: the execute block has no {}. ", key);
    const auto* name = std::get_if<std::string>(&*block);
    utils::check(name != nullptr, " Error: the {} of the execute block was not resolved to a name. "
                 "Did resolve_defaults run? ", key);
    return *name;
  };

  return std::make_tuple(name_of(exec.hamiltonian, "hamiltonian"), name_of(exec.wavefunction, "wavefunction"),
                         name_of(exec.walker_set, "walker_set"), name_of(exec.propagator, "propagator"));
}

template<MEMORY_SPACE MEM>
Wavefunction<MEM>& DriverFactory<MEM>::get_wavefunction(const std::string& wfn_name, const std::string& ham_name,
                                                        WALKER_TYPES walker_type, bool finiteT, int nWalkers)
{
  /*
   * Note: Hamiltonian is only needed to construct Wavefunction.
   *       If Wavefunction already exists in the factory (constructed in a previous exec block)
   *       there is no need to build Hamiltonian.
   */
  if(not WfnFac.is_constructed(wfn_name))
  {
    Hamiltonian& ham = HamFac.getHamiltonian(mpi, ham_name);
    WfnFac.getWavefunction(mpi, wfn_name, walker_type, finiteT, std::addressof(ham), nWalkers);
  }

  // wfn builder should not use Hamiltonian pointer now
  return WfnFac.getWavefunction(mpi, wfn_name, walker_type, finiteT, nullptr, nWalkers);
}

template<MEMORY_SPACE MEM>
bool DriverFactory<MEM>::executeAFQMCDriver(std::string title, int m_series, const ExecuteParameters& exec)
{
  // reset timers
  timers.reset_all();
  auto [ham_name,wfn_name,wset_name,prop_name] = get_component_ids(exec);

  std::string hdf_read_restart;
  bool set_nWalker_target;
  double dt;
  hdf_read_restart = exec.hdf_read_file;
  set_nWalker_target = exec.set_nwalker_to_target;
  dt = exec.timestep;
  int nWalkers = exec.n_walkers_per_mpi_task;

  bool restarted = false;
  int step0      = 0;
  int block0     = 0;
  double Eshift = exec.initial_Eshift;

  utils::SeedType iseed = (exec.seed ? utils::split_seed(*exec.seed, mpi->comm)
                                     : utils::make_seed(mpi->comm));
  std::shared_ptr<utils::RandomGenerator_t<>> rng_wlk = std::make_shared<utils::RandomGenerator_t<>>(iseed);
  iseed = (exec.seed ? utils::split_seed(*exec.seed, mpi->comm) : utils::make_seed(mpi->comm));
  std::shared_ptr<utils::RandomGenerator_t<MEM>> rng = std::make_shared<utils::RandomGenerator_t<MEM>>(iseed);

  app_log(1, banner("Beginning Driver initialization"));

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

  // walker_type is read early from the walker-set input block
  // the WalkerSet is built after the wavefunction
  WALKER_TYPES walker_type = WSetFac.get_walker_type(wset_name);

  bool finiteT = false;
  auto& wfn0 = get_wavefunction(wfn_name, ham_name, walker_type, finiteT, nWalkers);

  // propagator
  auto& prop0 = PropFac.getPropagator(mpi, prop_name, wfn0, rng);
  bool hybrid       = prop0.hybrid_propagation();

  // Build and populate the walker set: from the restart file, or from the wavefunction's initial guess.
  auto& wset = [&]() -> decltype(auto) {
    if(restarted) {
      h5::file file(hdf_read_restart,'r');
      return WSetFac.getWalkerSetFromHDF5(mpi, wset_name, rng_wlk, walker_type, file, nWalkers, set_nWalker_target);
    } else {
      return WSetFac.getWalkerSet(mpi, wset_name, rng_wlk, walker_type, WfnFac.getInitialGuess(wfn_name), nWalkers);
    }
  }();

  // perform runtime optimization
  wfn0.runtime_optimization(wset);
  wfn0.Energy(wset);

  if (not restarted)
  {
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
  auto estim0 = EstimatorHandler<MEM>(mpi, title, exec, wset, WfnFac, wfn0,
         prop0, HamFac, dt, addEnergyEstim, !free_proj);

  app_log(1, banner("Finished Driver initialization"));

  AFQMCDriver<MEM> driver(mpi, title, m_series, block0, step0, Eshift, exec, wfn0, prop0, estim0);

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
bool DriverFactory<MEM>::executeFTAFQMCDriver(std::string title, int m_series, const ExecuteParameters& exec)
{
  // reset timers
  timers.reset_all();
  auto [ham_name,wfn_name,wset_name,prop_name] = get_component_ids(exec);

  std::string hdf_read_restart;
  // read but unused: finite-T restart is not yet supported, so the walker set is
  // always built fresh from the wavefunction guess (see below).
  [[maybe_unused]] bool set_nWalker_target;
  hdf_read_restart = exec.hdf_read_file;
  set_nWalker_target = exec.set_nwalker_to_target;
  int nWalkers = exec.n_walkers_per_mpi_task;

  bool restarted = false;
  int step0      = 0;
  int block0     = 0;
  double Eshift = exec.initial_Eshift;

  utils::SeedType iseed = (exec.seed ? utils::split_seed(*exec.seed, mpi->comm)
                                     : utils::make_seed(mpi->comm));
  std::shared_ptr<utils::RandomGenerator_t<>> rng_wlk = std::make_shared<utils::RandomGenerator_t<>>(iseed);
  iseed = (exec.seed ? utils::split_seed(*exec.seed, mpi->comm) : utils::make_seed(mpi->comm));
  std::shared_ptr<utils::RandomGenerator_t<MEM>> rng = std::make_shared<utils::RandomGenerator_t<MEM>>(iseed);

  app_log(1, banner("Beginning Driver initialization"));

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
    app_log(1,"Restarted from file. Block={}, step={}",block0,step0);
    app_log(1,"                     Eshift: {}", Eshift);
    mpi->comm.broadcast_value(Eshift);
    mpi->comm.broadcast_value(block0);
    mpi->comm.broadcast_value(step0);
  }

  /*
   * to do:
   *  - add logic for estimators, e.g. whether to evaluate energy, which wfn to use, etc.
   */

  // walker_type is read early from the walker-set input block; the WalkerSet is
  // built after the wavefunction. The FT driver forces finite_temperature = true.
  WALKER_TYPES walker_type = WSetFac.get_walker_type(wset_name);
  bool finiteT = true;
  auto& wfn0 = get_wavefunction(wfn_name, ham_name, walker_type, finiteT, nWalkers);

  // propagator
  auto& prop0 = PropFac.getPropagator(mpi, prop_name, wfn0, rng);
  bool hybrid       = prop0.hybrid_propagation();

  // Build and populate the finite-temperature walker set from the wavefunction's
  // rank-4 UDV initial guess. FT restart is not yet supported.
  utils::check(not restarted, "Restart not yet implemented for finite-T calculations");
  auto& wset = WSetFac.getWalkerSetFT(mpi, wset_name, rng_wlk, walker_type,
                                      WfnFac.getInitialGuess_ft(wfn_name), nWalkers);
  wset.setTauStep(0); // time-slice initialized to 0

  // perform runtime optimization; ntau implicitly set to 0 here
  wfn0.runtime_optimization(wset);
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

  // is this run using importance sampling? 
// MAM: should be asking for importance sampling and not for free_propagation...
  bool free_proj = prop0.free_propagation();
  // if hybrid calculation, set to true
  bool addEnergyEstim = hybrid;

  // estimator setup
  auto estim0 = EstimatorHandler<MEM>(mpi, title, exec, wset, WfnFac, wfn0,
         prop0, HamFac, exec.timestep, addEnergyEstim, !free_proj);

  app_log(1, banner("Finished Driver initialization"));

  FTAFQMCDriver<MEM> driver(mpi, title, m_series, block0, step0, Eshift, exec, wfn0, prop0, estim0);

  if (!driver.run(wset))
  {
    app_error("Problems with FTAFQMCDriver::run().");
    return false;
  }

  if (!driver.clear())
  {
    app_error("Problems with FTAFQMCDriver::clear().");
    return false;
  }

  return true;
}

template<MEMORY_SPACE MEM>
bool DriverFactory<MEM>::executeCSAFQMCDriver([[maybe_unused]] std::string title,
                                             [[maybe_unused]] int m_series,
                                             [[maybe_unused]] const ExecuteParameters& exec)
{
  utils::check(false, "The csafqmc driver is not implemented.");
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

  utils::SeedType iseed = ( (pt.get<int>("seed") == 0) ? 
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

  app_log(1, banner("Beginning Driver initialization"));

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
    bool finiteT = false; // if this driver is re-implemented, it is ground-state only
    wset_ref.emplace_back(std::ref(wset));

    if (not WfnFac.is_constructed(wfn_name) && wfn_restart == "")
    {
      // hamiltonian
      Hamiltonian& ham0 = HamFac.getHamiltonian(mpi, ham_name);

      // build wavefunction
      [[maybe_unused]] Wavefunction& wfn0 = WfnFac.getWavefunction(mpi, wfn_name, walker_type, finiteT, &ham0, nWalkers);
    }

    // wfn builder should not use Hamiltonian pointer now
    Wavefunction& wfn0 = WfnFac.getWavefunction(mpi, wfn_name, walker_type, finiteT, nullptr, nWalkers);
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
    estimators.emplace_back(EstimatorHandler(TGHandler, AFinfo.nup, AFinfo.ndown,
		tag, pt_in, wset,
		WfnFac, wfn0, prop0, HamFac, ham_name, dt,
                addEnergyEstim, !free_proj));
  }

  // now that all propagators and hamops are constructed, set rng block size
  int max_nCV=0;
  for(auto& v : wfn_ref) max_nCV = std::max(max_nCV, v.get().local_number_of_cholesky_vectors()); 
  for(auto& v : prop_ref) v.get().set_rng_block_size(max_nCV);

  app_log(1, banner("Finished Driver initialization"));

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
#define __inst__(M)                                                                            \
template bool DriverFactory<M>::executeDriver(DriverType,std::string,int,const ExecuteParameters&);  \
template std::tuple<std::string,std::string,std::string,std::string>                           \
  DriverFactory<M>::get_component_ids(const ExecuteParameters&);                                    \
template bool DriverFactory<M>::executeAFQMCDriver(std::string,int,const ExecuteParameters&);       \
template bool DriverFactory<M>::executeFTAFQMCDriver(std::string,int,const ExecuteParameters&);     \
template bool DriverFactory<M>::executeCSAFQMCDriver(std::string,int,const ExecuteParameters&);

__inst__(HOST_MEMORY)
#if defined(ENABLE_DEVICE)
__inst__(DEVICE_MEMORY)
#endif

} // namespace afqmc
} // namespace sfqmc
