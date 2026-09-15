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

#include "config.h"
#include "IO/banner.hpp"
#include "utilities/check.hpp"


#include "utilities/Random.hpp"
#include "AFQMC/Drivers/DriverFactory.h"
#include "AFQMC/Drivers/AFQMCDriver.h"
#include "AFQMC/Drivers/FTAFQMCDriver.h"

#include "AFQMC/Walkers/WalkerSetFactory.hpp"
#include "AFQMC/Hamiltonians/HamiltonianFactory.h"
#include "AFQMC/Wavefunctions/WavefunctionFactory.h"
#include "AFQMC/Propagators/PropagatorFactory.h"

#include "AFQMC/Hamiltonians/Hamiltonian.hpp"
#include "AFQMC/Wavefunctions/Wavefunction.hpp"
#include "AFQMC/Propagators/Propagator.hpp"

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
  int nWalkers = exec.n_walkers_per_mpi_task;

  bool restarted = false;
  int step0      = 0;
  std::optional<double> Eshift = exec.initial_Eshift;

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
      step0  = Idata[1];
      restarted = true;
    }
  }
  mpi->comm.broadcast_value(restarted);
  if (restarted)
  {
    app_log(1," Restarted from file. step={}",step0);
    app_log(1,"                      Eshift: {}", Eshift.value());
    mpi->comm.broadcast_value(Eshift.value());
    mpi->comm.broadcast_value(step0);
  }

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
      return WSetFac.getWalkerSetFromHDF5(mpi, wset_name, rng_wlk, walker_type, file, nWalkers, false);
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
      if (Eshift)
      {
        app_warning("user set expert-level parameter, \"initial_Eshift\" : Using user-provided initial Eshift = {}", Eshift.value());
      }
    } else {
      if (Eshift)
      {
        app_log(1, "[Warning] : User set initial Eshift {} with local energy importance. This value is ignored.", Eshift.value());
      }
      Eshift = real(ComplexType(wset[0].energy()));
    }
  }

  if(!Eshift) {
    Eshift = 0; // TODO fill with energy regardless
  }

  // estimator setup
  Estimators<MEM> estim0{mpi, m_series, exec, wset, WfnFac, wfn0, prop0, HamFac};

  if(!Eshift) {
    Eshift = 0; // TODO: use sensible energy
  }
  
  AFQMCDriver<MEM> driver(mpi, title, step0, Eshift.value(), exec, wfn0, prop0, estim0);

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
  int nWalkers = exec.n_walkers_per_mpi_task;

  bool restarted = false;
  int step0      = 0;
  std::optional<double> Eshift = exec.initial_Eshift;

  utils::SeedType iseed = (exec.seed ? utils::split_seed(*exec.seed, mpi->comm)
                                     : utils::make_seed(mpi->comm));
  std::shared_ptr<utils::RandomGenerator_t<>> rng_wlk = std::make_shared<utils::RandomGenerator_t<>>(iseed);
  iseed = (exec.seed ? utils::split_seed(*exec.seed, mpi->comm) : utils::make_seed(mpi->comm));
  std::shared_ptr<utils::RandomGenerator_t<MEM>> rng = std::make_shared<utils::RandomGenerator_t<MEM>>(iseed);

  app_log(1, banner("Beginning Driver initialization"));

  mpi->comm.broadcast_value(restarted);

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
    if (Eshift)
    {
      app_warning("user set expert-level parameter, \"initial_Eshift\" : Using user-provided initial Eshift = {}", Eshift.value());
    } else {
      Eshift = 0.0;
    }
  } else {
    if (Eshift)
    {
      app_log(1, "[Warning] : User set initial Eshift {} with local energy importance. This value is ignored.", Eshift.value());
    }
    Eshift = real(ComplexType(wset[0].energy()));
  }

  // estimator setup
  Estimators<MEM> estim0{mpi, m_series, exec, wset, WfnFac, wfn0, prop0, HamFac};

  FTAFQMCDriver<MEM> driver(mpi, title, step0, Eshift.value(), exec, wfn0, prop0, estim0);

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

// Instantiate
#define __inst__(M)                                                                            \
template bool DriverFactory<M>::executeDriver(DriverType,std::string,int,const ExecuteParameters&);  \
template std::tuple<std::string,std::string,std::string,std::string>                           \
  DriverFactory<M>::get_component_ids(const ExecuteParameters&);                                    \
template bool DriverFactory<M>::executeAFQMCDriver(std::string,int,const ExecuteParameters&);       \
template bool DriverFactory<M>::executeFTAFQMCDriver(std::string,int,const ExecuteParameters&);

__inst__(HOST_MEMORY)
#if defined(ENABLE_DEVICE)
__inst__(DEVICE_MEMORY)
#endif

} // namespace afqmc
} // namespace sfqmc
