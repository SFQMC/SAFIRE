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

#undef NDEBUG

#include "catch2/catch_test_macros.hpp"

#include "config.h"
#include "IO/app_loggers.h"
#include "AFQMC/parameters.hpp"
#include "AFQMC/parameter_defaults.hpp"
#include "utilities/Random.hpp"
#include "utilities/Timer.hpp"
#include "test_common.hpp"
#include "utilities/check.hpp"

#include <string>
#include <vector>
#include <complex>
#include <iomanip>

#include "nda/nda.hpp"
#include "nda/tensor.hpp"
#include "nda/h5.hpp"
#include "numerics/sparse/sparse.hpp"

#include "test_utils.hpp"
#include "AFQMC/Utilities/readWfn.h" 

#include "AFQMC/Hamiltonians/HamiltonianFactory.h"
#include "AFQMC/Hamiltonians/Hamiltonian.hpp"
#include "AFQMC/Wavefunctions/WavefunctionFactory.h"
#include "AFQMC/Propagators/PropagatorFactory.h"
#include "AFQMC/Walkers/WalkerSet.hpp"

using std::cerr;
using std::complex;
using std::cout;
using std::endl;
using std::ifstream;
using std::setprecision;
using std::string;

extern std::string UTEST_HAMIL, UTEST_WFN;

namespace sfqmc
{
using namespace afqmc;

template<MEMORY_SPACE MEM>
void propagator_factory_build(std::shared_ptr<utils::mpi_context_t<boost::mpi3::communicator>> mpi,
             std::string hamil_file, std::string wfn_file, bool dense_trial, bool finiteT)
{
  using nda::range;

  int NMO = read_nmo_from_hdf(hamil_file);
  auto[wfn_NMO,nup, ndown] = read_info_from_wfn(wfn_file,"any");
  utils::check(NMO == wfn_NMO, "Error: NMO != wfn_NMO.");
  WALKER_TYPES type         = getWalkerType(wfn_file);
  int nspin                 = type == COLLINEAR ? 2 : 1;
  int npol                  = type == NONCOLLINEAR ? 2 : 1;
  // finite-T imaginary-time slice count (the wfn "nup" field for a finite-T guess)
  int ntau                  = nup;

  HamiltonianFactory HamFac;
  HamFac.push("ham0", HamiltonianParameters{.name = "ham0", .filename = hamil_file, .shift_1body = true});
  Hamiltonian& ham = HamFac.getHamiltonian(mpi, "ham0");

  int nwalk = 11; 
  std::shared_ptr<utils::RandomGenerator_t<HOST_MEMORY>> rng = std::make_shared<utils::RandomGenerator_t<HOST_MEMORY>>();
  std::shared_ptr<utils::RandomGenerator_t<MEM>> rng_dev = std::make_shared<utils::RandomGenerator_t<MEM>>(777);

  const WalkerSetParameters wlk_params{.name = "wset0", .walker_type = type};

  WavefunctionFactory<MEM> WfnFac{};
  WfnFac.push("wfn0", WavefunctionParameters{.name = "wfn0", .filename = wfn_file, .dense_trial = dense_trial});
  auto& wfn = WfnFac.getWavefunction(mpi, "wfn0", type, finiteT, &ham, nwalk);

  auto wset = [&]() {
    if(!finiteT)
    {
      auto const& initial_guess = WfnFac.getInitialGuess("wfn0");
      REQUIRE(int(initial_guess.size()) == nspin);
      REQUIRE(initial_guess[0].shape() == std::array<long,2>{npol*NMO,nup});
      return WalkerSet<MEM>(mpi, wlk_params, rng, type, initial_guess, nwalk);
    }
    else
    {
      auto initial_guess_ft = WfnFac.getInitialGuess_ft("wfn0");
      REQUIRE(initial_guess_ft.shape() == std::array<long,4>{3,nspin,npol*NMO,NMO});
      return WalkerSet<MEM>(mpi, wlk_params, rng, type, initial_guess_ft, nwalk);
    }
  }();

  PropagatorFactory<MEM> PropgFac;
  PropagatorParameters prop_params{.name = "prop0", .denseP2 = true};
  apply_defaults(prop_params, ham.getHamType());
  PropgFac.push("prop0", prop_params);
  auto& prop = PropgFac.getPropagator(mpi, "prop0", wfn, rng_dev);

  std::cout << setprecision(8);
  wfn.Energy(wset);
  {
    ComplexType eav = 0, ov = 0;
    for(int iw = 0; iw < wset.size(); ++iw)
    {
      auto w = wset[iw];
      eav += w.get_property(WEIGHT) * (w.energy());
      ov += w.get_property(WEIGHT);
    }
    app_log(1," Initial Energy: {}", (eav / ov).real());
  }
  double tot_time = 0;
  RealType dt     = 0.01;
  RealType Eshift = std::abs(wset[0].get_property(OVLP));
  if(!finiteT){
    for (int i = 0; i < 10; i++)
    {
      prop.Propagate(wset, Eshift, dt);
      wfn.Energy(wset);
      ComplexType eav = 0, ov = 0;
      for(int iw = 0; iw < wset.size(); ++iw)
      {
        auto w = wset[iw];
        eav += w.get_property(WEIGHT) * (w.energy());
        ov += w.get_property(WEIGHT);
      }
      tot_time += dt;
      app_log(1," -- {}  {}  {}",i,tot_time,(eav / ov).real());
      prop.Orthogonalize(wset);
    }
    for (int i = 0; i < 10; i++)
    {
      prop.Propagate(wset, Eshift, 2 * dt);
      wfn.Energy(wset);
      ComplexType eav = 0, ov = 0;
      for(int iw = 0; iw < wset.size(); ++iw)
      {
        auto w = wset[iw];
        eav += w.get_property(WEIGHT) * (w.energy());
        ov += w.get_property(WEIGHT);
      }
      tot_time += 2 * dt;
      app_log(1," -- {}  {}  {}",i,tot_time,(eav / ov).real());
      prop.Orthogonalize(wset);
    }
  } 
  else {

    dt = 0.099;
    //int ntau_test = 10;
    for(int i = 0; i < ntau-1; i++)
    {
      prop.Propagate(wset, Eshift, dt, i+1);
      wfn.Energy(wset, i+1);
      ComplexType eav = 0, ov = 0;
      for(int iw = 0; iw < wset.size(); ++iw)
      {
        auto w = wset[iw];
        eav += w.get_property(WEIGHT) * (w.energy());
        ov += w.get_property(WEIGHT);
      }
      tot_time += dt;
      app_log(1," -- {}  {}  {}",i,tot_time,(eav / ov).real());
      prop.Orthogonalize(wset);
    }
  }
  app_log(1," setup: {}",timers.setup.total_time);
  if(mpi->comm.root()) timers.print_all();
}

TEST_CASE("propagator_factory: build", "[propagator_factory]")
{
  auto& mpi = utils::make_unit_test_mpi_context();

  using namespace utils;

  run_test_with_files([&]<auto MEM>(std::string hamil_file, std::string wfn_file, WALKER_TYPES, bool finiteT) {
    propagator_factory_build<MEM>(mpi, hamil_file, wfn_file, true, finiteT);
    propagator_factory_build<MEM>(mpi, hamil_file, wfn_file, false, finiteT);
  }, UTEST_HAMIL, UTEST_WFN, TestFiles::RHF | TestFiles::UHF | TestFiles::GHF | TestFiles::NOMSD | TestFiles::FINITE_T | TestFiles::ALL_SYSTEMS);
}


} // namespace sfqmc
