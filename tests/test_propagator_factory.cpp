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

#include "AFQMC/Hamiltonians/Hamiltonian.hpp"
#include "AFQMC/Wavefunctions/Wavefunction.hpp"
#include "AFQMC/Propagators/AFQMCBasePropagator.h"
#include "AFQMC/Propagators/Propagator.hpp"
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

  Hamiltonian ham = Hamiltonian::from_params(
      mpi, HamiltonianParameters{.name = "ham0", .filename = hamil_file, .shift_1body = true});

  int nwalk = 11;
  std::shared_ptr<utils::RandomGenerator_t<HOST_MEMORY>> rng = std::make_shared<utils::RandomGenerator_t<HOST_MEMORY>>();
  std::shared_ptr<utils::RandomGenerator_t<MEM>> rng_dev = std::make_shared<utils::RandomGenerator_t<MEM>>(777);

  const WalkerSetParameters wlk_params{.name = "wset0", .walker_type = type};

  auto wfn = Wavefunction<MEM>::from_params(
      mpi, WavefunctionParameters{.name = "wfn0", .filename = wfn_file, .dense_trial = dense_trial}, type, finiteT,
      ham, nwalk);

  auto const& guess = wfn.initial_guess();
  if(!finiteT)
  {
    REQUIRE(int(guess.slater().size()) == nspin);
    REQUIRE(guess.slater()[0].shape() == std::array<long,2>{npol*NMO,nup});
  }
  else
  {
    REQUIRE(guess.udv().shape() == std::array<long,4>{3,nspin,npol*NMO,NMO});
  }
  auto wset = WalkerSet<MEM>(mpi, rng, wlk_params, guess, nwalk);

  RealType dt = finiteT ? 0.099 : 0.01;

  PropagatorParameters prop_params{.name = "prop0", .denseP2 = true};
  apply_defaults(prop_params, ham.getHamType());
  Propagator<MEM> prop{AFQMCBasePropagator<MEM>(prop_params, mpi, wfn, rng_dev, dt)};

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
  RealType Eshift = std::abs(wset[0].get_property(OVLP));
  if(!finiteT){
    for (int i = 0; i < 10; i++)
    {
      prop.Propagate(wset, Eshift);
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
    // a propagator is built for a single timestep, so the second block needs its own
    Propagator<MEM> prop2{AFQMCBasePropagator<MEM>(prop_params, mpi, wfn, rng_dev, 2 * dt)};
    for (int i = 0; i < 10; i++)
    {
      prop2.Propagate(wset, Eshift);
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

    //int ntau_test = 10;
    for(int i = 0; i < ntau-1; i++)
    {
      prop.Propagate(wset, Eshift, i+1);
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

/*
 * Free projection drops the constraint and keeps the phase of every step in the walker weight
 * instead of projecting it out, so the weights turn complex as soon as the propagation starts.
 */
template<MEMORY_SPACE MEM>
void propagator_free_projection(std::shared_ptr<utils::mpi_context_t<boost::mpi3::communicator>> mpi,
             std::string hamil_file, std::string wfn_file)
{
  WALKER_TYPES type = getWalkerType(wfn_file);
  int nwalk = 11;
  RealType dt = 0.01;

  Hamiltonian ham = Hamiltonian::from_params(
      mpi, HamiltonianParameters{.name = "ham0", .filename = hamil_file, .shift_1body = true});

  std::shared_ptr<utils::RandomGenerator_t<HOST_MEMORY>> rng = std::make_shared<utils::RandomGenerator_t<HOST_MEMORY>>();
  std::shared_ptr<utils::RandomGenerator_t<MEM>> rng_dev = std::make_shared<utils::RandomGenerator_t<MEM>>(777);

  auto wfn = Wavefunction<MEM>::from_params(
      mpi, WavefunctionParameters{.name = "wfn0", .filename = wfn_file, .dense_trial = true}, type, false,
      ham, nwalk);
  auto wset = WalkerSet<MEM>(mpi, rng, WalkerSetParameters{.name = "wset0", .walker_type = type},
                             wfn.initial_guess(), nwalk);

  PropagatorParameters prop_params{.name = "prop0", .free_projection = true, .denseP2 = true};
  apply_defaults(prop_params, ham.getHamType());
  Propagator<MEM> prop{AFQMCBasePropagator<MEM>(prop_params, mpi, wfn, rng_dev, dt)};

  RealType Eshift = std::abs(wset[0].get_property(OVLP));
  for(int i = 0; i < 5; i++) {
    prop.Propagate(wset, Eshift);
    prop.Orthogonalize(wset);
  }

  RealType max_relative_imag = 0.0;
  for(int iw = 0; iw < wset.size(); ++iw) {
    ComplexType weight = wset[iw].get_property(WEIGHT);
    REQUIRE(std::isfinite(weight.real()));
    REQUIRE(std::isfinite(weight.imag()));
    RealType magnitude = std::abs(weight);
    if(magnitude > 0.0) {
      max_relative_imag = std::max(max_relative_imag, std::abs(weight.imag()) / magnitude);
    }
  }
  REQUIRE(max_relative_imag > 1e-8);

  // free projection reconstructs the local energy from the overlap, so it cannot run the
  // local energy formalism
  PropagatorParameters bad_params{.name = "prop1", .hybrid = false, .free_projection = true, .denseP2 = true};
  apply_defaults(bad_params, ham.getHamType());
  REQUIRE_THROWS(AFQMCBasePropagator<MEM>(bad_params, mpi, wfn, rng_dev, dt));
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

TEST_CASE("propagator_factory: free projection", "[propagator_factory]")
{
  auto& mpi = utils::make_unit_test_mpi_context();

  using namespace utils;

  run_test_with_files([&]<auto MEM>(std::string hamil_file, std::string wfn_file, WALKER_TYPES, bool) {
    propagator_free_projection<MEM>(mpi, hamil_file, wfn_file);
  }, UTEST_HAMIL, UTEST_WFN, TestFiles::RHF | TestFiles::NOMSD | TestFiles::MOLECULES);
}


} // namespace sfqmc
