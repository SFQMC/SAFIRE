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
#include "IO/AppAbort.hpp"

#include "AFQMC/parameters.hpp"
#include "utilities/Random.hpp"
#include "IO/app_loggers.h"

#include "nda/nda.hpp"
#include "nda/tensor.hpp"
#include "nda/h5.hpp"

#include <string>
#include <vector>
#include <complex>
#include <iomanip>
#include <random>

#include "utilities/Timer.hpp"
#include "test_common.hpp"
#include "utilities/check.hpp"
#include "test_utils.hpp"
#include "AFQMC/Utilities/readWfn.h"

#include "AFQMC/Hamiltonians/HamiltonianFactory.h"
#include "AFQMC/Hamiltonians/Hamiltonian.hpp"
#include "AFQMC/Wavefunctions/WavefunctionFactory.h"
#include "AFQMC/Walkers/WalkerSet.hpp"

#include "numerics/sparse/sparse.hpp"

using std::complex;
using std::ifstream;
using std::string;

extern std::string UTEST_HAMIL, UTEST_WFN;
extern bool WRITE_REFERENCE;

namespace sfqmc
{
using namespace afqmc;

template<MEMORY_SPACE MEM>
void wfn_factory_sdet(std::shared_ptr<utils::mpi_context_t<boost::mpi3::communicator>> mpi,
             std::string hamil_file, std::string wfn_file, WALKER_TYPES type,
             bool finiteT, bool dense_trial, bool write_reference)
{
  using nda::range;
  auto all = range::all;

  // First strip path of filename.
  std::string base_name = wfn_file.substr(wfn_file.find_last_of("\\/") + 1);
  // Remove file extension.
  std::string test_wfn = base_name.substr(0, base_name.find_last_of("."));
  test_wfn = test_wfn.substr(test_wfn.find('_') + 1);

  auto reference_data = read_test_results_from_hdf<ComplexType>(hamil_file, test_wfn);
  auto [NMO,nup,ndown] = read_info_from_wfn(wfn_file, "any");
  utils::check(NMO == reference_data.NMO, "Incompatible NMO.");

  // 'type' is the *target* walker type. The wavefunction file has its own native type,
  // which the factory may convert to any compatible target.
  WALKER_TYPES from    = afqmc::getWalkerType(wfn_file, "any");
  bool native          = (type == from);
  // For now, only do reference testing on the native-type run. Refine to full combinations later.
  write_reference      = write_reference && native;
  bool compare         = native && reference_data.available && !write_reference;

  // Broadcast the electron counts from the native type to the target walker type,
  // mirroring broadcast_number_of_electrons() in the WavefunctionFactory.
  if(type == NONCOLLINEAR) {
    nup   = nup + ndown;
    ndown = 0;
  }

  int nspin            = (type == COLLINEAR) ? 2 : 1;
  int npol             = (type == NONCOLLINEAR) ? 2 : 1;
  int nel              = (type == COLLINEAR) ? nup+ndown : nup;
  double dt(0.01);

  app_log(1, "wfn_factory_sdet: native type {} -> walker type {} (dense_trial={})",
          walkerTypeToString(from), walkerTypeToString(type), dense_trial);

  if(finiteT){
    nup = NMO;
    ndown = NMO;
  }

  HamiltonianFactory HamFac;
  HamFac.push("ham0", HamiltonianParameters{.name = "ham0", .filename = hamil_file});
  Hamiltonian& ham = HamFac.getHamiltonian(mpi, "ham0");

  int nwalk = 11; // choose prime number to force non-trivial splits in shared routines
  std::shared_ptr<utils::RandomGenerator_t<>> rng = std::make_shared<utils::RandomGenerator_t<>>();

  const WalkerSetParameters wlk_params{.name = "wset0", .walker_type = type};

  WavefunctionFactory<MEM> WfnFac{};
  WfnFac.push("wfn0", WavefunctionParameters{.name = "wfn0", .filename = wfn_file, .dense_trial = dense_trial});
  auto& wfn = WfnFac.getWavefunction(mpi, "wfn0", type, finiteT, &ham, nwalk);

  //nwalk=nw;
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

  // Perturb the initial guess by a deterministic non-trivial sequence.
  {
    std::array nels = {nup, ndown};
    for (int spin = 0; spin < nspin; spin++) {
      long nuv = (finiteT ? 2 : 1);
      nda::array<ComplexType, 1> p_h(nuv * nwalk * npol * NMO * nels[spin]);
      for (long k = 0; k < p_h.size(); ++k) {
        double v = 0.1 * (k + 1);
        p_h[k] = {std::cos(v), std::sin(v * v)};
      }
      if (finiteT) {
        memory::array<MEM, ComplexType, 4> p(reshape(p_h, 2, nwalk, npol * NMO, nels[spin]));
        auto UM = wset.UMatrices(static_cast<SpinTypes>(spin));
        auto VM = wset.VMatrices(static_cast<SpinTypes>(spin));
        auto DM = wset.DMatrices(static_cast<SpinTypes>(spin));
        nda::tensor::add(1, p(0,nda::ellipsis{}), 1, UM);
        nda::tensor::add(1, p(1,nda::ellipsis{}), 1, VM);
        nda::tensor::add(1, p(0,all, 0, all), 1, DM);
      } else {
        memory::array<MEM, ComplexType, 3> p(reshape(p_h, nwalk, npol * NMO, nels[spin]));
        auto SM = wset.SlaterMatrices(static_cast<SpinTypes>(spin));
        if(SM.size() > 0) {
          SM() = p();
        }
      }
    }
  }

  // Overlap
  wfn.Log_Overlap(wset);

  // optimize HOps evaluation
  wfn.runtime_optimization(wset);

  wfn.Energy(wset);

  nda::array<ComplexType, 1> e1_w(nwalk), ej_w(nwalk), exx_w(nwalk);
  wset.getProperty(E1_,  e1_w);
  wset.getProperty(EJ_,  ej_w);
  wset.getProperty(EXX_, exx_w);

  if (!write_reference)
  {
    if(compare) {
      CHECK_THAT(e1_w, utils::Approx(reference_data.E1));
      CHECK_THAT(ej_w, utils::Approx(reference_data.EJ));
      CHECK_THAT(exx_w, utils::Approx(reference_data.EXX));
    }
  }
  else
  {
    reference_data.E1 = e1_w;
    reference_data.EJ = ej_w;
    reference_data.EXX = exx_w;
    // app_log(1," E0+E1: {}", e1_w);
    // app_log(1," EJ: {}", ej_w);
    // app_log(1," EXX: {}", exx_w);
  }
  
  // must initialize discrete propagators for lattice models before calling vMF, vbias, etc.
  // technically, only for discrete propagators, but we don't access to that info here.
  if (wfn.getHamType() == ModelHamiltonian) { 
      const long ncv = wfn.number_of_cholesky_vectors();
      memory::array<MEM,ComplexType, 1> vMF_discrete(ncv, ComplexType(0.0, 0.0));
      memory::host_array<ComplexType, 1> nMF(2 * NMO, ComplexType(0.0, 0.0));
      wfn.update_potentials(dt, nMF, vMF_discrete, false);
  }

  // vMF
  {
    memory::array<MEM,ComplexType,1> v(wfn.number_of_cholesky_vectors());
    wfn.vMF(v,dt);
  }

  // G_MF
  {
    auto gMF = wfn.G_MF();
    ComplexType trG = 0;
    for(int spin = 0; spin < nspin; spin++) {
      auto gMF_spin = gMF()(spin,all,all);
      trG += nda::sum(nda::diagonal(gMF_spin)); 
      CHECK_THAT(gMF_spin, utils::Approx(nda::transpose(gMF_spin)));
    }
    if(!finiteT) {
      CHECK_THAT(trG.real(), utils::Approx(nel));
    }
  }

  // update_potentials with natural_shift=true (mirrors production flow)
  {
    nda::array<ComplexType,1> nMF_natural(2*NMO, ComplexType(1.0));
    memory::array<MEM,ComplexType,1> vMF_natural(wfn.number_of_cholesky_vectors());
    wfn.update_potentials(dt, nMF_natural, vMF_natural, true);
  }

  memory::array<MEM,ComplexType,2> X(nwalk,wfn.number_of_cholesky_vectors());
  wfn.vbias(wset, X, dt);
  //std::cout<<"X = "<<X()<<std::endl;
  {
    auto X_h = nda::to_host(X);
    if (!write_reference) {
      if(compare) {
        CHECK_THAT(X_h, utils::Approx(reference_data.vbias));
      }
    } else {
      reference_data.vbias = X_h;
    }
  }

  // One-body propagator matrix shape check
  {
    auto X_h = nda::to_host(X);
    nda::array<ComplexType,1> X_h_real = nda::real(X_h(0,all)); // cannot use finite imaginary part for vMF
    auto h1 = wfn.getOneBodyPropagatorMatrix(dt, X_h_real);
    REQUIRE( h1.shape() == std::array<long,3>{nspin,npol*NMO,npol*NMO} );
  }

  // Dense vHS shape + Vsum value check
  {
    auto[vHS_nspin, vHS_npol] = wfn.vHS_dims();
    auto vHS_dense = wfn.vHS(X, dt);
    REQUIRE( vHS_dense.shape() == std::array<long,4>{vHS_nspin,nwalk,vHS_npol*NMO,NMO} );
    auto vHS_h = nda::to_host(vHS_dense);

    if (!write_reference)
    {
      if(compare) {
        CHECK_THAT(vHS_h, utils::Approx(reference_data.VHS));
      }
    }
    else
    {
      reference_data.VHS = vHS_h;
    }
  }

  // Sparse vHS shape + Vsum value check (ModelHamiltonian only)
  if (wfn.getHamType() == ModelHamiltonian)
  {
    auto vHS_sp = wfn.vHS_sparse(X, dt);
    utils::check(vHS_sp.extent(0) == nspin, "Size mismatch");
    utils::check((vHS_sp(0).shape() == std::array<long,2>{nwalk*npol*NMO,nwalk*npol*NMO}) and
                 (vHS_sp(nspin-1).shape() == std::array<long,2>{nwalk*npol*NMO,nwalk*npol*NMO}),
                 "Size mismatch");
    if (compare) {
      auto vHS_sp_dense = math::sparse::to_array<'N'>(vHS_sp(0));
      auto[vHS_nspin, vHS_npol] = wfn.vHS_dims();
      CHECK_THAT(vHS_sp_dense(range(vHS_npol*NMO), range(NMO)), utils::Approx(reference_data.VHS(0,0,nda::ellipsis{})));
    }
  }

  if(write_reference) {
    write_test_results_to_hdf(hamil_file, test_wfn, reference_data);
  }      
}

TEST_CASE("wfn_factory: sdet", "[wfn_factory]")
{
  auto& mpi = utils::make_unit_test_mpi_context();

  app_log(0,"WavefunctionFactory unit testing.");

  using namespace utils;

  bool write_reference = WRITE_REFERENCE;

  run_test_with_files([&]<auto MEM>(std::string hamil_file, std::string wfn_file, WALKER_TYPES, bool finiteT) {
    WALKER_TYPES from = afqmc::getWalkerType(wfn_file, "any");
    if(finiteT) {
      // finite-T wavefunctions are tested only at their native (base) walker type.
      wfn_factory_sdet<MEM>(mpi, hamil_file, wfn_file, from, true, true,  write_reference && MEM == HOST_MEMORY);
      wfn_factory_sdet<MEM>(mpi, hamil_file, wfn_file, from, true, false, false);
    } else {
      // Test the wfn's native walker type plus every walker type it can be converted to.
      for(auto to : {CLOSED, COLLINEAR, NONCOLLINEAR}) {
        if(!walkerTypeIsConvertible(from, to)) continue;
        wfn_factory_sdet<MEM>(mpi, hamil_file, wfn_file, to, false, true,  write_reference && MEM == HOST_MEMORY);
        wfn_factory_sdet<MEM>(mpi, hamil_file, wfn_file, to, false, false, false);
      }
    }
  }, UTEST_HAMIL, UTEST_WFN, TestFiles::RHF | TestFiles::UHF | TestFiles::GHF | TestFiles::NOMSD | TestFiles::FINITE_T | TestFiles::ALL_SYSTEMS);
  
}


} // namespace sfqmc
