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

#include "IO/ptree/ptree_utilities.hpp"
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
             std::string hamil_file, std::string wfn_file, bool dense_trial, bool write_reference)
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

  WALKER_TYPES type    = afqmc::getWalkerType(wfn_file, "any");
  int nspin            = (type == COLLINEAR or type == COLLINEAR_FT) ? 2 : 1;
  int npol             = (type == NONCOLLINEAR or type == NONCOLLINEAR_FT) ? 2 : 1;
  int nel              = (type == COLLINEAR or type == COLLINEAR_FT) ? nup+ndown : nup;  
  double dt(0.01);

  int ntau(0);
  if(type == COLLINEAR_FT or type == NONCOLLINEAR_FT){
    ntau = nup;
    nup = NMO;
    ndown = NMO;
  }

  std::map<std::string, AFQMCInfo> InfoMap;
  InfoMap.insert(std::pair<std::string, AFQMCInfo>("info0", AFQMCInfo{"info0", NMO, nup, ndown, ntau}));

  ptree ham_pt;
  ham_pt.put("name","ham0");
  ham_pt.put("system","info0");
  ham_pt.put("filename",hamil_file);

  HamiltonianFactory HamFac(InfoMap);
  HamFac.push("ham0", ham_pt); 
  Hamiltonian& ham = HamFac.getHamiltonian(mpi, "ham0");

  int nwalk = 11; // choose prime number to force non-trivial splits in shared routines
  std::shared_ptr<utils::RandomGenerator_t<>> rng = std::make_shared<utils::RandomGenerator_t<>>();

  ptree wlk_pt;
  wlk_pt.put("name","wset0");
  wlk_pt.put("walker_type", walkerTypeToString(type));

  ptree wfn_pt;
  wfn_pt.put("name","wfn0");
  wfn_pt.put("system","info0");
  wfn_pt.put("filename",wfn_file);
  wfn_pt.put("dense_trial",dense_trial);

  WavefunctionFactory<MEM> WfnFac(InfoMap);
  WfnFac.push("wfn0", wfn_pt);
  auto& wfn = WfnFac.getWavefunction(mpi, "wfn0", type, &ham, nwalk);

  //nwalk=nw;
  auto wset = make_WalkerSet<MEM>(mpi, wlk_pt, InfoMap["info0"], rng);

  if(type != COLLINEAR_FT and type != NONCOLLINEAR_FT)
  {
    auto initial_guess = WfnFac.getInitialGuess("wfn0"); 
    REQUIRE(initial_guess.shape() == std::array<long,3>{nspin,npol*NMO,nup});

    wset.resize(nwalk, initial_guess);
  }
  else
  {
    auto initial_guess_ft = WfnFac.getInitialGuess_ft("wfn0"); 
    REQUIRE(initial_guess_ft.shape() == std::array<long,4>{3,nspin,npol*NMO,NMO});

    wset.resize(nwalk, initial_guess_ft);
  }

  // Perturb the initial guess by a deterministic non-trivial sequence.
  {
    std::array nels = {nup, ndown};
    bool ft = (type == COLLINEAR_FT or type == NONCOLLINEAR_FT);
    for (int spin = 0; spin < nspin; spin++) {
      long nuv = (ft ? 2 : 1);
      nda::array<ComplexType, 1> p_h(nuv * nwalk * npol * NMO * nels[spin]);
      for (long k = 0; k < p_h.size(); ++k) {
        double v = 0.1 * (k + 1);
        p_h[k] = {std::cos(v), std::sin(v * v)};
      }
      if (ft) {
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
        nda::tensor::add(p, "ijk", SM, "ijk");
      }
    }
  }

  // Overlap
  //if(type != COLLINEAR_FT and type != NONCOLLINEAR_FT)
  wfn.Log_Overlap(wset);

  Watch Time;
  Time.reset();

  // optimize HOps evaluation
  wfn.runtime_optimization(wset);

  wfn.Energy(wset);

  nda::array<ComplexType, 1> e1_w(nwalk), ej_w(nwalk), exx_w(nwalk);
  wset.getProperty(E1_,  e1_w);
  wset.getProperty(EJ_,  ej_w);
  wset.getProperty(EXX_, exx_w);

  if (!write_reference)
  {
    if(reference_data.available) {
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
    if(type != COLLINEAR_FT && type != NONCOLLINEAR_FT) {
      CHECK_THAT(trG.real(), utils::Approx(nel));
    }
  }

  // update_potentials with natural_shift=true (mirrors production flow)
  {
    nda::array<ComplexType,1> nMF_natural(2*NMO, ComplexType(1.0));
    memory::array<MEM,ComplexType,1> vMF_natural(wfn.number_of_cholesky_vectors());
    wfn.update_potentials(dt, nMF_natural, vMF_natural, true);
  }

  Time.reset();
  memory::array<MEM,ComplexType,2> X(nwalk,wfn.number_of_cholesky_vectors());
  wfn.vbias(wset, X, dt);
  //std::cout<<"X = "<<X()<<std::endl;
  {
    auto X_h = nda::to_host(X);
    if (!write_reference) {
      if(reference_data.available) {
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
      if(reference_data.available) {
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
    if (!write_reference && reference_data.available) {
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

  run_test_with_files([&]<auto MEM>(std::string hamil_file, std::string wfn_file, WALKER_TYPES) {
    wfn_factory_sdet<MEM>(mpi, hamil_file, wfn_file, true, write_reference && MEM == HOST_MEMORY);
    wfn_factory_sdet<MEM>(mpi, hamil_file, wfn_file, false, false);
  }, UTEST_HAMIL, UTEST_WFN, TestFiles::RHF | TestFiles::UHF | TestFiles::GHF | TestFiles::NOMSD | TestFiles::FINITE_T | TestFiles::ALL_SYSTEMS);

}


// ----------------------------------------------------------------------------
// StochasticWfn delegate-limit parity (Phases 1a-3a, tag [stochastic_wfn]).
//
// At the delegate limit (inner_nwalkers = 1, inner_nsteps = 0) the inner trial
// ensemble collapses to the single trial-determinant anchor, so every stochastic
// override (Log_Overlap, Energy, MixedDensityMatrix_for_vbias -> vbias) must
// reproduce a plain NOMSD on the same outer walkers, for a single-determinant
// (ndet == 1) trial. A multi-determinant trial diverges by design (a single-det
// inner ensemble cannot reproduce a CI-weighted NOMSD); see StochasticDevelopment.md.
//
// This check is HamOp-agnostic: at inner_nsteps = 0 the stochastic vbias uses the
// *compact* path, so it runs on any cholesky/THC NOMSD fixture (including the
// harness's built-in utils/tests/functional/ files). Full-G (inner_nsteps > 0)
// parity is a separate test that needs the Ne_cc-pvdz DenseFactorized + RHF
// fixture (-> Real3IndexFactorization); see the fixture note in StochasticDevelopment.md.
// ----------------------------------------------------------------------------
template<MEMORY_SPACE MEM>
void stochastic_wfn_matches_nomsd(std::shared_ptr<utils::mpi_context_t<boost::mpi3::communicator>> mpi,
                                  std::string hamil_file, std::string wfn_file)
{
  // StochasticWfn is only built on the NOMSD path; skip PHMSD inputs.
  if (getWavefunctionType(wfn_file) != NOMSD_WFN)
    return;

  const auto info  = read_info_from_wfn(wfn_file, "any");
  const int  NMO   = std::get<0>(info);
  const int  nup   = std::get<1>(info);
  const int  ndown = std::get<2>(info);
  WALKER_TYPES type = afqmc::getWalkerType(wfn_file, "any");
  // Finite-temperature trials are out of scope for the stochastic delegate limit.
  if (type == COLLINEAR_FT or type == NONCOLLINEAR_FT)
    return;
  const int nspin = (type == COLLINEAR) ? 2 : 1;
  const int npol  = (type == NONCOLLINEAR) ? 2 : 1;
  const double dt(0.01);

  std::map<std::string, AFQMCInfo> InfoMap;
  InfoMap.insert(std::pair<std::string, AFQMCInfo>("info0", AFQMCInfo{"info0", NMO, nup, ndown, 0}));

  ptree ham_pt;
  ham_pt.put("name", "ham0");
  ham_pt.put("system", "info0");
  ham_pt.put("filename", hamil_file);
  HamiltonianFactory HamFac(InfoMap);
  HamFac.push("ham0", ham_pt);
  Hamiltonian& ham = HamFac.getHamiltonian(mpi, "ham0");

  const int nwalk = 11; // prime: forces non-trivial splits in shared routines
  std::shared_ptr<utils::RandomGenerator_t<>> rng = std::make_shared<utils::RandomGenerator_t<>>();

  ptree wlk_pt;
  wlk_pt.put("name", "wset0");
  wlk_pt.put("walker_type", walkerTypeToString(type));

  WavefunctionFactory<MEM> WfnFac(InfoMap);

  // Plain NOMSD reference.
  ptree nomsd_pt;
  nomsd_pt.put("name", "wfn_nomsd");
  nomsd_pt.put("system", "info0");
  nomsd_pt.put("filename", wfn_file);
  WfnFac.push("wfn_nomsd", nomsd_pt);
  auto& wfn_nomsd = WfnFac.getWavefunction(mpi, "wfn_nomsd", type, &ham, nwalk);

  // Stochastic trial at the delegate limit: inner_nwalkers = 1, inner_nsteps = 0 (default).
  ptree stoch_pt;
  stoch_pt.put("name", "wfn_stoch");
  stoch_pt.put("system", "info0");
  stoch_pt.put("filename", wfn_file);
  stoch_pt.put("stochastic", true);
  stoch_pt.put("inner_nwalkers", 1);
  WfnFac.push("wfn_stoch", stoch_pt);
  auto& wfn_stoch = WfnFac.getWavefunction(mpi, "wfn_stoch", type, &ham, nwalk);
  REQUIRE(wfn_stoch.is_stochastic_wavefunction());
  // Initialize the inner trial ensemble (anchored at the trial determinant).
  WfnFac.maybe_initialize_stochastic_inner_walkers(wfn_stoch, "wfn_stoch", type, wlk_pt);
  REQUIRE(wfn_stoch.stochastic_inner_walkers_initialized());

  // Bisection checkpoints (pinpoint SIGSEGV; trim once the test is stable).
  REQUIRE(wfn_nomsd.number_of_cholesky_vectors() > 0);
  REQUIRE(wfn_stoch.number_of_cholesky_vectors() > 0);
  REQUIRE(wfn_nomsd.number_of_cholesky_vectors() == wfn_stoch.number_of_cholesky_vectors());
  const bool delegate_limit = (wfn_nomsd.total_number_of_references() == 1);
  REQUIRE(delegate_limit);

  // Deterministic, identical perturbation of the outer walkers (mirrors wfn_factory_sdet),
  // so the two walker sets are bit-for-bit identical going into the reductions.
  auto perturb = [&](auto& wset) {
    std::array<int,2> nels = {nup, ndown};
    for (int spin = 0; spin < nspin; spin++) {
      nda::array<ComplexType, 1> p_h(long(nwalk) * npol * NMO * nels[spin]);
      for (long k = 0; k < p_h.size(); ++k) {
        double v = 0.1 * (k + 1);
        p_h[k] = {std::cos(v), std::sin(v * v)};
      }
      memory::array<MEM, ComplexType, 3> p(reshape(p_h, nwalk, npol * NMO, nels[spin]));
      auto SM = wset.SlaterMatrices(static_cast<SpinTypes>(spin));
      nda::tensor::add(p, "ijk", SM, "ijk");
    }
  };

  // Run Log_Overlap / Energy / vbias and harvest per-walker quantities.
  auto harvest = [&](auto& wfn, auto& wset) {
    wfn.Log_Overlap(wset);
    wfn.runtime_optimization(wset);
    wfn.Energy(wset);
    nda::array<ComplexType, 1> ov(nwalk), e1(nwalk), exx(nwalk), ej(nwalk);
    wset.getProperty(OVLP, ov);
    wset.getProperty(E1_,  e1);
    wset.getProperty(EXX_, exx);
    wset.getProperty(EJ_,  ej);
    // Discrete (model) propagators must initialize potentials before vbias.
    if (wfn.getHamType() == ModelHamiltonian) {
      const long ncv = wfn.number_of_cholesky_vectors();
      memory::array<MEM, ComplexType, 1> vMF_discrete(ncv, ComplexType(0.0, 0.0));
      memory::host_array<ComplexType, 1> nMF(2 * NMO, ComplexType(0.0, 0.0));
      wfn.update_potentials(dt, nMF, vMF_discrete, false);
    }
    memory::array<MEM, ComplexType, 2> X(nwalk, wfn.number_of_cholesky_vectors());
    wfn.vbias(wset, X, dt);
    return std::make_tuple(std::move(ov), std::move(e1), std::move(exx), std::move(ej), nda::to_host(X));
  };

  auto wset_nomsd = make_WalkerSet<MEM>(mpi, wlk_pt, InfoMap["info0"], rng);
  REQUIRE(wset_nomsd.size() == 0);
  wset_nomsd.resize(nwalk, WfnFac.getInitialGuess("wfn_nomsd"));
  REQUIRE(wset_nomsd.size() == nwalk);
  perturb(wset_nomsd);
  REQUIRE(wset_nomsd.size() == nwalk);
  wfn_nomsd.Log_Overlap(wset_nomsd);
  REQUIRE(wset_nomsd.size() == nwalk);
  wfn_nomsd.runtime_optimization(wset_nomsd);
  REQUIRE(wset_nomsd.size() == nwalk);
  wfn_nomsd.Energy(wset_nomsd);
  REQUIRE(wset_nomsd.size() == nwalk);
  nda::array<ComplexType, 1> ov_n(nwalk), e1_n(nwalk), exx_n(nwalk), ej_n(nwalk);
  wset_nomsd.getProperty(OVLP, ov_n);
  wset_nomsd.getProperty(E1_, e1_n);
  wset_nomsd.getProperty(EXX_, exx_n);
  wset_nomsd.getProperty(EJ_, ej_n);
  REQUIRE(ov_n.size() == nwalk);
  memory::array<MEM, ComplexType, 2> X_n(nwalk, wfn_nomsd.number_of_cholesky_vectors());
  wfn_nomsd.vbias(wset_nomsd, X_n, dt);
  REQUIRE(X_n.extent(0) == nwalk);

  auto wset_stoch = make_WalkerSet<MEM>(mpi, wlk_pt, InfoMap["info0"], rng);
  REQUIRE(wset_stoch.size() == 0);
  wset_stoch.resize(nwalk, WfnFac.getInitialGuess("wfn_stoch"));
  REQUIRE(wset_stoch.size() == nwalk);
  perturb(wset_stoch);
  auto [ov_s, e1_s, exx_s, ej_s, X_s] = harvest(wfn_stoch, wset_stoch);
  REQUIRE(ov_s.size() == nwalk);

  if (delegate_limit) {
    CHECK_THAT(ov_s, utils::Approx(ov_n));
    CHECK_THAT(e1_s, utils::Approx(e1_n));
    CHECK_THAT(exx_s, utils::Approx(exx_n));
    CHECK_THAT(ej_s, utils::Approx(ej_n));
    CHECK_THAT(X_s, utils::Approx(nda::to_host(X_n)));
  }
}

TEST_CASE("stochastic_wfn_matches_nomsd", "[wfn_factory][stochastic_wfn]")
{
  auto& mpi = utils::make_unit_test_mpi_context();

  app_log(0,"StochasticWfn delegate-limit parity unit test.");

  using namespace utils;

  run_test_with_files([&]<auto MEM>(std::string hamil_file, std::string wfn_file, WALKER_TYPES) {
    stochastic_wfn_matches_nomsd<MEM>(mpi, hamil_file, wfn_file);
  }, UTEST_HAMIL, UTEST_WFN, TestFiles::RHF | TestFiles::UHF | TestFiles::NOMSD | TestFiles::ALL_SYSTEMS);

}


// ----------------------------------------------------------------------------
// StochasticWfn build + inner-init smoke test (tag [stochastic_wfn]).
//
// Isolates the StochasticWfn *construction* and inner-walker initialization from
// the reductions and from the plain-NOMSD path: it builds ONLY a stochastic trial
// and initializes its inner ensemble -- no outer walker set, no Log_Overlap/Energy/
// vbias, no second wavefunction. Triangulating the SIGSEGV in stochastic_wfn_matches_nomsd:
//   - If THIS test SIGSEGVs   -> fault is in the stochastic build / inner-walker init.
//   - If THIS test passes but `wfn_factory: sdet` SIGSEGVs on the same fixture
//                              -> fault is in the plain dense-Hamiltonian path (not stochastic).
//   - If both pass            -> fault is specific to running reductions after the stochastic
//                                 build (e.g. shared buffer-manager / global state interaction).
// ----------------------------------------------------------------------------
template<MEMORY_SPACE MEM>
void stochastic_build_smoke(std::shared_ptr<utils::mpi_context_t<boost::mpi3::communicator>> mpi,
                            std::string hamil_file, std::string wfn_file)
{
  if (getWavefunctionType(wfn_file) != NOMSD_WFN)
    return;

  const auto info  = read_info_from_wfn(wfn_file, "any");
  const int  NMO   = std::get<0>(info);
  const int  nup   = std::get<1>(info);
  const int  ndown = std::get<2>(info);
  WALKER_TYPES type = afqmc::getWalkerType(wfn_file, "any");
  if (type == COLLINEAR_FT or type == NONCOLLINEAR_FT)
    return;

  std::map<std::string, AFQMCInfo> InfoMap;
  InfoMap.insert(std::pair<std::string, AFQMCInfo>("info0", AFQMCInfo{"info0", NMO, nup, ndown, 0}));

  ptree ham_pt;
  ham_pt.put("name", "ham0");
  ham_pt.put("system", "info0");
  ham_pt.put("filename", hamil_file);
  HamiltonianFactory HamFac(InfoMap);
  HamFac.push("ham0", ham_pt);
  Hamiltonian& ham = HamFac.getHamiltonian(mpi, "ham0");

  ptree wlk_pt;
  wlk_pt.put("name", "wset0");
  wlk_pt.put("walker_type", walkerTypeToString(type));

  WavefunctionFactory<MEM> WfnFac(InfoMap);
  ptree stoch_pt;
  stoch_pt.put("name", "wfn_stoch");
  stoch_pt.put("system", "info0");
  stoch_pt.put("filename", wfn_file);
  stoch_pt.put("stochastic", true);
  stoch_pt.put("inner_nwalkers", 1);
  WfnFac.push("wfn_stoch", stoch_pt);

  app_log(0, "[stochastic_build_smoke] building stochastic wavefunction");
  auto& wfn_stoch = WfnFac.getWavefunction(mpi, "wfn_stoch", type, &ham, 11);
  app_log(0, "[stochastic_build_smoke] built; initializing inner walkers");
  REQUIRE(wfn_stoch.is_stochastic_wavefunction());
  WfnFac.maybe_initialize_stochastic_inner_walkers(wfn_stoch, "wfn_stoch", type, wlk_pt);
  REQUIRE(wfn_stoch.stochastic_inner_walkers_initialized());
  app_log(0, "[stochastic_build_smoke] inner walkers initialized OK");
}

TEST_CASE("stochastic_build_smoke", "[wfn_factory][stochastic_wfn]")
{
  auto& mpi = utils::make_unit_test_mpi_context();

  app_log(0,"StochasticWfn build + inner-init smoke test.");

  using namespace utils;

  run_test_with_files([&]<auto MEM>(std::string hamil_file, std::string wfn_file, WALKER_TYPES) {
    stochastic_build_smoke<MEM>(mpi, hamil_file, wfn_file);
  }, UTEST_HAMIL, UTEST_WFN, TestFiles::RHF | TestFiles::UHF | TestFiles::NOMSD | TestFiles::ALL_SYSTEMS);

}


} // namespace sfqmc
