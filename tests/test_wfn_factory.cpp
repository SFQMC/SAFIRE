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
#include "AFQMC/Propagators/PropagatorFactory.h"

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


// ============================================================================
// Stochastic hot-path overrides: delegate-limit parity (Phases 2a/2b/3a) and the
// dynamic ensemble (Phase 3b), tag [stochastic_wfn].
//
// These exercise the StochasticWfn reductions through the public Wavefunction API
// only (the overhaul variant keeps its typed internals private): Log_Overlap (2a),
// Energy (2b), and vbias (3a) on the OUTER walkers, plus the dynamic free-projection
// drive (3b). Two invariants are checked, mirroring the develop reference suite:
//   (1) inner_nwalkers invariance -- a static replicated ensemble (inner_nsteps = 0)
//       gives observables independent of inner_nwalkers (holds for ANY trial);
//   (2) delegate limit -- for a single-determinant trial the stochastic reduction
//       equals the plain NOMSD result. Multi-determinant trials diverge by design (a
//       single-determinant inner ensemble cannot reproduce a CI-weighted NOMSD), so
//       (2) is gated on ndet == 1. See StochasticDevelopment.md.
// ============================================================================

// Deterministic, reproducible perturbation of the outer walker Slater matrices, identical to the
// sequence in stochastic_wfn_matches_nomsd, so independently-built walker sets are bit-for-bit
// identical going into the reductions (a meaningful parity check needs non-trivial overlaps).
template<MEMORY_SPACE MEM>
void perturb_stochastic_walkers(WalkerSet<MEM>& wset, WALKER_TYPES type, int NMO, int nup, int ndown)
{
  const int nspin = (type == COLLINEAR) ? 2 : 1;
  const int npol  = (type == NONCOLLINEAR) ? 2 : 1;
  const int nwalk = wset.size();
  std::array<int, 2> nels = {nup, ndown};
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
}

// The OVLP property stores a complex LOG overlap whose imaginary part (phase) is defined only mod
// 2*pi. NOMSD::Log_Overlap accumulates the unwrapped log-det phase, while the stochastic reduction
// sums in linear space and then takes the log, so it returns the principal branch -- the two can
// differ by an integer multiple of 2*pi*i while describing the SAME overlap. Compare the physical
// (linear) overlaps exp(log_ov), which are branch-independent (this also matches the develop
// reference suite, which compared linear overlaps directly).
inline nda::array<ComplexType, 1> linear_overlap(nda::array<ComplexType, 1> const& log_ov)
{
  nda::array<ComplexType, 1> lin(log_ov.size());
  for (long i = 0; i < log_ov.size(); ++i)
    lin(i) = std::exp(log_ov(i));
  return lin;
}

// Phase 2a: StochasticWfn::Log_Overlap reduces the inner ensemble into an effective trial overlap
// (Eq. 24 of arXiv:2505.18519, static-ensemble limit). Overlap is read WITHOUT a following Energy
// call -- Energy overwrites the OVLP walker property and would otherwise mask the override.
template<MEMORY_SPACE MEM>
void stochastic_overlap_matches_nomsd(std::shared_ptr<utils::mpi_context_t<boost::mpi3::communicator>> mpi,
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

  const int nwalk = 11; // prime: forces non-trivial splits in shared routines
  std::shared_ptr<utils::RandomGenerator_t<>> rng = std::make_shared<utils::RandomGenerator_t<>>();
  ptree wlk_pt;
  wlk_pt.put("name", "wset0");
  wlk_pt.put("walker_type", walkerTypeToString(type));

  WavefunctionFactory<MEM> WfnFac(InfoMap);

  ptree nomsd_pt;
  nomsd_pt.put("name", "wfn_nomsd_ov");
  nomsd_pt.put("system", "info0");
  nomsd_pt.put("filename", wfn_file);
  WfnFac.push("wfn_nomsd_ov", nomsd_pt);
  auto& wfn_nomsd = WfnFac.getWavefunction(mpi, "wfn_nomsd_ov", type, &ham, nwalk);

  // Stochastic trials at the static limit (inner_nsteps = 0) with inner_nwalkers = 1 and = 3.
  auto build_stoch = [&](const std::string& name, int inner_nwalkers) -> Wavefunction<MEM>& {
    ptree pt;
    pt.put("name", name);
    pt.put("system", "info0");
    pt.put("filename", wfn_file);
    pt.put("stochastic", true);
    pt.put("inner_nwalkers", inner_nwalkers);
    WfnFac.push(name, pt);
    auto& w = WfnFac.getWavefunction(mpi, name, type, &ham, nwalk);
    REQUIRE(w.is_stochastic_wavefunction());
    WfnFac.maybe_initialize_stochastic_inner_walkers(w, name, type, wlk_pt);
    REQUIRE(w.stochastic_inner_walkers_initialized());
    return w;
  };
  auto& wfn_s1 = build_stoch("wfn_stoch_ov1", 1);
  auto& wfn_s3 = build_stoch("wfn_stoch_ov3", 3);

  // Every walker set is initialized from the SAME guess and SAME deterministic perturbation, so the
  // three wavefunctions see bit-for-bit identical outer walkers.
  auto collect_overlaps = [&](Wavefunction<MEM>& wfn) {
    auto wset = make_WalkerSet<MEM>(mpi, wlk_pt, InfoMap["info0"], rng);
    wset.resize(nwalk, WfnFac.getInitialGuess("wfn_nomsd_ov"));
    perturb_stochastic_walkers<MEM>(wset, type, NMO, nup, ndown);
    wfn.Log_Overlap(wset);
    nda::array<ComplexType, 1> ov(nwalk);
    wset.getProperty(OVLP, ov);
    return ov;
  };
  auto ov_ref = collect_overlaps(wfn_nomsd);
  auto ov_s1  = collect_overlaps(wfn_s1);
  auto ov_s3  = collect_overlaps(wfn_s3);

  // (1) inner_nwalkers invariance (holds for any trial).
  CHECK_THAT(linear_overlap(ov_s3), utils::Approx(linear_overlap(ov_s1)));
  // (2) delegate limit: single-determinant trial => stochastic overlap == NOMSD overlap.
  if (wfn_nomsd.total_number_of_references() == 1)
    CHECK_THAT(linear_overlap(ov_s1), utils::Approx(linear_overlap(ov_ref)));
}

TEST_CASE("stochastic_overlap_matches_nomsd", "[wfn_factory][stochastic_wfn]")
{
  auto& mpi = utils::make_unit_test_mpi_context();
  app_log(0, "StochasticWfn Log_Overlap delegate-limit parity (Phase 2a).");
  using namespace utils;
  run_test_with_files([&]<auto MEM>(std::string hamil_file, std::string wfn_file, WALKER_TYPES) {
    stochastic_overlap_matches_nomsd<MEM>(mpi, hamil_file, wfn_file);
  }, UTEST_HAMIL, UTEST_WFN, TestFiles::RHF | TestFiles::UHF | TestFiles::NOMSD | TestFiles::ALL_SYSTEMS);
}

// Phase 2b: StochasticWfn::Energy reduces the inner ensemble into an effective local energy
// (E1, EXX, EJ) and overlap per outer walker (Eq. 27 of arXiv:2505.18519, static-ensemble limit).
template<MEMORY_SPACE MEM>
void stochastic_energy_matches_nomsd(std::shared_ptr<utils::mpi_context_t<boost::mpi3::communicator>> mpi,
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

  const int nwalk = 11;
  std::shared_ptr<utils::RandomGenerator_t<>> rng = std::make_shared<utils::RandomGenerator_t<>>();
  ptree wlk_pt;
  wlk_pt.put("name", "wset0");
  wlk_pt.put("walker_type", walkerTypeToString(type));

  WavefunctionFactory<MEM> WfnFac(InfoMap);

  ptree nomsd_pt;
  nomsd_pt.put("name", "wfn_nomsd_en");
  nomsd_pt.put("system", "info0");
  nomsd_pt.put("filename", wfn_file);
  WfnFac.push("wfn_nomsd_en", nomsd_pt);
  auto& wfn_nomsd = WfnFac.getWavefunction(mpi, "wfn_nomsd_en", type, &ham, nwalk);

  auto build_stoch = [&](const std::string& name, int inner_nwalkers) -> Wavefunction<MEM>& {
    ptree pt;
    pt.put("name", name);
    pt.put("system", "info0");
    pt.put("filename", wfn_file);
    pt.put("stochastic", true);
    pt.put("inner_nwalkers", inner_nwalkers);
    WfnFac.push(name, pt);
    auto& w = WfnFac.getWavefunction(mpi, name, type, &ham, nwalk);
    WfnFac.maybe_initialize_stochastic_inner_walkers(w, name, type, wlk_pt);
    return w;
  };
  auto& wfn_s1 = build_stoch("wfn_stoch_en1", 1);
  auto& wfn_s3 = build_stoch("wfn_stoch_en3", 3);

  struct WalkerEnergies
  {
    nda::array<ComplexType, 1> ov, e1, exx, ej;
  };
  auto collect_energies = [&](Wavefunction<MEM>& wfn) {
    auto wset = make_WalkerSet<MEM>(mpi, wlk_pt, InfoMap["info0"], rng);
    wset.resize(nwalk, WfnFac.getInitialGuess("wfn_nomsd_en"));
    perturb_stochastic_walkers<MEM>(wset, type, NMO, nup, ndown);
    wfn.Energy(wset);
    WalkerEnergies out;
    out.ov.resize(nwalk);
    out.e1.resize(nwalk);
    out.exx.resize(nwalk);
    out.ej.resize(nwalk);
    wset.getProperty(OVLP, out.ov);
    wset.getProperty(E1_, out.e1);
    wset.getProperty(EXX_, out.exx);
    wset.getProperty(EJ_, out.ej);
    return out;
  };
  WalkerEnergies ref = collect_energies(wfn_nomsd);
  WalkerEnergies s1  = collect_energies(wfn_s1);
  WalkerEnergies s3  = collect_energies(wfn_s3);

  // (1) inner_nwalkers invariance.
  CHECK_THAT(linear_overlap(s3.ov), utils::Approx(linear_overlap(s1.ov)));
  CHECK_THAT(s3.e1, utils::Approx(s1.e1));
  CHECK_THAT(s3.exx, utils::Approx(s1.exx));
  CHECK_THAT(s3.ej, utils::Approx(s1.ej));

  // (2) delegate limit: single-determinant trial => stochastic energy/overlap == NOMSD.
  if (wfn_nomsd.total_number_of_references() == 1)
  {
    CHECK_THAT(linear_overlap(s1.ov), utils::Approx(linear_overlap(ref.ov)));
    CHECK_THAT(s1.e1, utils::Approx(ref.e1));
    CHECK_THAT(s1.exx, utils::Approx(ref.exx));
    CHECK_THAT(s1.ej, utils::Approx(ref.ej));
  }

  // (3) Overlap/Energy consistency: Energy's Ov is the same reduction as the Phase 2a Log_Overlap,
  // computed through a different code path, so the two must agree.
  {
    auto wset = make_WalkerSet<MEM>(mpi, wlk_pt, InfoMap["info0"], rng);
    wset.resize(nwalk, WfnFac.getInitialGuess("wfn_nomsd_en"));
    perturb_stochastic_walkers<MEM>(wset, type, NMO, nup, ndown);
    wfn_s1.Log_Overlap(wset);
    nda::array<ComplexType, 1> ov_ovlp(nwalk);
    wset.getProperty(OVLP, ov_ovlp);
    CHECK_THAT(linear_overlap(ov_ovlp), utils::Approx(linear_overlap(s1.ov)));
  }

  // (4) Propagator entry point: the 3-arg Energy(wset, E, Ov) -- called directly by the
  // local-energy propagation path in the propagator (not the property-setter form) -- agrees with
  // Energy(wset). Exercises the 3-arg overload with caller-allocated buffers.
  {
    auto wset = make_WalkerSet<MEM>(mpi, wlk_pt, InfoMap["info0"], rng);
    wset.resize(nwalk, WfnFac.getInitialGuess("wfn_nomsd_en"));
    perturb_stochastic_walkers<MEM>(wset, type, NMO, nup, ndown);
    memory::array<MEM, ComplexType, 2> E_direct(nwalk, 3);
    memory::array<MEM, ComplexType, 1> Ov_direct(nwalk);
    wfn_s1.Energy(wset, E_direct, Ov_direct);
    auto E_h  = nda::to_host(E_direct);
    auto Ov_h = nda::to_host(Ov_direct);
    nda::array<ComplexType, 1> e1_col(nwalk), exx_col(nwalk), ej_col(nwalk);
    for (int n = 0; n < nwalk; ++n)
    {
      e1_col(n)  = E_h(n, 0);
      exx_col(n) = E_h(n, 1);
      ej_col(n)  = E_h(n, 2);
    }
    CHECK_THAT(linear_overlap(Ov_h), utils::Approx(linear_overlap(s1.ov)));
    CHECK_THAT(e1_col, utils::Approx(s1.e1));
    CHECK_THAT(exx_col, utils::Approx(s1.exx));
    CHECK_THAT(ej_col, utils::Approx(s1.ej));
  }
}

TEST_CASE("stochastic_energy_matches_nomsd", "[wfn_factory][stochastic_wfn]")
{
  auto& mpi = utils::make_unit_test_mpi_context();
  app_log(0, "StochasticWfn Energy delegate-limit parity (Phase 2b).");
  using namespace utils;
  run_test_with_files([&]<auto MEM>(std::string hamil_file, std::string wfn_file, WALKER_TYPES) {
    stochastic_energy_matches_nomsd<MEM>(mpi, hamil_file, wfn_file);
  }, UTEST_HAMIL, UTEST_WFN, TestFiles::RHF | TestFiles::UHF | TestFiles::NOMSD | TestFiles::ALL_SYSTEMS);
}

// Phase 3a: StochasticWfn::MixedDensityMatrix_for_vbias reduces the inner ensemble into the mixed
// density matrix the force bias contracts against (estimator 3 of arXiv:2505.18519, static limit),
// and vbias contracts it (estimator 4, x_gamma[w] = L_gamma . G[w]) against the True-Ham Cholesky.
// The overhaul vbias(wset, X, dt) drives MixedDensityMatrix_for_vbias internally, so we compare the
// resulting force bias X (= L.G) directly; the intermediate G is not exposed on the variant.
template<MEMORY_SPACE MEM>
void stochastic_vbias_matches_nomsd(std::shared_ptr<utils::mpi_context_t<boost::mpi3::communicator>> mpi,
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

  const int nwalk = 11;
  std::shared_ptr<utils::RandomGenerator_t<>> rng = std::make_shared<utils::RandomGenerator_t<>>();
  ptree wlk_pt;
  wlk_pt.put("name", "wset0");
  wlk_pt.put("walker_type", walkerTypeToString(type));

  WavefunctionFactory<MEM> WfnFac(InfoMap);

  ptree nomsd_pt;
  nomsd_pt.put("name", "wfn_nomsd_vb");
  nomsd_pt.put("system", "info0");
  nomsd_pt.put("filename", wfn_file);
  WfnFac.push("wfn_nomsd_vb", nomsd_pt);
  auto& wfn_nomsd = WfnFac.getWavefunction(mpi, "wfn_nomsd_vb", type, &ham, nwalk);

  auto build_stoch = [&](const std::string& name, int inner_nwalkers) -> Wavefunction<MEM>& {
    ptree pt;
    pt.put("name", name);
    pt.put("system", "info0");
    pt.put("filename", wfn_file);
    pt.put("stochastic", true);
    pt.put("inner_nwalkers", inner_nwalkers);
    WfnFac.push(name, pt);
    auto& w = WfnFac.getWavefunction(mpi, name, type, &ham, nwalk);
    WfnFac.maybe_initialize_stochastic_inner_walkers(w, name, type, wlk_pt);
    return w;
  };
  auto& wfn_s1 = build_stoch("wfn_stoch_vb1", 1);
  auto& wfn_s3 = build_stoch("wfn_stoch_vb3", 3);

  auto collect_vbias = [&](Wavefunction<MEM>& wfn) {
    auto wset = make_WalkerSet<MEM>(mpi, wlk_pt, InfoMap["info0"], rng);
    wset.resize(nwalk, WfnFac.getInitialGuess("wfn_nomsd_vb"));
    perturb_stochastic_walkers<MEM>(wset, type, NMO, nup, ndown);
    // Discrete (model) propagators must initialize potentials before vbias.
    if (wfn.getHamType() == ModelHamiltonian)
    {
      const long ncv = wfn.number_of_cholesky_vectors();
      memory::array<MEM, ComplexType, 1> vMF_discrete(ncv, ComplexType(0.0, 0.0));
      memory::host_array<ComplexType, 1> nMF(2 * NMO, ComplexType(0.0, 0.0));
      wfn.update_potentials(dt, nMF, vMF_discrete, false);
    }
    memory::array<MEM, ComplexType, 2> X(nwalk, wfn.number_of_cholesky_vectors());
    wfn.vbias(wset, X, dt);
    return nda::to_host(X);
  };
  auto X_ref = collect_vbias(wfn_nomsd);
  auto X_s1  = collect_vbias(wfn_s1);
  auto X_s3  = collect_vbias(wfn_s3);

  // (1) inner_nwalkers invariance.
  CHECK_THAT(X_s3, utils::Approx(X_s1));
  // (2) delegate limit: single-determinant trial => stochastic force bias == NOMSD.
  if (wfn_nomsd.total_number_of_references() == 1)
    CHECK_THAT(X_s1, utils::Approx(X_ref));
}

TEST_CASE("stochastic_vbias_matches_nomsd", "[wfn_factory][stochastic_wfn]")
{
  auto& mpi = utils::make_unit_test_mpi_context();
  app_log(0, "StochasticWfn vbias delegate-limit parity (Phase 3a).");
  using namespace utils;
  run_test_with_files([&]<auto MEM>(std::string hamil_file, std::string wfn_file, WALKER_TYPES) {
    stochastic_vbias_matches_nomsd<MEM>(mpi, hamil_file, wfn_file);
  }, UTEST_HAMIL, UTEST_WFN, TestFiles::RHF | TestFiles::UHF | TestFiles::NOMSD | TestFiles::ALL_SYSTEMS);
}


// ============================================================================


// Phase 3b: the dynamic ensemble + un-rotated full-G kernels (CLOSED/CPU only this phase).
//
// inner_nsteps > 0 routes the reductions through the un-rotated full-G energy/force-bias kernels
// (energy_from_fullG / the full-G layout in vbias_from_G) instead of the compact nd = 0 half-rotated
// path. The construction rejects inner_nsteps > 0 unless the trial is CLOSED (RHF) and the build is
// CPU, so these tests skip on non-CLOSED inputs and on DEVICE_MEMORY.
// ============================================================================

// Build the un-rotated full-G path WITHOUT ever resampling (inner_nsteps = 1 but begin_inner_step is
// never called), so the inner ensemble stays at the anchor |phi_T>. There the full-G kernels must
// reproduce the compact nd = 0 path (inner_nsteps = 0, which delegates to NOMSD) and NOMSD itself.
template<MEMORY_SPACE MEM>
void stochastic_full_g_matches_compact(std::shared_ptr<utils::mpi_context_t<boost::mpi3::communicator>> mpi,
                                       std::string hamil_file, std::string wfn_file)
{
  if (getWavefunctionType(wfn_file) != NOMSD_WFN)
    return;
  if constexpr (MEM != HOST_MEMORY)
    return; // Phase 3b un-rotated full-G kernels are CPU-only this phase.
  else
  {
    const auto info  = read_info_from_wfn(wfn_file, "any");
    const int  NMO   = std::get<0>(info);
    const int  nup   = std::get<1>(info);
    const int  ndown = std::get<2>(info);
    WALKER_TYPES type = afqmc::getWalkerType(wfn_file, "any");
    if (type != CLOSED)
      return; // Phase 3b un-rotated full-G kernels support CLOSED (RHF) trials only.
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

    const int nwalk = 11;
    std::shared_ptr<utils::RandomGenerator_t<>> rng = std::make_shared<utils::RandomGenerator_t<>>();
    ptree wlk_pt;
    wlk_pt.put("name", "wset0");
    wlk_pt.put("walker_type", walkerTypeToString(type));

    WavefunctionFactory<MEM> WfnFac(InfoMap);

    ptree nomsd_pt;
    nomsd_pt.put("name", "wfn_nomsd_fg");
    nomsd_pt.put("system", "info0");
    nomsd_pt.put("filename", wfn_file);
    WfnFac.push("wfn_nomsd_fg", nomsd_pt);
    auto& wfn_nomsd = WfnFac.getWavefunction(mpi, "wfn_nomsd_fg", type, &ham, nwalk);

    auto build_stoch = [&](const std::string& name, int inner_nwalkers, int inner_nsteps) -> Wavefunction<MEM>& {
      ptree pt;
      pt.put("name", name);
      pt.put("system", "info0");
      pt.put("filename", wfn_file);
      pt.put("stochastic", true);
      pt.put("inner_nwalkers", inner_nwalkers);
      pt.put("inner_nsteps", inner_nsteps);
      WfnFac.push(name, pt);
      auto& w = WfnFac.getWavefunction(mpi, name, type, &ham, nwalk);
      WfnFac.maybe_initialize_stochastic_inner_walkers(w, name, type, wlk_pt);
      return w;
    };
    auto& wfn_compact = build_stoch("wfn_stoch_fg0", 1, 0); // compact nd = 0 (delegates to NOMSD)
    auto& wfn_full    = build_stoch("wfn_stoch_fg1", 1, 1); // un-rotated full-G; NOT resampled

    struct WalkerEnergies
    {
      nda::array<ComplexType, 1> ov, e1, exx, ej;
    };
    auto collect_energies = [&](Wavefunction<MEM>& wfn) {
      auto wset = make_WalkerSet<MEM>(mpi, wlk_pt, InfoMap["info0"], rng);
      wset.resize(nwalk, WfnFac.getInitialGuess("wfn_nomsd_fg"));
      perturb_stochastic_walkers<MEM>(wset, type, NMO, nup, ndown);
      wfn.Energy(wset); // never calls begin_inner_step -> full-G evaluated at the anchor
      WalkerEnergies out;
      out.ov.resize(nwalk);
      out.e1.resize(nwalk);
      out.exx.resize(nwalk);
      out.ej.resize(nwalk);
      wset.getProperty(OVLP, out.ov);
      wset.getProperty(E1_, out.e1);
      wset.getProperty(EXX_, out.exx);
      wset.getProperty(EJ_, out.ej);
      return out;
    };
    WalkerEnergies ref  = collect_energies(wfn_nomsd);
    WalkerEnergies comp = collect_energies(wfn_compact);
    WalkerEnergies full = collect_energies(wfn_full);

    // full-G energy at the anchor == compact nd = 0 == NOMSD (single-determinant delegate limit).
    CHECK_THAT(full.e1, utils::Approx(comp.e1));
    CHECK_THAT(full.exx, utils::Approx(comp.exx));
    CHECK_THAT(full.ej, utils::Approx(comp.ej));
    CHECK_THAT(linear_overlap(full.ov), utils::Approx(linear_overlap(comp.ov)));
    if (wfn_nomsd.total_number_of_references() == 1)
    {
      CHECK_THAT(full.e1, utils::Approx(ref.e1));
      CHECK_THAT(full.exx, utils::Approx(ref.exx));
      CHECK_THAT(full.ej, utils::Approx(ref.ej));
      CHECK_THAT(linear_overlap(full.ov), utils::Approx(linear_overlap(ref.ov)));
    }

    // Force bias: the full-Likn contraction at the anchor == the compact half-rotated one. Compare
    // the bias X = L.G ([nwalk][nCV] in both), which is layout-independent (G layouts differ).
    auto collect_vbias = [&](Wavefunction<MEM>& wfn) {
      auto wset = make_WalkerSet<MEM>(mpi, wlk_pt, InfoMap["info0"], rng);
      wset.resize(nwalk, WfnFac.getInitialGuess("wfn_nomsd_fg"));
      perturb_stochastic_walkers<MEM>(wset, type, NMO, nup, ndown);
      memory::array<MEM, ComplexType, 2> X(nwalk, wfn.number_of_cholesky_vectors());
      wfn.vbias(wset, X, dt);
      return nda::to_host(X);
    };
    auto X_comp = collect_vbias(wfn_compact);
    auto X_full = collect_vbias(wfn_full);
    CHECK_THAT(X_full, utils::Approx(X_comp));
  }
}

TEST_CASE("stochastic_full_g_matches_compact", "[wfn_factory][stochastic_wfn]")
{
  auto& mpi = utils::make_unit_test_mpi_context();
  app_log(0, "StochasticWfn un-rotated full-G vs compact at the anchor (Phase 3b).");
  using namespace utils;
  run_test_with_files([&]<auto MEM>(std::string hamil_file, std::string wfn_file, WALKER_TYPES) {
    stochastic_full_g_matches_compact<MEM>(mpi, hamil_file, wfn_file);
  }, UTEST_HAMIL, UTEST_WFN, TestFiles::RHF | TestFiles::UHF | TestFiles::NOMSD | TestFiles::ALL_SYSTEMS);
}

// Drive the dynamic ensemble: with inner_nsteps > 0 and begin_inner_step() before each step, the
// inner ensemble is reset to the anchor and advanced inner_nsteps free-projection B_T steps, then
// scored with the un-rotated full-G kernels on the MOVED walkers. Exercises resample + all four
// overrides end to end; asserts finiteness (the values are stochastic, not fixed).
template<MEMORY_SPACE MEM>
void stochastic_dynamic_ensemble_smoke(std::shared_ptr<utils::mpi_context_t<boost::mpi3::communicator>> mpi,
                                       std::string hamil_file, std::string wfn_file)
{
  if (getWavefunctionType(wfn_file) != NOMSD_WFN)
    return;
  if constexpr (MEM != HOST_MEMORY)
    return; // Phase 3b un-rotated full-G kernels are CPU-only this phase.
  else
  {
    const auto info  = read_info_from_wfn(wfn_file, "any");
    const int  NMO   = std::get<0>(info);
    const int  nup   = std::get<1>(info);
    const int  ndown = std::get<2>(info);
    WALKER_TYPES type = afqmc::getWalkerType(wfn_file, "any");
    if (type != CLOSED)
      return;
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

    const int nwalk = 11;
    std::shared_ptr<utils::RandomGenerator_t<>> rng = std::make_shared<utils::RandomGenerator_t<>>();
    ptree wlk_pt;
    wlk_pt.put("name", "wset0");
    wlk_pt.put("walker_type", walkerTypeToString(type));

    WavefunctionFactory<MEM> WfnFac(InfoMap);
    ptree pt;
    pt.put("name", "wfn_stoch_dyn");
    pt.put("system", "info0");
    pt.put("filename", wfn_file);
    pt.put("stochastic", true);
    pt.put("inner_nwalkers", 4);
    pt.put("inner_nsteps", 1);
    ptree inner_prop;
    inner_prop.put("timestep", 0.01);
    pt.put_child("inner_propagator", inner_prop);
    WfnFac.push("wfn_stoch_dyn", pt);
    auto& wfn = WfnFac.getWavefunction(mpi, "wfn_stoch_dyn", type, &ham, nwalk);
    WfnFac.maybe_initialize_stochastic_inner_walkers(wfn, "wfn_stoch_dyn", type, wlk_pt);

    auto wset = make_WalkerSet<MEM>(mpi, wlk_pt, InfoMap["info0"], rng);
    wset.resize(nwalk, WfnFac.getInitialGuess("wfn_stoch_dyn"));
    perturb_stochastic_walkers<MEM>(wset, type, NMO, nup, ndown);

    // Each "step" mimics one outer propagator step: arm the latch, then run the hot-path overrides in
    // order. begin_inner_step() + the first reduction resamples the inner ensemble once; the rest
    // score the same moved ensemble. Assert finiteness (values are stochastic, not fixed).
    for (int step = 0; step < 3; ++step)
    {
      wfn.begin_inner_step();
      memory::array<MEM, ComplexType, 2> X(nwalk, wfn.number_of_cholesky_vectors());
      wfn.vbias(wset, X, dt); // first reduction -> resamples; full-G force bias on the moved ensemble
      wfn.Energy(wset);       // energy_fullG on the same ensemble
      wfn.Log_Overlap(wset);
      nda::array<ComplexType, 1> ov(nwalk), e1(nwalk), exx(nwalk), ej(nwalk);
      wset.getProperty(OVLP, ov);
      wset.getProperty(E1_, e1);
      wset.getProperty(EXX_, exx);
      wset.getProperty(EJ_, ej);
      auto X_h = nda::to_host(X);
      for (int w = 0; w < nwalk; ++w)
      {
        REQUIRE(std::isfinite(real(ov(w))));
        REQUIRE(std::isfinite(imag(ov(w))));
        REQUIRE(std::isfinite(real(e1(w))));
        REQUIRE(std::isfinite(real(exx(w))));
        REQUIRE(std::isfinite(real(ej(w))));
      }
      for (int w = 0; w < nwalk; ++w)
        for (int g = 0; g < X_h.extent(1); ++g)
          REQUIRE(std::isfinite(real(X_h(w, g))));
    }
  }
}

TEST_CASE("stochastic_dynamic_ensemble_smoke", "[wfn_factory][stochastic_wfn]")
{
  auto& mpi = utils::make_unit_test_mpi_context();
  app_log(0, "StochasticWfn dynamic free-projection ensemble smoke (Phase 3b).");
  using namespace utils;
  run_test_with_files([&]<auto MEM>(std::string hamil_file, std::string wfn_file, WALKER_TYPES) {
    stochastic_dynamic_ensemble_smoke<MEM>(mpi, hamil_file, wfn_file);
  }, UTEST_HAMIL, UTEST_WFN, TestFiles::RHF | TestFiles::UHF | TestFiles::NOMSD | TestFiles::ALL_SYSTEMS);
}

// End-to-end propagator integration: build a real OUTER AFQMCBasePropagator (default hybrid) bound to
// the dynamic stochastic trial (inner_nsteps = 1) and run Propagate() steps. Drives the full hot path
// THROUGH the propagator (vbias -> vHS -> apply -> Log_Overlap), validating that the stochastic
// overrides plug into a real propagation step. Asserts the walkers stay finite.
template<MEMORY_SPACE MEM>
void stochastic_propagator_step(std::shared_ptr<utils::mpi_context_t<boost::mpi3::communicator>> mpi,
                                std::string hamil_file, std::string wfn_file)
{
  if (getWavefunctionType(wfn_file) != NOMSD_WFN)
    return;
  if constexpr (MEM != HOST_MEMORY)
    return; // Phase 3b un-rotated full-G kernels are CPU-only this phase.
  else
  {
    const auto info  = read_info_from_wfn(wfn_file, "any");
    const int  NMO   = std::get<0>(info);
    const int  nup   = std::get<1>(info);
    const int  ndown = std::get<2>(info);
    WALKER_TYPES type = afqmc::getWalkerType(wfn_file, "any");
    if (type != CLOSED)
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

    const int nwalk = 11;
    std::shared_ptr<utils::RandomGenerator_t<>> rng = std::make_shared<utils::RandomGenerator_t<>>();
    std::shared_ptr<utils::RandomGenerator_t<MEM>> rng_dev =
        std::make_shared<utils::RandomGenerator_t<MEM>>(utils::make_rng<MEM>(13));
    ptree wlk_pt;
    wlk_pt.put("name", "wset0");
    wlk_pt.put("walker_type", walkerTypeToString(type));

    WavefunctionFactory<MEM> WfnFac(InfoMap);
    ptree pt;
    pt.put("name", "wfn_stoch_prop");
    pt.put("system", "info0");
    pt.put("filename", wfn_file);
    pt.put("stochastic", true);
    pt.put("inner_nwalkers", 4);
    pt.put("inner_nsteps", 1);
    ptree inner_prop;
    inner_prop.put("timestep", 0.01);
    pt.put_child("inner_propagator", inner_prop);
    WfnFac.push("wfn_stoch_prop", pt);
    auto& wfn = WfnFac.getWavefunction(mpi, "wfn_stoch_prop", type, &ham, nwalk);
    WfnFac.maybe_initialize_stochastic_inner_walkers(wfn, "wfn_stoch_prop", type, wlk_pt);

    auto wset = make_WalkerSet<MEM>(mpi, wlk_pt, InfoMap["info0"], rng);
    wset.resize(nwalk, WfnFac.getInitialGuess("wfn_stoch_prop"));

    // Prime overlaps/energies (anchor ensemble; begin_inner_step armed by the propagator each step)
    // and pick an energy shift so the hybrid weights stay well-scaled over the test steps.
    wfn.Log_Overlap(wset);
    wfn.Energy(wset);
    ComplexType eav(0.0), ow(0.0);
    for (auto it = wset.begin(); it != wset.end(); ++it)
    {
      eav += it->get_property(WEIGHT) * it->energy();
      ow += it->get_property(WEIGHT);
    }
    RealType Eshift = (std::abs(ow) > 1e-12) ? real(eav / ow) : RealType(0);

    // Build the OUTER propagator (default hybrid) bound to the stochastic trial.
    ptree prop_pt;
    prop_pt.put("name", "prop_stoch");
    prop_pt.put("system", "info0");
    PropagatorFactory<MEM> PropgFac(InfoMap);
    PropgFac.push("prop_stoch", prop_pt);
    auto& prop = PropgFac.getPropagator(mpi, "prop_stoch", wfn, rng_dev);

    RealType dt = 0.01;
    for (int step = 0; step < 3; ++step)
    {
      prop.Propagate(wset, Eshift, dt); // one full hot-path step; inner ensemble resampled once
      prop.Orthogonalize(wset);
      wfn.Energy(wset);
      for (auto it = wset.begin(); it != wset.end(); ++it)
      {
        REQUIRE(std::isfinite(real(it->get_property(WEIGHT))));
        REQUIRE(std::isfinite(real(it->energy())));
        REQUIRE(std::isfinite(imag(it->energy())));
        REQUIRE(std::isfinite(real(it->get_property(OVLP))));
      }
    }
  }
}

TEST_CASE("stochastic_propagator_step", "[wfn_factory][stochastic_wfn]")
{
  auto& mpi = utils::make_unit_test_mpi_context();
  app_log(0, "StochasticWfn end-to-end outer propagator step on a dynamic trial (Phase 3b).");
  using namespace utils;
  run_test_with_files([&]<auto MEM>(std::string hamil_file, std::string wfn_file, WALKER_TYPES) {
    stochastic_propagator_step<MEM>(mpi, hamil_file, wfn_file);
  }, UTEST_HAMIL, UTEST_WFN, TestFiles::RHF | TestFiles::UHF | TestFiles::NOMSD | TestFiles::ALL_SYSTEMS);
}

// Phase 3c-i: walker-conditioned inner sampling. With inner_conditioning = true the inner field paths
// are importance-sampled conditioned on each outer walker phi_w (Eq. 23 of arXiv:2505.18519): the inner
// ensemble is grown to nwalk*inner_nwalkers (block w conditioned on phi_w via the custom force bias
// x_bar(phi_w) = sqrt(dt)*L^var.<phi_T|c+c|phi_w>/<phi_T|phi_w>, built by reusing the inner NOMSD's
// vbias on the OUTER wset). Run a real OUTER AFQMCBasePropagator over the dynamic conditioned trial and
// assert the walkers stay finite. The internal block-structure size checks (inner.size() == nwalk*P) in
// reduce_inner_cross_dm / Log_Overlap validate the nw*P resize. The leapfrog / exact N(phi) cancellation
// is Phase 3c-ii; this is a finiteness smoke, not NOMSD parity.
template<MEMORY_SPACE MEM>
void stochastic_conditioned_propagator_step(std::shared_ptr<utils::mpi_context_t<boost::mpi3::communicator>> mpi,
                                            std::string hamil_file, std::string wfn_file)
{
  if (getWavefunctionType(wfn_file) != NOMSD_WFN)
    return;
  if constexpr (MEM != HOST_MEMORY)
    return; // Phase 3c-i conditioned sampling is CPU-only this phase (full-G kernels CPU-only).
  else
  {
    const auto info  = read_info_from_wfn(wfn_file, "any");
    const int  NMO   = std::get<0>(info);
    const int  nup   = std::get<1>(info);
    const int  ndown = std::get<2>(info);
    WALKER_TYPES type = afqmc::getWalkerType(wfn_file, "any");
    if (type != CLOSED)
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

    const int nwalk = 11;
    const int inner_nwalkers = 4;
    std::shared_ptr<utils::RandomGenerator_t<>> rng = std::make_shared<utils::RandomGenerator_t<>>();
    std::shared_ptr<utils::RandomGenerator_t<MEM>> rng_dev =
        std::make_shared<utils::RandomGenerator_t<MEM>>(utils::make_rng<MEM>(13));
    ptree wlk_pt;
    wlk_pt.put("name", "wset0");
    wlk_pt.put("walker_type", walkerTypeToString(type));

    WavefunctionFactory<MEM> WfnFac(InfoMap);
    ptree pt;
    pt.put("name", "wfn_stoch_cond");
    pt.put("system", "info0");
    pt.put("filename", wfn_file);
    pt.put("stochastic", true);
    pt.put("inner_nwalkers", inner_nwalkers);
    pt.put("inner_nsteps", 1);
    pt.put("inner_conditioning", true);
    ptree inner_prop;
    inner_prop.put("timestep", 0.01);
    pt.put_child("inner_propagator", inner_prop);
    WfnFac.push("wfn_stoch_cond", pt);
    auto& wfn = WfnFac.getWavefunction(mpi, "wfn_stoch_cond", type, &ham, nwalk);
    WfnFac.maybe_initialize_stochastic_inner_walkers(wfn, "wfn_stoch_cond", type, wlk_pt);

    auto wset = make_WalkerSet<MEM>(mpi, wlk_pt, InfoMap["info0"], rng);
    wset.resize(nwalk, WfnFac.getInitialGuess("wfn_stoch_cond"));

    // Prime overlaps/energies and pick an energy shift so the hybrid weights stay well-scaled.
    wfn.Log_Overlap(wset);
    wfn.Energy(wset);
    ComplexType eav(0.0), ow(0.0);
    for (auto it = wset.begin(); it != wset.end(); ++it)
    {
      eav += it->get_property(WEIGHT) * it->energy();
      ow += it->get_property(WEIGHT);
    }
    RealType Eshift = (std::abs(ow) > 1e-12) ? real(eav / ow) : RealType(0);

    // Build the OUTER propagator (default hybrid) bound to the conditioned stochastic trial.
    ptree prop_pt;
    prop_pt.put("name", "prop_stoch_cond");
    prop_pt.put("system", "info0");
    PropagatorFactory<MEM> PropgFac(InfoMap);
    PropgFac.push("prop_stoch_cond", prop_pt);
    auto& prop = PropgFac.getPropagator(mpi, "prop_stoch_cond", wfn, rng_dev);

    RealType dt = 0.01;
    for (int step = 0; step < 3; ++step)
    {
      prop.Propagate(wset, Eshift, dt); // hot-path step; inner ensemble resampled (nw*P, conditioned)
      prop.Orthogonalize(wset);
      wfn.Energy(wset);
      for (auto it = wset.begin(); it != wset.end(); ++it)
      {
        REQUIRE(std::isfinite(real(it->get_property(WEIGHT))));
        REQUIRE(std::isfinite(real(it->energy())));
        REQUIRE(std::isfinite(imag(it->energy())));
        REQUIRE(std::isfinite(real(it->get_property(OVLP))));
      }
    }
  }
}

TEST_CASE("stochastic_conditioned_propagator_step", "[wfn_factory][stochastic_wfn]")
{
  auto& mpi = utils::make_unit_test_mpi_context();
  app_log(0, "StochasticWfn walker-conditioned inner sampling over a real outer propagator (Phase 3c-i).");
  using namespace utils;
  run_test_with_files([&]<auto MEM>(std::string hamil_file, std::string wfn_file, WALKER_TYPES) {
    stochastic_conditioned_propagator_step<MEM>(mpi, hamil_file, wfn_file);
  }, UTEST_HAMIL, UTEST_WFN, TestFiles::RHF | TestFiles::UHF | TestFiles::NOMSD | TestFiles::ALL_SYSTEMS);
}

// Phase 3b-var regression anchor: the stochastic inner (Variational) stack can be built from a
// SEPARATE Hamiltonian named by the `inner_hamiltonian` input key, which the WavefunctionFactory
// builds on demand through the HamiltonianFactory passed to its two-argument constructor. Two trials
// are built in one factory: wfn_clone (no inner_hamiltonian -> inner stack clones the True Ham, the
// pre-3b-var path) and wfn_hvar (inner_hamiltonian = the SAME integral file -> factory builds a second
// Hamiltonian and uses it for the inner stack). With identical inputs (so identical inner_seed) and
// identical integrals, running the dynamic path (inner_nsteps = 1, so the inner Ham actually drives
// B_T) must give identical Energy/Log_Overlap/vbias. The has_input checks prove the factory actually
// traversed the inner_hamiltonian path (built + registered the second Ham) rather than silently
// ignoring the key. (Proving that a *different* Variational Ham changes B_T is the deferred research
// validation -- it needs a variational HDF5 fixture that does not yet exist in the repo.)
template<MEMORY_SPACE MEM>
void stochastic_inner_hamiltonian_same_as_true(std::shared_ptr<utils::mpi_context_t<boost::mpi3::communicator>> mpi,
                                               std::string hamil_file, std::string wfn_file)
{
  if (getWavefunctionType(wfn_file) != NOMSD_WFN)
    return;
  if constexpr (MEM != HOST_MEMORY)
    return; // Phase 3b un-rotated full-G kernels are CPU-only this phase.
  else
  {
    const auto info  = read_info_from_wfn(wfn_file, "any");
    const int  NMO   = std::get<0>(info);
    const int  nup   = std::get<1>(info);
    const int  ndown = std::get<2>(info);
    WALKER_TYPES type = afqmc::getWalkerType(wfn_file, "any");
    if (type != CLOSED)
      return;
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

    const int nwalk = 11;
    ptree wlk_pt;
    wlk_pt.put("name", "wset0");
    wlk_pt.put("walker_type", walkerTypeToString(type));

    // The two-argument WavefunctionFactory constructor wires in HamFac so the factory can build the
    // inner (Variational) Hamiltonian on demand.
    WavefunctionFactory<MEM> WfnFac(InfoMap, HamFac);
    auto build_pt = [&](std::string id, bool with_inner_ham) {
      ptree pt;
      pt.put("name", id);
      pt.put("system", "info0");
      pt.put("filename", wfn_file);
      pt.put("stochastic", true);
      pt.put("inner_nwalkers", 4);
      pt.put("inner_nsteps", 1);
      ptree inner_prop;
      inner_prop.put("timestep", 0.01);
      pt.put_child("inner_propagator", inner_prop);
      if (with_inner_ham)
      {
        ptree inner_ham_block;
        inner_ham_block.put("filename", hamil_file); // same integrals as the True Ham
        pt.put_child("inner_hamiltonian", inner_ham_block);
      }
      return pt;
    };

    WfnFac.push("wfn_clone", build_pt("wfn_clone", false));
    WfnFac.push("wfn_hvar", build_pt("wfn_hvar", true));
    auto& wfn_clone = WfnFac.getWavefunction(mpi, "wfn_clone", type, &ham, nwalk);
    auto& wfn_hvar  = WfnFac.getWavefunction(mpi, "wfn_hvar", type, &ham, nwalk);
    WfnFac.maybe_initialize_stochastic_inner_walkers(wfn_clone, "wfn_clone", type, wlk_pt);
    WfnFac.maybe_initialize_stochastic_inner_walkers(wfn_hvar, "wfn_hvar", type, wlk_pt);

    // The factory built (and registered) a second Ham for the inner_hamiltonian trial only.
    REQUIRE(HamFac.has_input("wfn_hvar__inner_hamiltonian__"));
    REQUIRE_FALSE(HamFac.has_input("wfn_clone__inner_hamiltonian__"));

    std::shared_ptr<utils::RandomGenerator_t<>> rng_a = std::make_shared<utils::RandomGenerator_t<>>();
    std::shared_ptr<utils::RandomGenerator_t<>> rng_b = std::make_shared<utils::RandomGenerator_t<>>();
    auto wset_a = make_WalkerSet<MEM>(mpi, wlk_pt, InfoMap["info0"], rng_a);
    auto wset_b = make_WalkerSet<MEM>(mpi, wlk_pt, InfoMap["info0"], rng_b);
    wset_a.resize(nwalk, WfnFac.getInitialGuess("wfn_clone"));
    wset_b.resize(nwalk, WfnFac.getInitialGuess("wfn_hvar"));
    perturb_stochastic_walkers<MEM>(wset_a, type, NMO, nup, ndown); // deterministic -> wset_a == wset_b
    perturb_stochastic_walkers<MEM>(wset_b, type, NMO, nup, ndown);

    const double tol = 1e-9;
    for (int step = 0; step < 3; ++step)
    {
      // Same hot-path order as stochastic_dynamic_ensemble_smoke, lockstep on both trials.
      wfn_clone.begin_inner_step();
      wfn_hvar.begin_inner_step();
      memory::array<MEM, ComplexType, 2> Xa(nwalk, wfn_clone.number_of_cholesky_vectors());
      memory::array<MEM, ComplexType, 2> Xb(nwalk, wfn_hvar.number_of_cholesky_vectors());
      wfn_clone.vbias(wset_a, Xa, dt); // first reduction -> resamples inner ensemble
      wfn_hvar.vbias(wset_b, Xb, dt);
      wfn_clone.Energy(wset_a);
      wfn_hvar.Energy(wset_b);
      wfn_clone.Log_Overlap(wset_a);
      wfn_hvar.Log_Overlap(wset_b);

      nda::array<ComplexType, 1> ova(nwalk), e1a(nwalk), exxa(nwalk), eja(nwalk);
      nda::array<ComplexType, 1> ovb(nwalk), e1b(nwalk), exxb(nwalk), ejb(nwalk);
      wset_a.getProperty(OVLP, ova); wset_a.getProperty(E1_, e1a);
      wset_a.getProperty(EXX_, exxa); wset_a.getProperty(EJ_, eja);
      wset_b.getProperty(OVLP, ovb); wset_b.getProperty(E1_, e1b);
      wset_b.getProperty(EXX_, exxb); wset_b.getProperty(EJ_, ejb);
      auto Xa_h = nda::to_host(Xa);
      auto Xb_h = nda::to_host(Xb);
      for (int w = 0; w < nwalk; ++w)
      {
        REQUIRE(std::abs(ova(w) - ovb(w)) < tol);
        REQUIRE(std::abs(e1a(w) - e1b(w)) < tol);
        REQUIRE(std::abs(exxa(w) - exxb(w)) < tol);
        REQUIRE(std::abs(eja(w) - ejb(w)) < tol);
      }
      for (int w = 0; w < nwalk; ++w)
        for (int g = 0; g < Xa_h.extent(1); ++g)
          REQUIRE(std::abs(Xa_h(w, g) - Xb_h(w, g)) < tol);
    }
  }
}

TEST_CASE("stochastic_inner_hamiltonian_same_as_true", "[wfn_factory][stochastic_wfn]")
{
  auto& mpi = utils::make_unit_test_mpi_context();
  app_log(0, "StochasticWfn inner_hamiltonian (same integral file) reproduces the clone path (Phase 3b-var).");
  using namespace utils;
  run_test_with_files([&]<auto MEM>(std::string hamil_file, std::string wfn_file, WALKER_TYPES) {
    stochastic_inner_hamiltonian_same_as_true<MEM>(mpi, hamil_file, wfn_file);
  }, UTEST_HAMIL, UTEST_WFN, TestFiles::RHF | TestFiles::UHF | TestFiles::NOMSD | TestFiles::ALL_SYSTEMS);
}


} // namespace sfqmc
