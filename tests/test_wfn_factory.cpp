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
#include <fstream>
#include <filesystem>
#include <format>
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

  ptree ham_pt;
  ham_pt.put("name","ham0");
  ham_pt.put("filename",hamil_file);

  HamiltonianFactory HamFac;
  HamFac.push("ham0", ham_pt); 
  Hamiltonian& ham = HamFac.getHamiltonian(mpi, "ham0");

  int nwalk = 11; // choose prime number to force non-trivial splits in shared routines
  std::shared_ptr<utils::RandomGenerator_t<>> rng = std::make_shared<utils::RandomGenerator_t<>>();

  ptree wlk_pt;
  wlk_pt.put("name","wset0");
  wlk_pt.put("walker_type", walkerTypeToString(type));
  wlk_pt.put("finite_temperature", finiteT);

  ptree wfn_pt;
  wfn_pt.put("name","wfn0");
  wfn_pt.put("filename",wfn_file);
  wfn_pt.put("dense_trial",dense_trial);

  WavefunctionFactory<MEM> WfnFac{};
  WfnFac.push("wfn0", wfn_pt);
  auto& wfn = WfnFac.getWavefunction(mpi, "wfn0", type, finiteT, &ham, nwalk);

  //nwalk=nw;
  auto wset = [&]() {
    if(!finiteT)
    {
      auto const& initial_guess = WfnFac.getInitialGuess("wfn0");
      REQUIRE(int(initial_guess.size()) == nspin);
      REQUIRE(initial_guess[0].shape() == std::array<long,2>{npol*NMO,nup});
      return WalkerSet<MEM>(mpi, wlk_pt, rng, type, initial_guess, nwalk);
    }
    else
    {
      auto initial_guess_ft = WfnFac.getInitialGuess_ft("wfn0");
      REQUIRE(initial_guess_ft.shape() == std::array<long,4>{3,nspin,npol*NMO,NMO});
      return WalkerSet<MEM>(mpi, wlk_pt, rng, type, initial_guess_ft, nwalk);
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
          nda::tensor::add(p, "ijk", SM, "ijk");
        }
      }
    }
  }

  // Overlap
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

  Time.reset();
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

namespace {
void mark_stochastic_wfn_input(ptree& pt) { pt.put("type", "stochasticwfn"); }

template<MEMORY_SPACE MEM>
void perturb_stochastic_walkers(WalkerSet<MEM>& wset, WALKER_TYPES type, int NMO, int nup, int ndown)
{
  const int nspin = (type == COLLINEAR) ? 2 : 1;
  const int npol  = (type == NONCOLLINEAR) ? 2 : 1;
  const int nwalk = wset.size();
  std::array<int, 2> nels = {nup, ndown};
  for (int spin = 0; spin < nspin; spin++)
  {
    nda::array<ComplexType, 1> p_h(long(nwalk) * npol * NMO * nels[spin]);
    for (long k = 0; k < p_h.size(); ++k)
    {
      double v = 0.1 * (k + 1);
      p_h[k] = {std::cos(v), std::sin(v * v)};
    }
    memory::array<MEM, ComplexType, 3> p(reshape(p_h, nwalk, npol * NMO, nels[spin]));
    auto SM = wset.SlaterMatrices(static_cast<SpinTypes>(spin));
    nda::tensor::add(p, "ijk", SM, "ijk");
  }
}
} // namespace

// inner_hamiltonian factory anchor: the stochastic inner (Variational) stack can be built from a separate
// Hamiltonian named by the `inner_hamiltonian` input key, which the WavefunctionFactory builds on demand
// through the HamiltonianFactory passed to its two-argument constructor. Two trials are built in one
// factory: wfn_clone (no inner_hamiltonian -> inner stack clones the True Ham) and wfn_hvar
// (inner_hamiltonian = the SAME integral file -> factory builds a second Hamiltonian and uses it for the
// inner stack). With identical inputs (so identical inner_seed) and identical integrals, running the
// dynamic path (inner_nsteps = 1, so the inner Ham actually drives
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
    return; // Un-rotated full-G kernels are CPU-only today.
  else
  {
    const auto info  = read_info_from_wfn(wfn_file, "any");
    const int  NMO   = std::get<0>(info);
    const int  nup   = std::get<1>(info);
    const int  ndown = std::get<2>(info);
    WALKER_TYPES type = afqmc::getWalkerType(wfn_file, "any");
    if (not utils::dynamic_inner_supports(type))
      return; // dynamic inner ensemble: CLOSED/COLLINEAR only
    if (type != CLOSED)
      return;
    const double dt(0.01);

    ptree ham_pt;
    ham_pt.put("name", "ham0");
    ham_pt.put("filename", hamil_file);
    HamiltonianFactory HamFac;
    HamFac.push("ham0", ham_pt);
    Hamiltonian& ham = HamFac.getHamiltonian(mpi, "ham0");

    const int nwalk = 11;
    ptree wlk_pt;
    wlk_pt.put("name", "wset0");
    wlk_pt.put("walker_type", walkerTypeToString(type));

    // The two-argument WavefunctionFactory constructor wires in HamFac so the factory can build the
    // inner (Variational) Hamiltonian on demand.
    WavefunctionFactory<MEM> WfnFac(HamFac);
    // Copy the fixture and stamp the trained-timestep attribute onto it (fixtures are shared and must
    // not be modified in place).
    const std::string stamped_hamil = "stamped_inner_hamil.h5";
    // MATCHES the value the clone arm sets by hand: this test's premise is that the two arms are the
    // same calculation, so they must share a timestep. That the stamp is what DRIVES the hvar arm is
    // asserted separately, with a distinctive value, in stochastic_inner_timestep_comes_from_the_stamp.
    const double stamped_dt = 0.01;
    std::filesystem::remove(stamped_hamil);
    std::filesystem::copy_file(hamil_file, stamped_hamil);
    {
      h5::file fh5(stamped_hamil, 'a');
      h5::group grp(fh5);
      h5::group hgrp = grp.open_group("Hamiltonian");
      h5::h5_write_attribute(hgrp, "inner_timestep", stamped_dt);
    }

    auto build_pt = [&](std::string id, bool with_inner_ham) {
      ptree pt;
      pt.put("name", id);
      pt.put("filename", wfn_file);
      mark_stochastic_wfn_input(pt);
      pt.put("inner_n_samples", 4);
      pt.put("inner_nsteps", 1);
      pt.put("inner_sampling_target", "gaussian"); // dynamic trials must name a mode; the bare draw suffices here
      if (not with_inner_ham)
      {
        // ONLY the no-inner_hamiltonian arm sets the timestep by hand -- that is the one configuration
        // where the input is still the source. Setting it alongside inner_hamiltonian is now a hard
        // error (it would be silently overwritten by the stamp), and setting it here for BOTH arms is
        // what previously let the factory's stamp-to-ptree store be dead without any test noticing.
        ptree inner_prop;
        inner_prop.put("timestep", 0.01);
        pt.put_child("inner_propagator", inner_prop);
      }
      if (with_inner_ham)
      {
        ptree inner_ham_block;
        // A STAMPED copy, not the bare fixture. inner_hamiltonian is how a trained trial supplies its
        // variational Hamiltonian, and SAFIRE now reads the trained B_T timestep from that file's
        // Hamiltonian/inner_timestep attribute rather than from the input (there is no default, so a
        // missing stamp is a hard error). Pointing at an unstamped fixture would test a configuration
        // production can no longer have; copying and stamping exercises the real contract.
        inner_ham_block.put("filename", stamped_hamil);
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

    // Both arms are driving B_T with the same dt -- the precondition for everything compared below.
    CHECK_THAT(wfn_hvar.stochastic_inner_timestep(), utils::Approx(stamped_dt));
    CHECK_THAT(wfn_clone.stochastic_inner_timestep(), utils::Approx(0.01));

    std::shared_ptr<utils::RandomGenerator_t<>> rng_a = std::make_shared<utils::RandomGenerator_t<>>();
    std::shared_ptr<utils::RandomGenerator_t<>> rng_b = std::make_shared<utils::RandomGenerator_t<>>();
    auto const& initial_guess_a = WfnFac.getInitialGuess("wfn_clone");
    auto const& initial_guess_b = WfnFac.getInitialGuess("wfn_hvar");
    auto wset_a = WalkerSet<MEM>(mpi, wlk_pt, rng_a, type, initial_guess_a, nwalk);
    auto wset_b = WalkerSet<MEM>(mpi, wlk_pt, rng_b, type, initial_guess_b, nwalk);
    perturb_stochastic_walkers<MEM>(wset_a, type, NMO, nup, ndown); // deterministic -> wset_a == wset_b
    perturb_stochastic_walkers<MEM>(wset_b, type, NMO, nup, ndown);

    const double tol = 1e-9;
    for (int step = 0; step < 3; ++step)
    {
      // Same hot-path order as stochastic_dynamic_ensemble_smoke, lockstep on both trials.
      wfn_clone.begin_inner_step(wset_a);
      wfn_hvar.begin_inner_step(wset_b);
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

TEST_CASE("stochastic_inner_hamiltonian_same_as_true", "[wfn_factory]")
{
  auto& mpi = utils::make_unit_test_mpi_context();
  app_log(0, "StochasticWfn inner_hamiltonian (same integral file) reproduces the clone path.");
  using namespace utils;
  run_test_with_files([&]<auto MEM>(std::string hamil_file, std::string wfn_file, WALKER_TYPES, bool finiteT) {
    stochastic_inner_hamiltonian_same_as_true<MEM>(mpi, hamil_file, wfn_file);
  }, UTEST_HAMIL, UTEST_WFN, TestFiles::DYNAMIC_INNER);
}

// -----------------------------------------------------------------------------------------------------
// WHERE THE TRAINED TIMESTEP COMES FROM.
//
// dt = ts_v^2 parameterizes B_T built from the variational Hamiltonian, so the two travel together in
// one file: export_safire stamps Hamiltonian/inner_timestep and the factory reads it. The input key is
// gone. Three things have to hold, and only the first is about reading the file:
//   (1) the stamped value REACHES the wavefunction -- a factory that reads the attribute into a ptree
//       nobody consumes looks identical to one that works, so this asserts the built object's own dt,
//       against a value that matches no default and no other arm of any test;
//   (2) an unstamped inner_hamiltonian is a hard error, not a silent fallback to some default;
//   (3) a hand-set inner_propagator.timestep alongside inner_hamiltonian is REFUSED, not overwritten --
//       a deck that sets it must fail rather than run with a number it did not ask for.
// -----------------------------------------------------------------------------------------------------
template<MEMORY_SPACE MEM>
void stochastic_inner_timestep_comes_from_the_stamp(
    std::shared_ptr<utils::mpi_context_t<boost::mpi3::communicator>> mpi, std::string hamil_file,
    std::string wfn_file)
{
  if (getWavefunctionType(wfn_file) != NOMSD_WFN)
    return;
  if constexpr (MEM != HOST_MEMORY)
    return; // Un-rotated full-G kernels are CPU-only today.
  else
  {
    WALKER_TYPES type = afqmc::getWalkerType(wfn_file, "any");
    if (not utils::dynamic_inner_supports(type))
      return; // dynamic inner ensemble: CLOSED/COLLINEAR only
    if (type != CLOSED)
      return;

    ptree ham_pt;
    ham_pt.put("name", "ham0");
    ham_pt.put("filename", hamil_file);
    HamiltonianFactory HamFac;
    HamFac.push("ham0", ham_pt);
    Hamiltonian& ham = HamFac.getHamiltonian(mpi, "ham0");

    // A value that is neither the retired input default nor any other test's dt: if this comes back
    // out of the wavefunction, it can only have come through the attribute.
    const double stamped_dt         = 0.0123;
    const std::string stamped_ham   = "timestep_stamp_hamil.h5";
    const std::string unstamped_ham = "timestep_unstamped_hamil.h5";
    for (auto const& f : {stamped_ham, unstamped_ham})
    {
      std::filesystem::remove(f);
      std::filesystem::copy_file(hamil_file, f); // fixtures are shared; never stamp them in place
    }
    {
      h5::file fh5(stamped_ham, 'a');
      h5::group grp(fh5);
      h5::group hgrp = grp.open_group("Hamiltonian");
      h5::h5_write_attribute(hgrp, "inner_timestep", stamped_dt);
    }

    auto build_pt = [&](std::string id, std::string inner_ham_file) {
      ptree pt;
      pt.put("name", id);
      pt.put("filename", wfn_file);
      mark_stochastic_wfn_input(pt);
      pt.put("inner_n_samples", 4);
      pt.put("inner_nsteps", 1);
      pt.put("inner_sampling_target", "gaussian");
      ptree inner_ham_block;
      inner_ham_block.put("filename", inner_ham_file);
      pt.put_child("inner_hamiltonian", inner_ham_block);
      return pt; // NOTE: no inner_propagator -- production cannot supply one here
    };

    WavefunctionFactory<MEM> WfnFac(HamFac);
    const int nwalk = 4;

    // (1) the stamp drives the built object
    WfnFac.push("wfn_stamped", build_pt("wfn_stamped", stamped_ham));
    auto& wfn = WfnFac.getWavefunction(mpi, "wfn_stamped", type, &ham, nwalk);
    CHECK_THAT(wfn.stochastic_inner_timestep(), utils::Approx(stamped_dt));

    // (2) no stamp is fatal
    WfnFac.push("wfn_unstamped", build_pt("wfn_unstamped", unstamped_ham));
    REQUIRE_THROWS_AS(WfnFac.getWavefunction(mpi, "wfn_unstamped", type, &ham, nwalk), AppAbortException);

    // (3) a hand-set timestep next to inner_hamiltonian is refused, not silently overwritten
    ptree both = build_pt("wfn_both", stamped_ham);
    ptree inner_prop;
    inner_prop.put("timestep", 0.05);
    both.put_child("inner_propagator", inner_prop);
    WfnFac.push("wfn_both", both);
    REQUIRE_THROWS_AS(WfnFac.getWavefunction(mpi, "wfn_both", type, &ham, nwalk), AppAbortException);
  }
}

TEST_CASE("stochastic_inner_timestep_comes_from_the_stamp", "[wfn_factory]")
{
  auto& mpi = utils::make_unit_test_mpi_context();
  app_log(0, "Trained inner timestep is read from inner_hamiltonian's stamp, and only from there.");
  using namespace utils;
  run_test_with_files([&]<auto MEM>(std::string hamil_file, std::string wfn_file, WALKER_TYPES, bool finiteT) {
    stochastic_inner_timestep_comes_from_the_stamp<MEM>(mpi, hamil_file, wfn_file);
  }, UTEST_HAMIL, UTEST_WFN, TestFiles::DYNAMIC_INNER);
}

// type: stochasticwfn builds StochasticWfn via WavefunctionFactory.
template<MEMORY_SPACE MEM>
void wfn_factory_stochasticwfn_type_smoke(
    std::shared_ptr<utils::mpi_context_t<boost::mpi3::communicator>> mpi, std::string hamil_file,
    std::string wfn_file)
{
  if (getWavefunctionType(wfn_file) != NOMSD_WFN)
    return;
  if constexpr (MEM != HOST_MEMORY)
    return;
  else
  {
    WALKER_TYPES type = afqmc::getWalkerType(wfn_file, "any");
    if (type != CLOSED)
      return;

    ptree ham_pt;
    ham_pt.put("name", "ham0");
    ham_pt.put("filename", hamil_file);
    HamiltonianFactory HamFac;
    HamFac.push("ham0", ham_pt);
    Hamiltonian& ham = HamFac.getHamiltonian(mpi, "ham0");

    ptree wlk_pt;
    wlk_pt.put("name", "wset0");
    wlk_pt.put("walker_type", walkerTypeToString(type));

    WavefunctionFactory<MEM> WfnFac{};
    ptree wfn_pt;
    wfn_pt.put("name", "wfn_stoch");
    wfn_pt.put("filename", wfn_file);
    mark_stochastic_wfn_input(wfn_pt);
    wfn_pt.put("inner_n_samples", 1);
    WfnFac.push("wfn_stoch", wfn_pt);
    auto& wfn_stoch = WfnFac.getWavefunction(mpi, "wfn_stoch", type, &ham, 4);
    REQUIRE(wfn_stoch.is_stochastic_wavefunction());
    WfnFac.maybe_initialize_stochastic_inner_walkers(wfn_stoch, "wfn_stoch", type, wlk_pt);
    REQUIRE(wfn_stoch.stochastic_inner_walkers_initialized());
  }
}

TEST_CASE("wfn_factory: stochasticwfn", "[wfn_factory]")
{
  auto& mpi = utils::make_unit_test_mpi_context();
  app_log(0, "type: stochasticwfn builds StochasticWfn via WavefunctionFactory.");
  using namespace utils;
  run_test_with_files([&]<auto MEM>(std::string hamil_file, std::string wfn_file, WALKER_TYPES, bool finiteT) {
    wfn_factory_stochasticwfn_type_smoke<MEM>(mpi, hamil_file, wfn_file);
  }, UTEST_HAMIL, UTEST_WFN, TestFiles::RHF | TestFiles::UHF | TestFiles::NOMSD | TestFiles::ALL_SYSTEMS);
}

// Wavefunction/StochasticWfn HDF5 marker is detected and builds StochasticWfn without input type.
template<MEMORY_SPACE MEM>
void stochastic_hdf5_type_smoke(std::shared_ptr<utils::mpi_context_t<boost::mpi3::communicator>> mpi,
                                std::string hamil_file, std::string wfn_file)
{
  if (getWavefunctionType(wfn_file) != NOMSD_WFN)
    return;
  if constexpr (MEM != HOST_MEMORY)
    return;
  else
  {
    WALKER_TYPES type = afqmc::getWalkerType(wfn_file, "any");
    if (type != CLOSED)
      return;

    const std::string marked = "stoch_wfn_marked.h5";
    if (mpi->comm.root())
    {
      std::filesystem::copy_file(wfn_file, marked, std::filesystem::copy_options::overwrite_existing);
      h5::file file(marked, 'a');
      h5::group grp(file);
      h5::group wgrp = grp.open_group("Wavefunction");
      wgrp.create_group("StochasticWfn");
    }
    mpi->comm.barrier();

    REQUIRE(getWavefunctionType(marked) == STOCHASTIC_WFN);

    ptree ham_pt;
    ham_pt.put("name", "ham0");
    ham_pt.put("filename", hamil_file);
    HamiltonianFactory HamFac;
    HamFac.push("ham0", ham_pt);
    Hamiltonian& ham = HamFac.getHamiltonian(mpi, "ham0");

    ptree wlk_pt;
    wlk_pt.put("name", "wset0");
    wlk_pt.put("walker_type", walkerTypeToString(type));

    WavefunctionFactory<MEM> WfnFac{};
    ptree wfn_pt;
    wfn_pt.put("name", "wfn_marked");
    wfn_pt.put("filename", marked);
    WfnFac.push("wfn_marked", wfn_pt);
    auto& wfn = WfnFac.getWavefunction(mpi, "wfn_marked", type, &ham, 4);
    REQUIRE(wfn.is_stochastic_wavefunction());
    WfnFac.maybe_initialize_stochastic_inner_walkers(wfn, "wfn_marked", type, wlk_pt);
    REQUIRE(wfn.stochastic_inner_walkers_initialized());

    if (mpi->comm.root())
      std::remove(marked.c_str());
    mpi->comm.barrier();
  }
}

TEST_CASE("stochastic_hdf5_type_smoke", "[wfn_factory]")
{
  auto& mpi = utils::make_unit_test_mpi_context();
  app_log(0, "Wavefunction/StochasticWfn HDF5 marker builds StochasticWfn.");
  using namespace utils;
  run_test_with_files([&]<auto MEM>(std::string hamil_file, std::string wfn_file, WALKER_TYPES, bool finiteT) {
    stochastic_hdf5_type_smoke<MEM>(mpi, hamil_file, wfn_file);
  }, UTEST_HAMIL, UTEST_WFN, TestFiles::RHF | TestFiles::UHF | TestFiles::NOMSD | TestFiles::ALL_SYSTEMS);
}

} // namespace sfqmc
