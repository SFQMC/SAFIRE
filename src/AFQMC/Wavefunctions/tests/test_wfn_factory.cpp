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

//#undef NDEBUG

#include "catch_amalgamated.hpp"

#include "config.h"
#include "Utilities/AppAbort.hpp"

#include "io/ptree/ptree_utilities.hpp"
#include "hdf/hdf_archive.h"
#include "Utilities/Random.hpp"
#include "Utilities/app_loggers.h"

#include <string>
#include <vector>
#include <complex>
#include <iomanip>
#include <random>

#include "Utilities/Timer.hpp"
#include "AFQMC/Utilities/test_utils.hpp"
#include "AFQMC/Utilities/readWfn.cpp"
#include "Memory/buffer_managers.h"

#include "AFQMC/Hamiltonians/HamiltonianFactory.h"
#include "AFQMC/Hamiltonians/Hamiltonian.hpp"
#include "AFQMC/Wavefunctions/WavefunctionFactory.h"
#include "AFQMC/Walkers/WalkerSet.hpp"

#include "SparseMatrix/csr_matrix_construct.hpp"
#include "Numerics/ma_blas.hpp"

using std::complex;
using std::ifstream;
using std::string;
using ma::real;
using ma::imag;

extern std::string UTEST_HAMIL, UTEST_WFN;

namespace sfqmc
{
using namespace afqmc;

// Shared ham/walker/factory setup for wavefunction factory unit tests.
// Wavefunction registration is explicit so future first-class StochasticWfn
// tests can use different filenames or factory inputs without changing this.
template<bool MP, class Allocator>
struct WfnTestContext
{
  static constexpr int nwalk = 11;

  boost::mpi3::communicator& world;
  GlobalTaskGroup gTG;
  TaskGroup_ TG;

  TEST_DATA<ComplexType> file_data;
  std::string wfn_type;
  WALKER_TYPES type;
  int NMO;
  int NAEA;
  int NAEB;
  int nspins;
  int npol;

  std::map<std::string, AFQMCInfo> InfoMap;
  HamiltonianFactory HamFac;
  Hamiltonian& ham;
  WavefunctionFactory WfnFac;

  ptree wlk_pt;
  utils::RandomGenerator_t rng;
  Allocator alloc_;

  explicit WfnTestContext(boost::mpi3::communicator& world_in)
      : world(world_in),
        gTG(world_in),
        TG(gTG, std::string("WfnTG"), 1, gTG.getTotalCores()),
        file_data([&]() {
          if (not file_exists(UTEST_HAMIL) || not file_exists(UTEST_WFN))
            APP_ABORT(" Hamiltonian or wavefunction file not found. Run unit test with --hamil /path/to/hamil.h5 and --wfn /path/to/wfn.h5.");
          std::string base_name = UTEST_WFN.substr(UTEST_WFN.find_last_of("\\/") + 1);
          std::string test_wfn  = base_name.substr(0, base_name.find_last_of("."));
          return read_test_results_from_hdf<ComplexType>(UTEST_HAMIL, test_wfn);
        }()),
        wfn_type(afqmc::getWavefunctionType(UTEST_WFN)),
        type(afqmc::getWalkerType(UTEST_WFN, wfn_type)),
        NMO(file_data.NMO),
        NAEA(file_data.NAEA),
        NAEB(file_data.NAEB),
        nspins((type == COLLINEAR) ? 2 : 1),
        npol((type == NONCOLLINEAR) ? 2 : 1),
        InfoMap(),
        HamFac(InfoMap),
        ham([&]() -> Hamiltonian& {
          InfoMap.insert({"info0", AFQMCInfo{"info0", NMO, NAEA, NAEB}});
          ptree ham_pt;
          ham_pt.put("name", "ham0");
          ham_pt.put("system", "info0");
          ham_pt.put("filename", UTEST_HAMIL);
          HamFac.push("ham0", ham_pt);
          return HamFac.getHamiltonian(gTG, "ham0");
        }()),
        WfnFac(InfoMap, MP),
        wlk_pt([&]() {
          ptree pt;
          pt.put("name", "wset0");
          if (type == CLOSED)
            pt.put("walker_type", "closed");
          else if (type == COLLINEAR)
            pt.put("walker_type", "collinear");
          else if (type == NONCOLLINEAR)
            pt.put("walker_type", "noncollinear");
          else if (type == FULLYPOLARIZED)
            pt.put("walker_type", "fullypolarized");
          return pt;
        }()),
        alloc_(make_localTG_allocator<ComplexType>(TG))
  {}

  ptree make_wfn_pt(const std::string& name,
                    const std::string& filename,
                    bool stochastic   = false,
                    int inner_nwalkers = 1) const
  {
    ptree pt;
    pt.put("name", name);
    pt.put("system", "info0");
    pt.put("filename", filename);
    if (stochastic)
    {
      pt.put("stochastic", true);
      pt.put("inner_nwalkers", inner_nwalkers);
    }
    return pt;
  }

  Wavefunction& register_wavefunction(const std::string& id,
                                      const ptree& wfn_pt,
                                      TaskGroup_& TGprop,
                                      TaskGroup_& TGwfn)
  {
    bool stochastic = wfn_pt.get<bool>("stochastic", false);
    ptree const* walker_ptr = stochastic ? &wlk_pt : nullptr;
    WfnFac.push(id, wfn_pt);
    return WfnFac.getWavefunction(TGprop, TGwfn, id, type, &ham, 1e-6, nwalk, walker_ptr);
  }

  Wavefunction& register_wavefunction(const std::string& id, const ptree& wfn_pt)
  {
    return register_wavefunction(id, wfn_pt, TG, TG);
  }

  WalkerSet make_walker_set() { return WalkerSet(TG, wlk_pt, InfoMap.at("info0"), &rng); }

  void init_walkers(WalkerSet& wset, const std::string& wfn_id)
  {
    auto initial_guess = WfnFac.getInitialGuess(wfn_id);
    REQUIRE(initial_guess.size(0) == 2);
    REQUIRE(initial_guess.size(1) == npol * NMO);
    REQUIRE(initial_guess.size(2) == NAEA);
    if (type == COLLINEAR)
      wset.resize(nwalk, initial_guess[0], initial_guess[1](initial_guess.extension(1), {0, NAEB}));
    else
      wset.resize(nwalk, initial_guess[0], initial_guess[0]);
  }

  void maybe_init_model_ham(Wavefunction& wfn, double dt)
  {
    if (wfn.getHamType() != ModelHamiltonian)
      return;
    auto nCV = wfn.local_number_of_cholesky_vectors();
    if (TG.Global().root())
    {
      using SPComplexType = typename to_working_precision<MP, ComplexType>::type;
      boost::multi::array<SPComplexType, 1> vMF_discrete(iextensions<1u>{nCV});
      boost::multi::array<SPComplexType, 1> nMF(iextensions<1u>{2 * NMO});
      for (int i = 0; i < npol * NMO; i++)
        nMF[i] = SPComplexType(0.0, 0.0);
      if (type == COLLINEAR)
        for (int i = 0; i < NMO; i++)
          nMF[i + NMO] = SPComplexType(0.0, 0.0);
      wfn.update_potentials(dt, nMF, vMF_discrete, false);
    }
    TG.Global().barrier();
  }
};

template<bool MP, class Allocator>
void wfn_fac(boost::mpi3::communicator& world)
{
  WfnTestContext<MP, Allocator> ctx(world);
  Wavefunction& wfn = ctx.register_wavefunction("wfn0", ctx.make_wfn_pt("wfn0", UTEST_WFN));
  WalkerSet wset    = ctx.make_walker_set();
  ctx.init_walkers(wset, "wfn0");

  // Overlap
  wfn.Overlap(wset);

  Watch Time;
  Time.reset();

  wfn.Energy(wset);
  ctx.TG.TG_local().barrier();
  if (std::abs(ctx.file_data.E0 + ctx.file_data.E1 + ctx.file_data.E2) > 1e-8)
  {
    for (auto it = wset.begin(); it != wset.end(); ++it)
    {
      REQUIRE(real(ComplexType(*it->E1())) == Approx(real(ctx.file_data.E0 + ctx.file_data.E1)));
      REQUIRE(real(ComplexType(*it->EXX()) + ComplexType(*it->EJ())) == Approx(real(ctx.file_data.E2)));
      REQUIRE(imag(it->energy()) == Approx(imag(ctx.file_data.E0 + ctx.file_data.E1 + ctx.file_data.E2)));
    }
  }
  else
  {
    app_log(1," E: {}", ComplexType(wset[0].energy())); 
    app_log(1," E0+E1: {}", ComplexType(*wset[0].E1()));
    app_log(1," EJ: {}", ComplexType(*wset[0].EJ())); 
    app_log(1," EXX: {}", ComplexType(*wset[0].EXX()));
  }

  auto size_of_G = wfn.size_of_G_for_vbias();
  int Gdim1      = (wfn.transposed_G_for_vbias() ? ctx.nwalk : size_of_G);
  int Gdim2      = (wfn.transposed_G_for_vbias() ? size_of_G : ctx.nwalk);
  using CMatrix = Matrix_<Allocator>;
  CMatrix G({Gdim1, Gdim2}, ctx.alloc_);
  wfn.MixedDensityMatrix_for_vbias(wset, G);

  double dt(0.01);
  auto nCV = wfn.local_number_of_cholesky_vectors();
  CMatrix X({nCV, ctx.nwalk}, ctx.alloc_);

  ctx.maybe_init_model_ham(wfn, dt);

  Time.reset();
  wfn.vbias(G, X, dt);
  ctx.TG.TG_local().barrier();
  ComplexType Xsum = 0;
  if (std::abs(ctx.file_data.Xsum) > 1e-8)
  {
    for (int n = 0; n < ctx.nwalk; n++)
    {
      Xsum = 0;
      for (int i = 0; i < X.size(0); i++)
        Xsum += X[i][n];
      REQUIRE(real(ComplexType(Xsum)) == Approx(real(ctx.file_data.Xsum)));
      REQUIRE(imag(ComplexType(Xsum)) == Approx(imag(ctx.file_data.Xsum)));
    }
  }
  else
  {
    Xsum              = 0;
    ComplexType Xsum2 = 0;
    for (int i = 0; i < X.size(0); i++)
    {
      Xsum += X[i][0];
      Xsum2 += ComplexType(0.5) * X[i][0] * X[i][0];
    }
    app_log(1," Xsum: {}", ComplexType(Xsum));
    app_log(1," Xsum2 (EJ): {}", ComplexType(Xsum2) / dt);
  }

  // spin dependent HS potential?
  // generalize later
  int nx = (wfn.getHamType() == ModelHamiltonian ? ctx.nspins * ctx.npol * ctx.npol : 1);
  int nspin_hst = (wfn.spin_dependent_vHS() ? 2 : 1);
  int vdim1     = (wfn.transposed_vHS() ? nspin_hst * ctx.nwalk : ctx.NMO * ctx.NMO * nx);
  int vdim2     = (wfn.transposed_vHS() ? ctx.NMO * ctx.NMO * nx : nspin_hst * ctx.nwalk);
  if (wfn.getHamType() == ModelHamiltonian) // only sparseP2 is used - denseP2 is hardcoded to never run!
  {
    Time.reset();
    auto [vHS_up, vHS_down] = wfn.vHS_sparse(X, dt); // vHS_sparse lives inside the ModelHamOps class
    ctx.TG.TG_local().barrier();

    // Convert sparse matrices to dense CMatrix objects for easier manipulation
    CMatrix vHS_up_dense({vHS_up->size(0), vHS_up->size(1)}, ctx.alloc_);
    CMatrix vHS_down_dense({vHS_down->size(0), vHS_down->size(1)}, ctx.alloc_);
        
#if defined(ENABLE_DEVICE)
        // For GPU builds, use host-side temporary arrays to avoid direct assignment to device memory
        boost::multi::array<ComplexType, 2> vHS_up_host({vHS_up->size(0), vHS_up->size(1)});
        boost::multi::array<ComplexType, 2> vHS_down_host({vHS_down->size(0), vHS_down->size(1)});
        
        // Initialize host arrays to zero
        std::fill_n(vHS_up_host.origin(), vHS_up_host.num_elements(), ComplexType(0.0));
        std::fill_n(vHS_down_host.origin(), vHS_down_host.num_elements(), ComplexType(0.0));
        
        // Convert sparse to dense using correct sparse matrix API on host
        for (int row = 0; row < static_cast<int>(vHS_up->size(0)); ++row) {
          auto [nnz, vals, cols] = vHS_up->sparse_row(row);
          for (size_t i = 0; i < nnz; ++i) {
            vHS_up_host[row][cols[i]] = vals[i];
          }
        }
        
        // For collinear systems, handle vHS_down if it's different from vHS_up
        // For noncollinear systems, vHS_up and vHS_down point to the same matrix
        if (vHS_up != vHS_down) {
          // COLLINEAR case: fill vHS_down_host separately
          for (int row = 0; row < static_cast<int>(vHS_down->size(0)); ++row) {
            auto [nnz, vals, cols] = vHS_down->sparse_row(row);
            for (size_t i = 0; i < nnz; ++i) {
              vHS_down_host[row][cols[i]] = vals[i];
            }
          }
        } else {
          // NONCOLLINEAR case: copy the same data
          std::copy_n(vHS_up_host.origin(), vHS_up_host.num_elements(), vHS_down_host.origin());
        }
        
        // Copy from host to device
        std::copy_n(vHS_up_host.origin(), vHS_up_host.num_elements(), vHS_up_dense.origin());
        std::copy_n(vHS_down_host.origin(), vHS_down_host.num_elements(), vHS_down_dense.origin());
#else
        // For CPU builds, initialize and directly assign to dense matrices
        std::fill_n(vHS_up_dense.origin(), vHS_up_dense.num_elements(), ComplexType(0.0));
        std::fill_n(vHS_down_dense.origin(), vHS_down_dense.num_elements(), ComplexType(0.0));
        
        // Convert sparse to dense using correct sparse matrix API
        for (int row = 0; row < static_cast<int>(vHS_up->size(0)); ++row) {
          auto [nnz, vals, cols] = vHS_up->sparse_row(row);
          for (size_t i = 0; i < nnz; ++i) {
            vHS_up_dense[row][cols[i]] = vals[i];
          }
        }
        
        // For collinear systems, handle vHS_down if it's different from vHS_up
        // For noncollinear systems, vHS_up and vHS_down point to the same matrix
        if (vHS_up != vHS_down) {
          // COLLINEAR case: fill vHS_down_dense separately
          for (int row = 0; row < static_cast<int>(vHS_down->size(0)); ++row) {
            auto [nnz, vals, cols] = vHS_down->sparse_row(row);
            for (size_t i = 0; i < nnz; ++i) {
              vHS_down_dense[row][cols[i]] = vals[i];
            }
          }
        } else {
          // NONCOLLINEAR case: copy the same data
          std::copy_n(vHS_up_dense.origin(), vHS_up_dense.num_elements(), vHS_down_dense.origin());
        }
#endif
        
        ComplexType Vsum = 0;
        if (std::abs(ctx.file_data.Vsum) > 1e-8)
        {
          for (int n = 0; n < ctx.nwalk; n++)
          {
            Vsum = 0;
            if (wfn.transposed_vHS())
            {
              for (int i = 0; i < vHS_up_dense.size(1); i++)
                Vsum += vHS_up_dense[n][i];
              for (int i = 0; i < vHS_down_dense.size(1); i++)
                Vsum += vHS_down_dense[n][i];
            }
            else
            {
              for (int i = 0; i < vHS_up_dense.size(0); i++)
                Vsum += vHS_up_dense[i][n];
              for (int i = 0; i < vHS_down_dense.size(0); i++)
                Vsum += vHS_down_dense[i][n];
            }
            REQUIRE(real(ComplexType(Vsum)) == Approx(real(ctx.file_data.Vsum)));
            REQUIRE(imag(ComplexType(Vsum)) == Approx(imag(ctx.file_data.Vsum)));
          }
        } else {
          Vsum = 0;
          if (wfn.transposed_vHS())
          {
            for (int i = 0; i < vHS_up_dense.size(1); i++)
              Vsum += vHS_up_dense[0][i];
            for (int i = 0; i < vHS_down_dense.size(1); i++)
              Vsum += vHS_down_dense[0][i];
          }
          else
          {
            for (int i = 0; i < vHS_up_dense.size(0); i++)
              Vsum += vHS_up_dense[i][0];
            for (int i = 0; i < vHS_down_dense.size(0); i++)
              Vsum += vHS_down_dense[i][0];
          }
          app_log(1," Vsum: {}", ComplexType(Vsum));
        }
  } else { // not a model Hamiltonian
    CMatrix vHS({vdim1, vdim2}, ctx.alloc_);
    Time.reset();
    wfn.vHS(X, vHS, dt);

    ctx.TG.TG_local().barrier();
    ComplexType Vsum = 0;
    if (std::abs(ctx.file_data.Vsum) > 1e-8)
    {
      for (int n = 0; n < ctx.nwalk; n++)
      {
        Vsum = 0;
        if (wfn.transposed_vHS())
        {
          for (int i = 0; i < vHS.size(1); i++)
            Vsum += vHS[n][i];
        }
        else
        {
          for (int i = 0; i < vHS.size(0); i++)
            Vsum += vHS[i][n];
        }
        REQUIRE(real(ComplexType(Vsum)) == Approx(real(ctx.file_data.Vsum)));
        REQUIRE(imag(ComplexType(Vsum)) == Approx(imag(ctx.file_data.Vsum)));
      }
    }
    else
    {
      Vsum = 0;
      if (wfn.transposed_vHS())
      {
        for (int i = 0; i < vHS.size(1); i++)
          Vsum += vHS[0][i];
      }
      else
      {
        for (int i = 0; i < vHS.size(0); i++)
          Vsum += vHS[i][0];
      }
      app_log(1," Vsum: {}", ComplexType(Vsum));
    }
  }
  return;

      /*
      // Restarting Wavefunction from file
      ptree wfn_pt2;
      wfn_pt2.put("name","wfn1");
      wfn_pt2.put("system","info0");
      wfn_pt2.put("filename","./dummy.h5");

      WfnFac.push("wfn1", wfn_pt2);
      Wavefunction& wfn2 = WfnFac.getWavefunction(TG, TG, "wfn1", type, nullptr, 1e-6, nwalk);

      WalkerSet wset2(TG, wlk_pt, InfoMap["info0"], &rng);
      //auto initial_guess = WfnFac.getInitialGuess("wfn0");
      REQUIRE(initial_guess.size(0) == 2);
      REQUIRE(initial_guess.size(1) == npol * NMO);
      REQUIRE(initial_guess.size(2) == NAEA);

      if (type == COLLINEAR)
        wset2.resize(nwalk, initial_guess[0], initial_guess[1](initial_guess.extension(1), {0, NAEB}));
      else
        wset2.resize(nwalk, initial_guess[0], initial_guess[0]);

      wfn2.Overlap(wset2);
      for (auto it = wset2.begin(); it != wset2.end(); ++it)
      {
        REQUIRE(real(ComplexType(*it->overlap())) == Approx(1.0));
        REQUIRE(imag(ComplexType(*it->overlap())) == Approx(0.0));
      }

      wfn2.Energy(wset2);
      if (std::abs(file_data.E0 + file_data.E1 + file_data.E2) > 1e-8)
      {
        for (auto it = wset2.begin(); it != wset2.end(); ++it)
        {
          REQUIRE(real(ComplexType(*it->E1())) == Approx(real(file_data.E0 + file_data.E1)));
          REQUIRE(real(*it->EXX() + *it->EJ()) == Approx(real(file_data.E2)));
          REQUIRE(imag(it->energy()) == Approx(imag(file_data.E0 + file_data.E1 + file_data.E2)));
        }
      }
      else
      {
        app_log(1," E: {}", ComplexType(wset[0].energy())); 
        app_log(1," E0+E1: {}", ComplexType(*wset[0].E1()));
        app_log(1," EJ: {}", ComplexType(*wset[0].EJ())); 
        app_log(1," EXX: {}", ComplexType(*wset[0].EXX())); 
      }

      REQUIRE(size_of_G == wfn2.size_of_G_for_vbias());
      wfn2.MixedDensityMatrix_for_vbias(wset2, G);
      REQUIRE(nCV == wfn2.local_number_of_cholesky_vectors());
      wfn2.vbias(G, X, dt);
      Xsum = 0;
      if (std::abs(file_data.Xsum) > 1e-8)
      {
        for (int n = 0; n < nwalk; n++)
        {
          Xsum = 0;
          for (int i = 0; i < X.size(0); i++)
            Xsum += X[i][n];
          REQUIRE(real(ComplexType(Xsum)) == Approx(real(file_data.Xsum)));
          REQUIRE(imag(ComplexType(Xsum)) == Approx(imag(file_data.Xsum)));
        }
      }
      else
      {
        Xsum = 0;
        ComplexType Xsum2(0.0);
        for (int i = 0; i < X.size(0); i++)
        {
          Xsum += X[i][0];
          Xsum2 += ComplexType(0.5) * X[i][0] * X[i][0];
        }
        app_log(1," Xsum: {}", ComplexType(Xsum)); 
        app_log(1," Xsum2 (EJ): {}", ComplexType(Xsum2) / dt);
      }

      wfn2.vHS(X, vHS, dt);
      TG.TG_local().barrier();
      Vsum = 0;
      if (std::abs(file_data.Vsum) > 1e-8)
      {
        for (int n = 0; n < nwalk; n++)
        {
          Vsum = 0;
          if (wfn.transposed_vHS())
          {
            for (int i = 0; i < vHS.size(1); i++)
              Vsum += vHS[n][i];
          }
          else
          {
            for (int i = 0; i < vHS.size(0); i++)
              Vsum += vHS[i][n];
          }
          REQUIRE(real(ComplexType(Vsum)) == Approx(real(file_data.Vsum)));
          REQUIRE(imag(ComplexType(Vsum)) == Approx(imag(file_data.Vsum)));
        }
      }
      else
      {
        Vsum = 0;
        if (wfn.transposed_vHS())
        {
          for (int i = 0; i < vHS.size(1); i++)
            Vsum += vHS[0][i];
        }
        else
        {
          for (int i = 0; i < vHS.size(0); i++)
            Vsum += vHS[i][0];
        }
        app_log(1," Vsum: {}", ComplexType(Vsum));
      }

      TG.Global().barrier();
      // remove temporary file
      if (TG.Node().root())
        remove("dummy.h5");
    }
  }*/
}

template<bool MP, class Allocator>
void stochastic_wfn_matches_nomsd(boost::mpi3::communicator& world)
{
  if (not file_exists(UTEST_HAMIL) || not file_exists(UTEST_WFN))
    APP_ABORT(" Hamiltonian or wavefunction file not found. Run unit test with --hamil /path/to/hamil.h5 and --wfn /path/to/wfn.h5.");
  if (afqmc::getWavefunctionType(UTEST_WFN) != "NOMSD")
    return;

  WfnTestContext<MP, Allocator> ctx(world);
  Wavefunction& wfn_nomsd =
      ctx.register_wavefunction("wfn_nomsd", ctx.make_wfn_pt("wfn_nomsd", UTEST_WFN, false));
  Wavefunction& wfn_stoch =
      ctx.register_wavefunction("wfn_stoch", ctx.make_wfn_pt("wfn_stoch", UTEST_WFN, true));

  REQUIRE(wfn_nomsd.size_of_G_for_vbias() == wfn_stoch.size_of_G_for_vbias());
  REQUIRE(wfn_nomsd.transposed_G_for_vbias() == wfn_stoch.transposed_G_for_vbias());
  REQUIRE(wfn_nomsd.local_number_of_cholesky_vectors() == wfn_stoch.local_number_of_cholesky_vectors());

  using CMatrix = Matrix_<Allocator>;
  auto size_of_G = wfn_nomsd.size_of_G_for_vbias();
  int Gdim1      = (wfn_nomsd.transposed_G_for_vbias() ? ctx.nwalk : size_of_G);
  int Gdim2      = (wfn_nomsd.transposed_G_for_vbias() ? size_of_G : ctx.nwalk);
  double dt(0.01);
  auto nCV = wfn_nomsd.local_number_of_cholesky_vectors();

  WalkerSet wset_nomsd = ctx.make_walker_set();
  ctx.init_walkers(wset_nomsd, "wfn_nomsd");
  wfn_nomsd.Overlap(wset_nomsd);
  wfn_nomsd.Energy(wset_nomsd);
  ctx.TG.TG_local().barrier();

  std::vector<ComplexType> ov_ref;
  std::vector<ComplexType> E1_ref;
  std::vector<ComplexType> EXX_ref;
  std::vector<ComplexType> EJ_ref;
  std::vector<ComplexType> Etot_ref;
  for (auto it = wset_nomsd.begin(); it != wset_nomsd.end(); ++it)
  {
    ov_ref.push_back(ComplexType(*it->overlap()));
    E1_ref.push_back(ComplexType(*it->E1()));
    EXX_ref.push_back(ComplexType(*it->EXX()));
    EJ_ref.push_back(ComplexType(*it->EJ()));
    Etot_ref.push_back(ComplexType(it->energy()));
  }

  CMatrix G_nomsd({Gdim1, Gdim2}, ctx.alloc_);
  wfn_nomsd.MixedDensityMatrix_for_vbias(wset_nomsd, G_nomsd);
  ctx.maybe_init_model_ham(wfn_nomsd, dt);
  CMatrix X_nomsd({nCV, ctx.nwalk}, ctx.alloc_);
  wfn_nomsd.vbias(G_nomsd, X_nomsd, dt);
  ctx.TG.TG_local().barrier();

  WalkerSet wset_stoch = ctx.make_walker_set();
  ctx.init_walkers(wset_stoch, "wfn_nomsd");
  wfn_stoch.Overlap(wset_stoch);
  wfn_stoch.Energy(wset_stoch);
  ctx.TG.TG_local().barrier();

  REQUIRE(wset_stoch.size() == ov_ref.size());
  for (int n = 0; n < static_cast<int>(ov_ref.size()); ++n)
  {
    REQUIRE(real(ComplexType(*wset_stoch[n].overlap())) == Approx(real(ov_ref[n])));
    REQUIRE(imag(ComplexType(*wset_stoch[n].overlap())) == Approx(imag(ov_ref[n])));
    REQUIRE(real(ComplexType(*wset_stoch[n].E1())) == Approx(real(E1_ref[n])));
    REQUIRE(imag(ComplexType(*wset_stoch[n].E1())) == Approx(imag(E1_ref[n])));
    REQUIRE(real(ComplexType(*wset_stoch[n].EXX())) == Approx(real(EXX_ref[n])));
    REQUIRE(imag(ComplexType(*wset_stoch[n].EXX())) == Approx(imag(EXX_ref[n])));
    REQUIRE(real(ComplexType(*wset_stoch[n].EJ())) == Approx(real(EJ_ref[n])));
    REQUIRE(imag(ComplexType(*wset_stoch[n].EJ())) == Approx(imag(EJ_ref[n])));
    REQUIRE(real(ComplexType(wset_stoch[n].energy())) == Approx(real(Etot_ref[n])));
    REQUIRE(imag(ComplexType(wset_stoch[n].energy())) == Approx(imag(Etot_ref[n])));
  }

  CMatrix G_stoch({Gdim1, Gdim2}, ctx.alloc_);
  wfn_stoch.MixedDensityMatrix_for_vbias(wset_stoch, G_stoch);
  for (int i = 0; i < G_stoch.size(0); ++i)
    for (int j = 0; j < G_stoch.size(1); ++j)
    {
      REQUIRE(real(ComplexType(G_stoch[i][j])) == Approx(real(ComplexType(G_nomsd[i][j]))));
      REQUIRE(imag(ComplexType(G_stoch[i][j])) == Approx(imag(ComplexType(G_nomsd[i][j]))));
    }

  ctx.maybe_init_model_ham(wfn_stoch, dt);
  CMatrix X_stoch({nCV, ctx.nwalk}, ctx.alloc_);
  wfn_stoch.vbias(G_stoch, X_stoch, dt);
  ctx.TG.TG_local().barrier();

  for (int i = 0; i < X_stoch.size(0); ++i)
    for (int j = 0; j < X_stoch.size(1); ++j)
    {
      REQUIRE(real(ComplexType(X_stoch[i][j])) == Approx(real(ComplexType(X_nomsd[i][j]))));
      REQUIRE(imag(ComplexType(X_stoch[i][j])) == Approx(imag(ComplexType(X_nomsd[i][j]))));
    }

  ctx.TG.Global().barrier();
}

template<bool MP, class Allocator>
void stochastic_inner_walkers_init(boost::mpi3::communicator& world)
{
  if (not file_exists(UTEST_HAMIL) || not file_exists(UTEST_WFN))
    APP_ABORT(" Hamiltonian or wavefunction file not found. Run unit test with --hamil /path/to/hamil.h5 and --wfn /path/to/wfn.h5.");
  if (afqmc::getWavefunctionType(UTEST_WFN) != "NOMSD")
    return;

  WfnTestContext<MP, Allocator> ctx(world);
  auto initial_guess = [&](const std::string& wfn_id) { return ctx.WfnFac.getInitialGuess(wfn_id); };

  // Default inner_nwalkers = 1
  Wavefunction& wfn_stoch =
      ctx.register_wavefunction("wfn_stoch", ctx.make_wfn_pt("wfn_stoch", UTEST_WFN, true));
  REQUIRE(wfn_stoch.is_stochastic_wavefunction());
  REQUIRE(wfn_stoch.stochastic_inner_walkers_initialized());
  REQUIRE(wfn_stoch.stochastic_inner_wset().size() == 1);

  // inner_nwalkers > 1
  const int inner_nwalk = 5;
  Wavefunction& wfn_ensemble = ctx.register_wavefunction(
      "wfn_ensemble", ctx.make_wfn_pt("wfn_ensemble", UTEST_WFN, true, inner_nwalk));
  REQUIRE(wfn_ensemble.stochastic_inner_wset().size() == inner_nwalk);

  auto guess = initial_guess("wfn_stoch");
  REQUIRE(guess.size(0) == 2);
  REQUIRE(guess.size(1) == ctx.npol * ctx.NMO);
  REQUIRE(guess.size(2) == ctx.NAEA);

  for (int iw = 0; iw < wfn_stoch.stochastic_inner_wset().size(); ++iw)
  {
    REQUIRE(*wfn_stoch.stochastic_inner_wset()[iw].SlaterMatrix(Alpha) == guess[0]);
    if (ctx.type == COLLINEAR)
      REQUIRE(*wfn_stoch.stochastic_inner_wset()[iw].SlaterMatrix(Beta) ==
              guess[1](guess.extension(1), {0, ctx.NAEB}));
  }

  // Inner NOMSD evaluates on inner walkers
  boost::apply_visitor(
      [&](auto&& a) {
        using Wfn = std::decay_t<decltype(a)>;
        if constexpr (wavefunction_detail::is_stochastic_wfn<Wfn>::value)
        {
          a.inner_wfn().Overlap(a.inner_wset());
          a.inner_wfn().Energy(a.inner_wset());
          ctx.TG.TG_local().barrier();

          Wavefunction& wfn_nomsd =
              ctx.register_wavefunction("wfn_nomsd_ref", ctx.make_wfn_pt("wfn_nomsd_ref", UTEST_WFN, false));
          WalkerSet wset_ref = ctx.make_walker_set();
          ctx.init_walkers(wset_ref, "wfn_nomsd_ref");
          wfn_nomsd.Overlap(wset_ref);
          wfn_nomsd.Energy(wset_ref);
          ctx.TG.TG_local().barrier();

          REQUIRE(real(ComplexType(*a.inner_wset()[0].overlap())) ==
                  Approx(real(ComplexType(*wset_ref[0].overlap()))));
          REQUIRE(imag(ComplexType(*a.inner_wset()[0].overlap())) ==
                  Approx(imag(ComplexType(*wset_ref[0].overlap()))));
          REQUIRE(real(ComplexType(a.inner_wset()[0].energy())) == Approx(real(ComplexType(wset_ref[0].energy()))));
          REQUIRE(imag(ComplexType(a.inner_wset()[0].energy())) == Approx(imag(ComplexType(wset_ref[0].energy()))));

          // Replicated initial guess gives identical inner-walker observables
          a.inner_wfn().Overlap(a.inner_wset());
          ComplexType ov0 = ComplexType(*a.inner_wset()[0].overlap());
          for (int iw = 1; iw < inner_nwalk; ++iw)
          {
            REQUIRE(real(ComplexType(*a.inner_wset()[iw].overlap())) == Approx(real(ov0)));
            REQUIRE(imag(ComplexType(*a.inner_wset()[iw].overlap())) == Approx(imag(ov0)));
          }
        }
      },
      wfn_ensemble);

  ctx.TG.Global().barrier();
}

template<bool MP, class Allocator>
void wfn_fac_distributed(boost::mpi3::communicator& world, int ngroups)
{

  if (not file_exists(UTEST_HAMIL) || not file_exists(UTEST_WFN))
  {
    APP_ABORT(" Hamiltonian or wavefunction file not found. Run unit test with --hamil /path/to/hamil.h5 and --wfn /path/to/wfn.h5.");
  }
  else
  {
    // Global Task Group
    GlobalTaskGroup gTG(world);

    // First strip path of filename.
    std::string base_name = UTEST_WFN.substr(UTEST_WFN.find_last_of("\\/") + 1);
    // Remove file extension.
    std::string test_wfn = base_name.substr(0, base_name.find_last_of("."));
    auto file_data       = read_test_results_from_hdf<ComplexType>(UTEST_HAMIL, test_wfn);
    int NMO              = file_data.NMO;
    int NAEA             = file_data.NAEA;
    int NAEB             = file_data.NAEB;
    std::string wfn_type = afqmc::getWavefunctionType(UTEST_WFN);
    WALKER_TYPES type    = afqmc::getWalkerType(UTEST_WFN, wfn_type);
    int npol             = (type == NONCOLLINEAR) ? 2 : 1;
    int nspins           = (type == COLLINEAR) ? 2 : 1;

    std::map<std::string, AFQMCInfo> InfoMap;
    InfoMap.insert(std::pair<std::string, AFQMCInfo>("info0", AFQMCInfo{"info0", NMO, NAEA, NAEB}));

    ptree ham_pt;
    ham_pt.put("name","ham0");
    ham_pt.put("system","info0");
    ham_pt.put("filename",UTEST_HAMIL);

    HamiltonianFactory HamFac(InfoMap);
    HamFac.push("ham0", ham_pt);
    Hamiltonian& ham = HamFac.getHamiltonian(gTG, "ham0");

    auto TG    = TaskGroup_(gTG, std::string("WfnTG"), 1, gTG.getTotalCores());
    auto TGwfn = TaskGroup_(gTG, std::string("WfnTG"), ngroups, gTG.getTotalCores());
    int nwalk  = 11; // choose prime number to force non-trivial splits in shared routines
    utils::RandomGenerator_t rng;

    Allocator alloc_(make_localTG_allocator<ComplexType>(TG));

    ptree wlk_pt;
    wlk_pt.put("name","wset0");
    if(type == CLOSED) wlk_pt.put("walker_type","closed");
    else if(type == COLLINEAR) wlk_pt.put("walker_type","collinear");
    else if(type == NONCOLLINEAR) wlk_pt.put("walker_type","noncollinear");
    else if (type == FULLYPOLARIZED) wlk_pt.put("walker_type","fullypolarized");
    WalkerSet wset(TG, wlk_pt, InfoMap["info0"], &rng);

    ptree wfn_pt;
    wfn_pt.put("name","wfn0");
    wfn_pt.put("system","info0");
    wfn_pt.put("filename",UTEST_WFN);

    WavefunctionFactory WfnFac(InfoMap, MP);
    WfnFac.push("wfn0", wfn_pt);
    Wavefunction& wfn = WfnFac.getWavefunction(TGwfn, TGwfn, "wfn0", type, &ham, 1e-6, nwalk);

    auto initial_guess = WfnFac.getInitialGuess("wfn0");
    REQUIRE(initial_guess.size(0) == 2);
    REQUIRE(initial_guess.size(1) == npol * NMO);
    REQUIRE(initial_guess.size(2) == NAEA);

    if (type == COLLINEAR)
      wset.resize(nwalk, initial_guess[0], initial_guess[1](initial_guess.extension(1), {0, NAEB}));
    else
      wset.resize(nwalk, initial_guess[0], initial_guess[0]);

    wfn.Overlap(wset);

    using CMatrix = ComplexMatrix<Allocator>;
    Watch Time;
    Time.reset();
    wfn.Energy(wset);
    TG.TG().barrier();

    if (std::abs(file_data.E0 + file_data.E1 + file_data.E2) > 1e-8)
    {
      for (auto it = wset.begin(); it != wset.end(); ++it)
      {
        REQUIRE(real(ComplexType(*it->E1())) == Approx(real(file_data.E0 + file_data.E1)));
        REQUIRE(real(*it->EXX() + *it->EJ()) == Approx(real(file_data.E2)));
        REQUIRE(imag(it->energy()) == Approx(imag(file_data.E0 + file_data.E1 + file_data.E2)));
      }
    }
    else
    {
      app_log(1," E: {}", ComplexType(wset[0].energy())); 
      app_log(1," E0+E1: {}", ComplexType(*wset[0].E1()));
      app_log(1," EJ: {}", ComplexType(*wset[0].EJ())); 
      app_log(1," EXX: {}", ComplexType(*wset[0].EXX())); 
    }

    auto size_of_G = wfn.size_of_G_for_vbias();
    int Gdim1      = (wfn.transposed_G_for_vbias() ? nwalk : size_of_G);
    int Gdim2      = (wfn.transposed_G_for_vbias() ? size_of_G : nwalk);
    CMatrix G({Gdim1, Gdim2}, alloc_);
    wfn.MixedDensityMatrix_for_vbias(wset, G);

    double dt(0.01);
    auto nCV      = wfn.local_number_of_cholesky_vectors();
    CMatrix X({nCV, nwalk}, alloc_);
    
    // Initialize discrete propagators if using model Hamiltonian
    // Only root process should do this to avoid race conditions
    if (wfn.getHamType() == ModelHamiltonian) {
      if (TG.Global().root()) {
        // Use single precision types when MP=true to match internal storage
        using SPComplexType = typename to_working_precision<MP, ComplexType>::type;
        boost::multi::array<SPComplexType, 1> vMF_discrete(iextensions<1u>{nCV});
        boost::multi::array<SPComplexType, 1> nMF(iextensions<1u>{2*NMO}); 
        CMatrix Gmf({nspins * npol * NMO, npol * NMO}, ComplexType(0.0, 0.0), alloc_);
        // setup sparse vector to generate <nI>
        wfn.G_MF(Gmf);
        for(int i = 0; i < npol*NMO; i++)
          nMF[i] = SPComplexType(Gmf[i][i]);	
        if(type == COLLINEAR)
          for(int i = 0; i < NMO; i++)
            nMF[i+NMO] = SPComplexType(Gmf[i+NMO][i]);
        wfn.update_potentials(dt, nMF, vMF_discrete, false);
      }
      TG.Global().barrier();
    }
    
    Time.reset();
    wfn.vbias(G, X, dt);
    TG.TG().barrier();

    ComplexType Xsum = 0;
    if (std::abs(file_data.Xsum) > 1e-8)
    {
      for (int n = 0; n < nwalk; n++)
      {
        Xsum = 0;
        if (TGwfn.TG_local().root())
          for (int i = 0; i < X.size(0); i++)
            Xsum += X[i][n];
        Xsum = (TGwfn.TG() += Xsum);
        REQUIRE(real(ComplexType(Xsum)) == Approx(real(file_data.Xsum)));
        REQUIRE(imag(ComplexType(Xsum)) == Approx(imag(file_data.Xsum)));
      }
    }
    else
    {
      Xsum = 0;
      if (TGwfn.TG_local().root())
        for (int i = 0; i < X.size(0); i++)
          Xsum += X[i][0];
      Xsum = (TGwfn.TG() += Xsum);
      app_log(1," Xsum: {}", ComplexType(Xsum)); 
    }

    // vbias must be reduced if false
    if (not wfn.distribution_over_cholesky_vectors())
    {
      boost::multi::array<ComplexType, 2> T({nCV, nwalk});
      if (TGwfn.TG_local().root())
        std::copy_n(X.origin(), X.num_elements(), T.origin());
      else
        std::fill_n(T.origin(), T.num_elements(), ComplexType(0.0, 0.0));
      TGwfn.TG().all_reduce_in_place_n(raw_pointer_cast(T.origin()), T.num_elements(), std::plus<>());
      if (TGwfn.TG_local().root())
        std::copy_n(T.origin(), T.num_elements(), X.origin());
      TGwfn.TG_local().barrier();
    }

    // spin dependent HS potential?
    // generalize later
    int nx = ( wfn.getHamType() == ModelHamiltonian ? nspins*npol*npol : 1 ); 
    int nspin_hst = (wfn.spin_dependent_vHS()?2:1);
    int vdim1 = (wfn.transposed_vHS() ? nspin_hst*nwalk : NMO * NMO * nx );
    int vdim2 = (wfn.transposed_vHS() ? NMO * NMO * nx : nspin_hst*nwalk );
    if (wfn.getHamType() == ModelHamiltonian) // only sparseP2 is used
    {
      Time.reset();
      auto [vHS_up, vHS_down] = wfn.vHS_sparse(X, dt);
      TG.TG_local().barrier();
      
      // Convert sparse matrices to dense CMatrix objects for easier manipulation
      CMatrix vHS_up_dense({vHS_up->size(0), vHS_up->size(1)}, alloc_);
      CMatrix vHS_down_dense({vHS_down->size(0), vHS_down->size(1)}, alloc_);
      
#if defined(ENABLE_DEVICE)
      // For GPU builds, use host-side temporary arrays to avoid direct assignment to device memory
      boost::multi::array<ComplexType, 2> vHS_up_host({vHS_up->size(0), vHS_up->size(1)});
      boost::multi::array<ComplexType, 2> vHS_down_host({vHS_down->size(0), vHS_down->size(1)});
      
      // Initialize host arrays to zero
      std::fill_n(vHS_up_host.origin(), vHS_up_host.num_elements(), ComplexType(0.0));
      std::fill_n(vHS_down_host.origin(), vHS_down_host.num_elements(), ComplexType(0.0));
      
      // Convert sparse to dense using correct sparse matrix API on host
      for (int row = 0; row < static_cast<int>(vHS_up->size(0)); ++row) {
        auto [nnz, vals, cols] = vHS_up->sparse_row(row);
        for (size_t i = 0; i < nnz; ++i) {
          vHS_up_host[row][cols[i]] = vals[i];
        }
      }
      
      // For collinear systems, handle vHS_down if it's different from vHS_up
      // For noncollinear systems, vHS_up and vHS_down point to the same matrix
      if (vHS_up != vHS_down) {
        // COLLINEAR case: fill vHS_down_host separately
        for (int row = 0; row < static_cast<int>(vHS_down->size(0)); ++row) {
          auto [nnz, vals, cols] = vHS_down->sparse_row(row);
          for (size_t i = 0; i < nnz; ++i) {
            vHS_down_host[row][cols[i]] = vals[i];
          }
        }
      } else {
        // NONCOLLINEAR case: copy the same data
        std::copy_n(vHS_up_host.origin(), vHS_up_host.num_elements(), vHS_down_host.origin());
      }
      
      // Copy from host to device
      std::copy_n(vHS_up_host.origin(), vHS_up_host.num_elements(), vHS_up_dense.origin());
      std::copy_n(vHS_down_host.origin(), vHS_down_host.num_elements(), vHS_down_dense.origin());
#else
      // For CPU builds, initialize and directly assign to dense matrices
      std::fill_n(vHS_up_dense.origin(), vHS_up_dense.num_elements(), ComplexType(0.0));
      std::fill_n(vHS_down_dense.origin(), vHS_down_dense.num_elements(), ComplexType(0.0));
      
      // Convert sparse to dense using correct sparse matrix API
      for (int row = 0; row < static_cast<int>(vHS_up->size(0)); ++row) {
        auto [nnz, vals, cols] = vHS_up->sparse_row(row);
        for (size_t i = 0; i < nnz; ++i) {
          vHS_up_dense[row][cols[i]] = vals[i];
        }
      }
      
      // For collinear systems, handle vHS_down if it's different from vHS_up
      // For noncollinear systems, vHS_up and vHS_down point to the same matrix
      if (vHS_up != vHS_down) {
        // COLLINEAR case: fill vHS_down_dense separately
        for (int row = 0; row < static_cast<int>(vHS_down->size(0)); ++row) {
          auto [nnz, vals, cols] = vHS_down->sparse_row(row);
          for (size_t i = 0; i < nnz; ++i) {
            vHS_down_dense[row][cols[i]] = vals[i];
          }
        }
      } else {
        // NONCOLLINEAR case: copy the same data
        std::copy_n(vHS_up_dense.origin(), vHS_up_dense.num_elements(), vHS_down_dense.origin());
      }
#endif
      
      ComplexType Vsum = 0;
      if (std::abs(file_data.Vsum) > 1e-8)
      {
        for (int n = 0; n < nwalk; n++)
        {
          Vsum = 0;
          if (TGwfn.TG_local().root())
          {
            if (wfn.transposed_vHS())
            {
              for (int i = 0; i < vHS_up_dense.size(1); i++)
                Vsum += vHS_up_dense[n][i];
              for (int i = 0; i < vHS_down_dense.size(1); i++)
                Vsum += vHS_down_dense[n][i];
            }
            else
            {
              for (int i = 0; i < vHS_up_dense.size(0); i++)
                Vsum += vHS_up_dense[i][n];
              for (int i = 0; i < vHS_down_dense.size(0); i++)
                Vsum += vHS_down_dense[i][n];
            }
          }
          Vsum = (TGwfn.TG() += Vsum);
          REQUIRE(real(ComplexType(Vsum)) == Approx(real(file_data.Vsum)));
          REQUIRE(imag(ComplexType(Vsum)) == Approx(imag(file_data.Vsum)));
        }
      } else {
        Vsum = 0;
        if (TGwfn.TG_local().root())
        {
          if (wfn.transposed_vHS())
          {
            for (int i = 0; i < vHS_up_dense.size(1); i++)
              Vsum += vHS_up_dense[0][i];
            for (int i = 0; i < vHS_down_dense.size(1); i++)
              Vsum += vHS_down_dense[0][i];
          }
          else
          {
            for (int i = 0; i < vHS_up_dense.size(0); i++)
              Vsum += vHS_up_dense[i][0];
            for (int i = 0; i < vHS_down_dense.size(0); i++)
              Vsum += vHS_down_dense[i][0];
          }
        }
        Vsum = (TGwfn.TG() += Vsum);
        app_log(1," Vsum: {}", ComplexType(Vsum));
      }
    } else { // not a model Hamiltonian
      CMatrix vHS({vdim1, vdim2}, alloc_);
      Time.reset();
      wfn.vHS(X, vHS, dt);
      TG.TG_local().barrier();
      ComplexType Vsum = 0;
      if (std::abs(file_data.Vsum) > 1e-8)
      {
        for (int n = 0; n < nwalk; n++)
        {
          Vsum = 0;
          if (TGwfn.TG_local().root())
          {
            if (wfn.transposed_vHS())
            {
              for (int i = 0; i < vHS.size(1); i++)
                Vsum += vHS[n][i];
            }
            else
            {
              for (int i = 0; i < vHS.size(0); i++)
                Vsum += vHS[i][n];
            }
          }
          Vsum = (TGwfn.TG() += Vsum);
          REQUIRE(real(ComplexType(Vsum)) == Approx(real(file_data.Vsum)));
          REQUIRE(imag(ComplexType(Vsum)) == Approx(imag(file_data.Vsum)));
        }
      }
      else
      {
        Vsum = 0;
        if (TGwfn.TG_local().root())
        {
          if (wfn.transposed_vHS())
          {
            for (int i = 0; i < vHS.size(1); i++)
              Vsum += vHS[0][i];
          }
          else
          {
            for (int i = 0; i < vHS.size(0); i++)
              Vsum += vHS[i][0];
          }
        }
        Vsum = (TGwfn.TG() += Vsum);
        app_log(1," Vsum: {}", ComplexType(Vsum));
      }
    }
    
    
    return; /*

    // Restarting Wavefunction from file
    ptree wfn_pt2;
    wfn_pt2.put("name","wfn1");
    wfn_pt2.put("system","info0");
    wfn_pt2.put("filename","./dummy.h5");

    WfnFac.push("wfn1", wfn_pt2);
    Wavefunction& wfn2 = WfnFac.getWavefunction(TG, TG, "wfn1", type, nullptr, 1e-6, nwalk);

    WalkerSet wset2(TG, wlk_pt, InfoMap["info0"], &rng);
    //auto initial_guess = WfnFac.getInitialGuess("wfn0");
    REQUIRE(initial_guess.size(0) == 2);
    REQUIRE(initial_guess.size(1) == npol * NMO);
    REQUIRE(initial_guess.size(2) == NAEA);

    if (type == COLLINEAR)
      wset2.resize(nwalk, initial_guess[0], initial_guess[1](initial_guess.extension(1), {0, NAEB}));
    else
      wset2.resize(nwalk, initial_guess[0], initial_guess[0]);

    wfn2.Overlap(wset2);
    //for(auto it = wset2.begin(); it!=wset2.end(); ++it) {
    //REQUIRE(real(*it->overlap()) == Approx(1.0));
    //REQUIRE(imag(*it->overlap()) == Approx(0.0));
    //}

    wfn2.Energy(wset2);
    if (std::abs(file_data.E0 + file_data.E1 + file_data.E2) > 1e-8)
    {
      for (auto it = wset2.begin(); it != wset2.end(); ++it)
      {
        REQUIRE(real(ComplexType(*it->E1())) == Approx(real(file_data.E0 + file_data.E1)));
        REQUIRE(real(*it->EXX() + *it->EJ()) == Approx(real(file_data.E2)));
        REQUIRE(imag(ComplexType(it->energy())) == Approx(imag(file_data.E0 + file_data.E1 + file_data.E2)));
      }
    }
    else
    {
      app_log(1," E: {}", ComplexType(wset[0].energy())); 
      app_log(1," E0+E1: {}", ComplexType(*wset[0].E1()));
      app_log(1," EJ: {}", ComplexType(*wset[0].EJ())); 
      app_log(1," EXX: {}", ComplexType(*wset[0].EXX())); 
    }

    REQUIRE(size_of_G == wfn2.size_of_G_for_vbias());
    wfn2.MixedDensityMatrix_for_vbias(wset2, G);

    nCV = wfn2.local_number_of_cholesky_vectors();
    wfn2.vbias(G, X, dt);
    Xsum = 0;
    if (std::abs(file_data.Xsum) > 1e-8)
    {
      for (int n = 0; n < nwalk; n++)
      {
        Xsum = 0;
        if (TGwfn.TG_local().root())
          for (int i = 0; i < X.size(0); i++)
            Xsum += X[i][n];
        Xsum = (TGwfn.TG() += Xsum);
        REQUIRE(real(ComplexType(Xsum)) == Approx(real(file_data.Xsum)));
        REQUIRE(imag(ComplexType(Xsum)) == Approx(imag(file_data.Xsum)));
      }
    }
    else
    {
      Xsum = 0;
      if (TGwfn.TG_local().root())
        for (int i = 0; i < X.size(0); i++)
          Xsum += X[i][0];
      Xsum = (TGwfn.TG() += Xsum);
      app_log(1," Xsum: {}", ComplexType(Xsum)); 
    }

    // vbias must be reduced if false
    if (not wfn.distribution_over_cholesky_vectors())
    {
      boost::multi::array<ComplexType, 2> T({nCV, nwalk});
      if (TGwfn.TG_local().root())
        std::copy_n(X.origin(), X.num_elements(), T.origin());
      else
        std::fill_n(T.origin(), T.num_elements(), ComplexType(0.0, 0.0));
      TGwfn.TG().all_reduce_in_place_n(raw_pointer_cast(T.origin()), T.num_elements(), std::plus<>());
      if (TGwfn.TG_local().root())
        std::copy_n(T.origin(), T.num_elements(), X.origin());
      TGwfn.TG_local().barrier();
    }

    if (wfn2.getHamType() == ModelHamiltonian) // only sparseP2 is used - denseP2 is hardcoded to never run!
    {
      auto [vHS_up2, vHS_down2] = wfn2.vHS_sparse(X, dt); // vHS_sparse lives inside the ModelHamOps class
      TG.TG_local().barrier();
      
      // Convert sparse matrices to dense CMatrix objects for easier manipulation
      CMatrix vHS_up_dense2({vHS_up2->size(0), vHS_up2->size(1)}, alloc_);
      CMatrix vHS_down_dense2({vHS_down2->size(0), vHS_down2->size(1)}, alloc_);
      
      // Initialize dense matrices to zero
      std::fill_n(vHS_up_dense2.origin(), vHS_up_dense2.num_elements(), ComplexType(0.0));
      std::fill_n(vHS_down_dense2.origin(), vHS_down_dense2.num_elements(), ComplexType(0.0));
      
      // Convert sparse to dense using correct sparse matrix API
      for (int row = 0; row < static_cast<int>(vHS_up2->size(0)); ++row) {
        auto [nnz, vals, cols] = vHS_up2->sparse_row(row);
        for (size_t i = 0; i < nnz; ++i) {
          vHS_up_dense2[row][cols[i]] = vals[i];
        }
      }
      
      // For collinear systems, handle vHS_down if it's different from vHS_up
      // For noncollinear systems, vHS_up and vHS_down point to the same matrix
      if (vHS_up2 != vHS_down2) {
        // COLLINEAR case: fill vHS_down_dense separately
        for (int row = 0; row < static_cast<int>(vHS_down2->size(0)); ++row) {
          auto [nnz, vals, cols] = vHS_down2->sparse_row(row);
          for (size_t i = 0; i < nnz; ++i) {
            vHS_down_dense2[row][cols[i]] = vals[i];
          }
        }
      } else {
        // NONCOLLINEAR case: copy the same data
        std::copy_n(vHS_up_dense2.origin(), vHS_up_dense2.num_elements(), vHS_down_dense2.origin());
      }
      
      ComplexType Vsum = 0;
      if (std::abs(file_data.Vsum) > 1e-8)
      {
        for (int n = 0; n < nwalk; n++)
        {
          Vsum = 0;
          if (TGwfn.TG_local().root())
          {
            if (wfn2.transposed_vHS())
            {
              for (int i = 0; i < vHS_up_dense2.size(1); i++)
                Vsum += vHS_up_dense2[n][i];
              for (int i = 0; i < vHS_down_dense2.size(1); i++)
                Vsum += vHS_down_dense2[n][i];
            }
            else
            {
              for (int i = 0; i < vHS_up_dense2.size(0); i++)
                Vsum += vHS_up_dense2[i][n];
              for (int i = 0; i < vHS_down_dense2.size(0); i++)
                Vsum += vHS_down_dense2[i][n];
            }
          }
          Vsum = (TGwfn.TG() += Vsum);
          REQUIRE(real(ComplexType(Vsum)) == Approx(real(file_data.Vsum)));
          REQUIRE(imag(ComplexType(Vsum)) == Approx(imag(file_data.Vsum)));
        }
      }
      else
      {
        Vsum = 0;
        if (TGwfn.TG_local().root())
        {
          if (wfn2.transposed_vHS())
          {
            for (int i = 0; i < vHS_up_dense2.size(1); i++)
              Vsum += vHS_up_dense2[0][i];
            for (int i = 0; i < vHS_down_dense2.size(1); i++)
              Vsum += vHS_down_dense2[0][i];
          }
          else
          {
            for (int i = 0; i < vHS_up_dense2.size(0); i++)
              Vsum += vHS_up_dense2[i][0];
            for (int i = 0; i < vHS_down_dense2.size(0); i++)
              Vsum += vHS_down_dense2[i][0];
          }
        }
        Vsum = (TGwfn.TG() += Vsum);
        app_log(1," Vsum: {}", ComplexType(Vsum));
      }
    } else { // not a model Hamiltonian
      CMatrix vHS({vdim1, vdim2}, alloc_);
      wfn2.vHS(X, vHS, dt);
      TG.TG_local().barrier();
      ComplexType Vsum = 0;
      if (std::abs(file_data.Vsum) > 1e-8)
      {
        for (int n = 0; n < nwalk; n++)
        {
          Vsum = 0;
          if (TGwfn.TG_local().root())
          {
            if (wfn.transposed_vHS())
            {
              for (int i = 0; i < vHS.size(1); i++)
                Vsum += vHS[n][i];
            }
            else
            {
              for (int i = 0; i < vHS.size(0); i++)
                Vsum += vHS[i][n];
            }
          }
          Vsum = (TGwfn.TG() += Vsum);
          REQUIRE(real(ComplexType(Vsum)) == Approx(real(file_data.Vsum)));
          REQUIRE(imag(ComplexType(Vsum)) == Approx(imag(file_data.Vsum)));
        }
      }
      else
      {
        Vsum = 0;
        if (TGwfn.TG_local().root())
        {
          if (wfn.transposed_vHS())
          {
            for (int i = 0; i < vHS.size(1); i++)
              Vsum += vHS[0][i];
          }
          else
          {
            for (int i = 0; i < vHS.size(0); i++)
              Vsum += vHS[i][0];
          }
        }
        Vsum = (TGwfn.TG() += Vsum);
        app_log(1," Vsum: {}", ComplexType(Vsum));
      }
    }

    TG.Global().barrier();
    // remove temporary file
    if (TG.Node().root())
      remove("dummy.h5");*/
  }
}

TEST_CASE("wfn_fac_sdet", "[wavefunction_factory]")
{
  auto world = boost::mpi3::environment::get_world_instance();
  auto node = world.split_shared(world.rank());
  setup_loggers(world.root(),2,2);

#if defined(ENABLE_DEVICE)

  arch::INIT(node);
  using Alloc = device::device_allocator<ComplexType>;
#else
  using Alloc = shared_allocator<ComplexType>;
#endif
  setup_memory_managers(node, 10uL * 1024uL * 1024uL);

  wfn_fac<false,Alloc>(world);
  wfn_fac<true,Alloc>(world);
  release_memory_managers();
}

TEST_CASE("stochastic_wfn_matches_nomsd", "[wavefunction_factory][stochastic_wfn]")
{
  auto world = boost::mpi3::environment::get_world_instance();
  auto node  = world.split_shared(world.rank());
  setup_loggers(world.root(), 2, 2);

#if defined(ENABLE_DEVICE)
  arch::INIT(node);
  using Alloc = device::device_allocator<ComplexType>;
#else
  using Alloc = shared_allocator<ComplexType>;
#endif
  setup_memory_managers(node, 10uL * 1024uL * 1024uL);

  stochastic_wfn_matches_nomsd<false, Alloc>(world);
  stochastic_wfn_matches_nomsd<true, Alloc>(world);
  release_memory_managers();
}

TEST_CASE("stochastic_inner_walkers_init", "[wavefunction_factory][stochastic_wfn]")
{
  auto world = boost::mpi3::environment::get_world_instance();
  auto node  = world.split_shared(world.rank());
  setup_loggers(world.root(), 2, 2);

#if defined(ENABLE_DEVICE)
  arch::INIT(node);
  using Alloc = device::device_allocator<ComplexType>;
#else
  using Alloc = shared_allocator<ComplexType>;
#endif
  setup_memory_managers(node, 10uL * 1024uL * 1024uL);

  stochastic_inner_walkers_init<false, Alloc>(world);
  stochastic_inner_walkers_init<true, Alloc>(world);
  release_memory_managers();
}

TEST_CASE("wfn_fac_distributed", "[wavefunction_factory]")
{
  auto world = boost::mpi3::environment::get_world_instance();
  setup_loggers(world.root(),2,0);

#if defined(ENABLE_DEVICE)
  auto node = world.split_shared(world.rank());
  int ngrp(world.size());

  arch::INIT(node);
  using Alloc = device::device_allocator<ComplexType>;
#else
  auto node   = world.split_shared(world.rank());
  int ngrp(world.size() / node.size());
  using Alloc = shared_allocator<ComplexType>;
#endif
  setup_memory_managers(node, 10uL * 1024uL * 1024uL);

  wfn_fac_distributed<false,Alloc>(world, ngrp);
  wfn_fac_distributed<true,Alloc>(world, ngrp);
  release_memory_managers();
}


} // namespace sfqmc
