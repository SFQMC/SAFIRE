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

#pragma once

#include <cmath>
#include <format>
#include <vector>
#include <string>
#include <tuple>

#include "IO/app_loggers.h"
#include "IO/banner.hpp"
#include "AFQMC/parameters.hpp"
#include "utilities/Random.hpp"
#include "utilities/check.hpp"

#include "AFQMC/config.h"

#include "numerics/shared_array/const_shared_array.hpp"
#include "AFQMC/Walkers/WalkerSet.hpp"
#include "AFQMC/Wavefunctions/Wavefunction.hpp"
#include "AFQMC/Propagators/WalkerSetUpdate.hpp"

namespace sfqmc
{
namespace afqmc
{
/*
 * Base AFQMC propagator.
 * For all hamiltonians that only use a dense vHS. For model hamiltonians, use AFQMCModelPropagator.
 */
template<MEMORY_SPACE MEM>
class AFQMCBasePropagator
{

public:
  AFQMCBasePropagator() = delete;

  AFQMCBasePropagator(const PropagatorParameters& params,
                      std::shared_ptr<utils::mpi_context_t<mpi3::communicator>> mpi_,
                      Wavefunction<MEM>& wfn_,
                      std::shared_ptr<utils::RandomGenerator_t<MEM>> r,
                      double dt)
      : mpi(mpi_),
        wfn(std::addressof(wfn_)),
        P1s(0),
        P1s_inv(0),
        rng(r),
        FieldTypes(wfn->getFieldTypes()),
#if defined(ENABLE_DEVICE)
        FieldTypes_dev(FieldTypes()),
#endif
        rng_block_size(wfn->number_of_cholesky_vectors()),
        timestep(dt)
  {
    app_log(1, section(std::format("Initializing Propagator \"{}\"", params.name)));

    utils::check(bool(mpi), "Error: Null mpi_context.");
    std::tie(nspins_in_vHS, npol_in_vHS) = wfn->vHS_dims();
    app_log(1,"vHS dimensions: nspins = {}, npol = {}", nspins_in_vHS, npol_in_vHS);
    auto hamtype(wfn->getHamType());
    vbias_bound        = resolved(params.vbias_bound, "vbias_bound");
    upper_cutoff_scale = resolved(params.upper_cutoff_scale, "upper_cutoff_scale");
    lower_cutoff_scale = resolved(params.lower_cutoff_scale, "lower_cutoff_scale");
    denseP2            = resolved(params.denseP2, "denseP2");

    order               = params.taylor_n;
    weight_bound_floor    = params.weight_bound_floor;
    weight_bound_fraction = params.weight_bound_fraction;
    subtractMF         = params.subtractMF;
    hybrid              = params.hybrid;
    printP1eV           = params.printP1eigval;
    if(not mpi->comm.root()) printP1eV = false;
    free_projection     = params.free_projection;
    denseP1             = params.denseP1;
    debug_verbosity     = params.debug_verbosity;
    natural_shift       = params.natural_shift;
    use_cp_constraint   = params.use_cp_constraint;
    project_force_bias      = params.project_force_bias;

    utils::check(!free_projection || hybrid, "BasePropagator: free_projection requires hybrid = true.");
    utils::check(weight_bound_floor > 0.0, "weight_bound_floor must be positive, got {}",
                 weight_bound_floor);
    utils::check(weight_bound_fraction > 0.0 && weight_bound_fraction <= 1.0,
                 "weight_bound_fraction must be in (0,1], got {}", weight_bound_fraction);

    utils::check(denseP2 or hamtype == HamiltonianType::model_hamiltonian, "denseP2=false only allowed with ModelHamiltonian.");

    if ((hamtype == HamiltonianType::kp_factorized || hamtype == HamiltonianType::kpthc) && denseP1)
    {
      app_error("dense Ham. with kpoints");
      utils::check(false,"BasePropagator: set denseP1 to false");
    }

    app_log(1,"energy offset (E0): {}", wfn->energy_offset());
    app_log(1,"cutoff scales (upper/lower): {} / {}", upper_cutoff_scale, lower_cutoff_scale);
    app_log(1,"weight bound: max({}, {} * target population) per walker",
            weight_bound_floor, weight_bound_fraction);
    if(denseP1)
      app_log(1,"Using dense 1-body propagator");
    else
      app_log(1,"Using sparse 1-body propagator");
    if(denseP2)
      app_log(1,"Using dense 2-body propagator (vHS)");
    else
      app_log(1,"Using sparse 2-body propagator (vHS)");

    if(nspins_in_vHS>1) 
      app_log(1, "Using a spin-dependent vHS.");
    if(npol_in_vHS>1) 
      app_log(1, "Using a polarization-dependent vHS.");

    if (hybrid)
      app_log(1,"Using hybrid method to calculate the weights during the propagation.");
    else
      app_log(1,"Using local energy method to calculate the weights during the propagation.");
    if(natural_shift)
      app_log(1, "Using natural shifts with discrete propagators. ");

    if (debug_verbosity) {
      app_warning("Using debug verbosity. THIS WILL GENERATE A LOT OF OUTPUT. Intended for debugging purposes with a few walkers only.");
    }

    // read orbital matrix if excited state propagator
    excitedState = false;
    // Excited-state propagator is disabled pending re-implementation (it was an
    // abort-only stub).
    /*
    if (excited_file != "" && i_ >= 0 && a_ >= 0)
    {
      if (i_ < NMO && a_ < NMO)
      {
        if (i_ >= nup || a_ < nup)
          utils::check(false," Errors: Inconsistent excited orbitals for alpha electrons. ");
        excitedState        = true;
        maxOccupExtendedMat = {a_, ndown};
        numExcitations      = {1, 0};
        excitations.push_back({i_, a_});
      }
      else if (i_ >= NMO && a_ >= NMO)
      {
        if (i_ >= NMO + ndown || a_ < NMO + ndown)
          utils::check(false," Errors: Inconsistent excited orbitals for beta electrons. ");
        excitedState        = true;
        maxOccupExtendedMat = {nup, a_ - NMO};
        numExcitations      = {0, 1};
        excitations.push_back({i_ - NMO, a_ - NMO});
      }
      else
      {
        utils::check(false," Errors: Inconsistent excited orbitals. ");
      }
      utils::check(false," Error: Finish implementation. ");
      // read from hdf5
      //readWfn(excited_file, excitedOrbMat_, NMO, maxOccupExtendedMat.first, maxOccupExtendedMat.second);
    }
    */

    generateP1();
  }

  void Propagate(WalkerSet<MEM>& wset, RealType Eshift, int nt = 0);

  void BackPropagate(int nbpsteps, int nStabalize, WalkerSet<MEM>& wset,
                     memory::array_view<MEM,ComplexType,4> Refs,
                     memory::array_view<MEM,ComplexType,2> logdetR);

  void PropagateOperators(int steps, WalkerSet<MEM>& wset,
                          memory::array_view<MEM,ComplexType,4> X,
                          memory::array_view<MEM,ComplexType,4> Y,
                          memory::array_view<MEM,ComplexType,4> M);

  /// Whether Propagate leaves the components of the local energy on the walkers. Only the
  /// local energy update evaluates them; hybrid propagation does not.
  bool stores_local_energy() const { return !hybrid; }

  int number_of_cholesky_vectors() const { return wfn->number_of_cholesky_vectors(); }

  void Orthogonalize(WalkerSet<MEM>& wset);

  void set_rng_block_size(int sz) { rng_block_size = sz; }

  // Report, at the end of the calculation, how often the propagation bounding boxes were
  // triggered: the force-bias (vbias) clamp and the local-energy (eloc) clamp. Counters are
  // aggregated across all ranks; only the root prints. `Eshift` is the value the caller ends
  // the run with, which fixes the eloc clamp window reported here.
  void printBoundStatistics(RealType Eshift) {
    long buf[9] = {vbias_bound_stats.total,  vbias_bound_stats.upper,  vbias_bound_stats.lower,
                   eloc_bound_stats.total,   eloc_bound_stats.upper,   eloc_bound_stats.lower,
                   weight_bound_stats.total, weight_bound_stats.upper, weight_bound_stats.lower};
    mpi->comm.all_reduce_in_place_n(&buf[0], 9, std::plus<>());
    if (not mpi->comm.root()) return;
    long vb_tot = buf[0], vb_up = buf[1];
    long el_tot = buf[3], el_up = buf[4], el_lo = buf[5];
    long wb_tot = buf[6], wb_up = buf[7];
    auto pct = [](long h, long t) { return t > 0 ? 100.0 * double(h) / double(t) : 0.0; };

    app_log(1, "\n{}", banner("Bounding-box trigger statistics"));

    app_log(1, " Force-bias clamp  [|vbias| > vbias_bound], per (walker,field):");
    if (vb_tot == 0)
      app_log(1, "   not measured (host-side counting only).");
    else
      app_log(1, "   operations: {}   hits: {} ({:.4f}%)  [upper/magnitude: {} ({:.4f}%)]",
              vb_tot, vb_up, pct(vb_up, vb_tot), vb_up, pct(vb_up, vb_tot));

    app_log(1, " Local-energy clamp  [outside Eshift ± (upper|lower)_cutoff_scale*sqrt(2/dt)], per walker:");
    app_log(1, "   final Eshift: {}, clamp window: [{}, {}]", Eshift,
            Eshift - lower_cutoff_scale * std::sqrt(2.0 / timestep),
            Eshift + upper_cutoff_scale * std::sqrt(2.0 / timestep));
    if (el_tot == 0)
      app_log(1, "   not triggered (0 operations counted).");
    else
      app_log(1, "   operations: {}   hits: {} ({:.4f}%)  [upper: {} ({:.4f}%), lower: {} ({:.4f}%)]",
              el_tot, el_up + el_lo, pct(el_up + el_lo, el_tot),
              el_up, pct(el_up, el_tot), el_lo, pct(el_lo, el_tot));

    app_log(1, " Walker-weight clamp  [|w| > max({}, {}*target population)], per walker:",
            weight_bound_floor, weight_bound_fraction);
    if (wb_tot == 0)
      app_log(1, "   not triggered (0 operations counted).");
    else
      app_log(1, "   operations: {}   hits: {} ({:.4f}%)", wb_tot, wb_up, pct(wb_up, wb_tot));
    app_log(1, "{}\n", hrule());
  }


private:
  // constructs the 1-body hamiltonian for the timestep of this propagator and generates the
  // propagator. If Pinv = true, the routine also builds the inverse of the propagator and
  // stores it in P1d_inv/P1s_inv.
  void generateP1(bool Pinv = false);

  // mpi_context
  std::shared_ptr<utils::mpi_context_t<mpi3::communicator>> mpi;

  Wavefunction<MEM>* wfn = nullptr;

  // 1Body propagator in sparse and dense forms
  bool denseP1 = false;
  // vHS in sparse and dense forms
  // Only ModelHamiltonian has sparse vHS
  bool denseP2 = true;
  // P1s[ispin](npol*NMO,npol*NMO)
  nda::array<PsiT_Matrix<MEM>, 1> P1s;
  // P1d[ispin,npol*NMO,npol*NMO]
  memory::const_shared_array<MEM, ComplexType, 3> P1d;

  // used to propagate operator orbitals
  nda::array<PsiT_Matrix<MEM>, 1> P1s_inv;
  memory::const_shared_array<MEM, ComplexType, 3> P1d_inv;

  memory::const_shared_array<MEM, ComplexType, 1> vMF;

  std::shared_ptr<utils::RandomGenerator_t<MEM>> rng;

// erase
//utils::RandomGenerator_t<> rng_h = utils::RandomGenerator_t<>(777);

  nda::array<int,1> FieldTypes;
#if defined(ENABLE_DEVICE)
  memory::array<MEM,int,1> FieldTypes_dev;
#endif

  // number of random numbers to generate for each walker at each step.
  // In general, rng_block_size will be set to the number of cholesky vectors.
  // In correlated sampling calculations, this will be the maximum number of cholesky
  // vectors in all systems, which is needed to keep the generators synchronized. 
  int rng_block_size = 0;

  RealType timestep;
  int order = 6;
  bool printP1eV = false;

  RealType vbias_bound;
  bool subtractMF = true;

  // type of propagation
  bool free_projection = false;
  bool hybrid = true;
  double upper_cutoff_scale = 10.0;
  double lower_cutoff_scale = 1.0;
  double weight_bound_floor = 100.0;
  double weight_bound_fraction = 0.1;
  bool natural_shift = true;
  bool use_cp_constraint = false;
  bool project_force_bias = false;

  int nspins_in_vHS = 1;
  int npol_in_vHS   = 1;

  bool debug_verbosity = false;

  // Diagnostic counters: how often the propagation bounding boxes are triggered over the run.
  BoundStats vbias_bound_stats;  // force-bias (vbias) clamp, counted per (walker,field)
  BoundStats eloc_bound_stats;   // local-energy (eloc) clamp, counted per walker
  BoundStats weight_bound_stats; // walker-weight clamp, counted per walker

  // excited state propagator
  bool excitedState = false;
  std::vector<std::pair<int, int>> excitations;
  memory::const_shared_array<MEM, ComplexType, 3> excitedOrbMat;
  std::pair<int, int> maxOccupExtendedMat;
  std::pair<int, int> numExcitations;

  void assemble_X(memory::array_view<MEM,ComplexType,2> X,
                  memory::array_view<MEM,ComplexType,1> HWs,
                  bool addRAND = true);

  // Definitions live in AFQMCBasePropagator.cpp, next to their only callers.
  template<char TA, typename VHS_t>
  void apply_propagators(WalkerSet<MEM>& wset, VHS_t const& v, bool P1inv = false);

  template<char TA, typename VHS_t, nda::MemoryArrayOfRank<3> Mat3>
  void apply_propagators(WALKER_TYPES wtype, int npol, Mat3&& Xa, Mat3&& Xb,
                         VHS_t const& v, bool P1inv = false);

};

} // namespace afqmc

} // namespace sfqmc

