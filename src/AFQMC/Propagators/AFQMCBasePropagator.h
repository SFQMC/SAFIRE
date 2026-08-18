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
#include "AFQMC/Wavefunctions/Wavefunction.hpp"
#include "AFQMC/SlaterDeterminantOperations/propagate.hpp"
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
                      std::shared_ptr<utils::RandomGenerator_t<MEM>> r)
      : mpi(mpi_),
        wfn(std::addressof(wfn_)),
        P1s(0),
        P1s_inv(0),
        rng(r),
        FieldTypes(wfn->getFieldTypes()),
#if defined(ENABLE_DEVICE)
        FieldTypes_dev(FieldTypes()),
#endif
        rng_block_size(wfn->number_of_cholesky_vectors())
  {
    utils::check(bool(mpi), "Error: Null mpi_context.");
    const int NMO = wfn->getNMO();
    std::tie(nspins_in_vHS, npol_in_vHS) = wfn->vHS_dims();
    app_log(1,"vHS dimensions: nspins = {}, npol = {}", nspins_in_vHS, npol_in_vHS);
    auto hamtype(wfn->getHamType());
    vbias_bound        = resolved(params.vbias_bound, "vbias_bound");
    upper_cutoff_scale = resolved(params.upper_cutoff_scale, "upper_cutoff_scale");
    lower_cutoff_scale = resolved(params.lower_cutoff_scale, "lower_cutoff_scale");
    denseP2            = resolved(params.denseP2, "denseP2");
    symmetric_split    = resolved(params.symmetric_split, "symmetric_split");

    const std::string& external_field = params.external_field;
    const double external_field_scale = params.external_field_scale;
    order               = params.taylor_n;
    apply_constrain     = params.apply_constrain;
    importance_sampling = params.importance_sampling;
    substractMF         = params.substractMF;
    hybrid              = params.hybrid;
    printP1eV           = params.printP1eigval;
    if(not mpi->comm.root()) printP1eV = false;
    free_projection     = params.free_projection;
    denseP1             = params.denseP1;
    debug_verbosity     = params.debug_verbosity;
    natural_shift       = params.natural_shift;
    use_cp_constraint   = params.use_cp_constraint;
    use_real_vbias      = params.use_real_vbias;

    if (free_projection)
    {
      if (importance_sampling || !hybrid || apply_constrain)
      {
        app_error("Free projection requires:");
        app_error(" importance_sampling = no, currently {}", importance_sampling);
        app_error(" hybrid = yes, currently {}", hybrid);
        app_error(" apply_constrain = no, currently {}", apply_constrain);
        utils::check(false,"BasePropagator: free_projection");
      }
    }
    utils::check(denseP2 or hamtype == ModelHamiltonian, "denseP2=false only allowed with ModelHamiltonian.");

    if ((hamtype == KPFactorized || hamtype == KPTHC) && denseP1)
    {
      app_error("dense Ham. with kpoints");
      utils::check(false,"BasePropagator: set denseP1 to false");
    }

    app_log(1,"cutoff scales (upper/lower): {} / {}", upper_cutoff_scale, lower_cutoff_scale);
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

    utils::check(importance_sampling || free_projection, "importance_sampling=false without free projection does not make sense.");

    if (hybrid)
      app_log(1,"Using hybrid method to calculate the weights during the propagation.");
    else
      app_log(1,"Using local energy method to calculate the weights during the propagation.");
    if(natural_shift)
      app_log(1, "Using natural shifts with discrete propagators. ");
    if(symmetric_split)
      app_log(1, "Using symmetric split of walker weight update.");

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
    if (external_field != std::string(""))
    {
      //    read_external_field(H1ext);
      auto walker_type = wfn->getWalkerType();
      int npol  = walker_type == NONCOLLINEAR ? 2 : 1;
      int nspin = walker_type == COLLINEAR ? 2 : 1;
      external_H1 = true;
      H1ext = memory::share_from_root(*mpi, [&] {
        // use hdf5 format!!!
        nda::array<ComplexType,3> h1ext(nspin, npol*NMO, npol*NMO);
        std::ifstream in(external_field.c_str());
        for (int is = 0; is < nspin; is++)
          for (int i = 0; i < npol*NMO; i++)
            for (int j = 0; j < npol*NMO; j++) {
              in >> h1ext(is,i,j);
              utils::check(not in.fail()," Error: Problems with external field.");
            }
        h1ext *= external_field_scale;
        return h1ext;
      });
    }

  }

  AFQMCBasePropagator(AFQMCBasePropagator const& other) = delete;
  AFQMCBasePropagator& operator=(AFQMCBasePropagator const& other) = delete;
  AFQMCBasePropagator(AFQMCBasePropagator&& other)                 = default;
  AFQMCBasePropagator& operator=(AFQMCBasePropagator&& other) = default;

  template<class WlkSet>
  void Propagate(WlkSet& wset, RealType E1, RealType dt, int nt = 0);

  template<class WlkSet>
  void BackPropagate(int nbpsteps, int nStabalize, WlkSet& wset, 
        nda::MemoryArrayOfRank<4> auto&& Refs, nda::MemoryArrayOfRank<2> auto&& logdetR);           
  template<class WlkSet> 
  void PropagateOperators(int steps, WlkSet& wset,  
        nda::MemoryArrayOfRank<4> auto&& X, nda::MemoryArrayOfRank<4> auto&& Y,
        nda::MemoryArrayOfRank<4> auto&& M);       

  bool hybrid_propagation() { return hybrid; }

  bool free_propagation() { return free_projection; }

  int number_of_cholesky_vectors() const { return wfn->number_of_cholesky_vectors(); }

  // constructs the 1-body hamiltonian for propagation and generates the propagator
  // if Pinv = true, the routine builds the inverse of the propagator and stores it in P_inv
  void generateP1(double dt, WALKER_TYPES walker_type, bool Pinv = false);

  template<class WlkSet>
  void Orthogonalize(WlkSet& wset);

  void set_rng_block_size(int sz) { rng_block_size = sz; }

  // Report, at the end of the calculation, how often the propagation bounding boxes were
  // triggered: the force-bias (vbias) clamp and the local-energy (eloc) clamp. Counters are
  // aggregated across all ranks; only the root prints.
  void printBoundStatistics()
  {
    long buf[6] = {vbias_bound_stats.total, vbias_bound_stats.upper, vbias_bound_stats.lower,
                   eloc_bound_stats.total,  eloc_bound_stats.upper,  eloc_bound_stats.lower};
    mpi->comm.all_reduce_in_place_n(&buf[0], 6, std::plus<>());
    if (not mpi->comm.root()) return;
    long vb_tot = buf[0], vb_up = buf[1];
    long el_tot = buf[3], el_up = buf[4], el_lo = buf[5];
    auto pct = [](long h, long t) { return t > 0 ? 100.0 * double(h) / double(t) : 0.0; };

    app_log(1, "\n{}", banner("Bounding-box trigger statistics"));

    app_log(1, " Force-bias (vbias) clamp  [|vbias| > vbias_bound*sqrt(dt)], per (walker,field):");
    if (vb_tot == 0)
      app_log(1, "   not measured (host-side counting only).");
    else
      app_log(1, "   operations: {}   hits: {} ({:.4f}%)  [upper/magnitude: {} ({:.4f}%)]",
              vb_tot, vb_up, pct(vb_up, vb_tot), vb_up, pct(vb_up, vb_tot));

    app_log(1, " Local-energy (eloc) clamp  [eloc outside Eshift +/- cutoff_scale*sqrt(2/dt)], per walker:");
    if (el_tot == 0)
      app_log(1, "   not triggered (0 operations counted).");
    else
      app_log(1, "   operations: {}   hits: {} ({:.4f}%)  [upper: {} ({:.4f}%), lower: {} ({:.4f}%)]",
              el_tot, el_up + el_lo, pct(el_up + el_lo, el_tot),
              el_up, pct(el_up, el_tot), el_lo, pct(el_lo, el_tot));
    app_log(1, "{}\n", hrule());
  }


protected:
  // mpi_context
  std::shared_ptr<utils::mpi_context_t<mpi3::communicator>> mpi;

  Wavefunction<MEM>* wfn = nullptr;

  memory::const_shared_array<HOST_MEMORY,ComplexType,3> H1ext;

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

  RealType old_dt = -123456.789;
  int order = 6;
  bool external_H1 = false; 
  bool printP1eV = false;

  RealType vbias_bound;
  bool substractMF = true;

  // type of propagation
  bool free_projection = false;
  bool hybrid = true;
  bool importance_sampling = true;
  bool apply_constrain = true;
  double upper_cutoff_scale = 10.0;
  double lower_cutoff_scale = 1.0;
  bool natural_shift = true;
  bool symmetric_split = true;
  bool use_cp_constraint = false;
  bool use_real_vbias = false;

  int nspins_in_vHS = 1;
  int npol_in_vHS   = 1;

  bool debug_verbosity = false;

  // Diagnostic counters: how often the propagation bounding boxes are triggered over the run.
  BoundStats vbias_bound_stats;  // force-bias (vbias) clamp, counted per (walker,field)
  BoundStats eloc_bound_stats;   // local-energy (eloc) clamp, counted per walker

  // excited state propagator
  bool excitedState = false;
  std::vector<std::pair<int, int>> excitations;
  memory::const_shared_array<MEM, ComplexType, 3> excitedOrbMat;
  std::pair<int, int> maxOccupExtendedMat;
  std::pair<int, int> numExcitations;

  void assemble_X(RealType sqrtdt,
                  nda::MemoryArrayOfRank<2> auto&& X,
                  nda::MemoryArrayOfRank<1> auto&& MF,
                  nda::MemoryArrayOfRank<1> auto&& HWs,
                  int nt = 0,
                  bool addRAND = true);

  template<char TA, class WlkSet, typename VHS_t>
  void apply_propagators(WlkSet& wset, VHS_t const& v, bool P1inv = false)  
  {
    if(P1inv) {
      if(denseP1) {
        det_ops::Propagate<MEM,TA>(wset,P1d_inv(),v,order);
      } else {
        det_ops::Propagate<MEM,TA>(wset,P1s_inv(),v,order);
      }
    } else {
      if(denseP1) {
        det_ops::Propagate<MEM,TA>(wset,P1d(),v,order);
      } else {
        det_ops::Propagate<MEM,TA>(wset,P1s(),v,order);
      }
    }
  }

  template<char TA, typename VHS_t>
  void apply_propagators(WALKER_TYPES wtype, int npol, 
                         nda::MemoryArrayOfRank<3> auto&& Xa, 
                         nda::MemoryArrayOfRank<3> auto&& Xb, 
                         VHS_t const& v, bool P1inv = false)
  {
    if(P1inv) {
      if(denseP1) {
        if(wtype == COLLINEAR)
          det_ops::Propagate<MEM,TA>(wtype,npol,Xa,Xb,P1d_inv(),v,order);
        else
          det_ops::Propagate<MEM,TA>(wtype,npol,Xa,P1d_inv(),v,order);
      } else {
        if(wtype == COLLINEAR)
          det_ops::Propagate<MEM,TA>(wtype,npol,Xa,Xb,P1s_inv(),v,order);
        else
          det_ops::Propagate<MEM,TA>(wtype,npol,Xa,P1s_inv(),v,order);
      }
    } else {
      if(denseP1) {
        if(wtype == COLLINEAR)
          det_ops::Propagate<MEM,TA>(wtype,npol,Xa,Xb,P1d(),v,order);
        else
          det_ops::Propagate<MEM,TA>(wtype,npol,Xa,P1d(),v,order);
      } else {
        if(wtype == COLLINEAR)
          det_ops::Propagate<MEM,TA>(wtype,npol,Xa,Xb,P1s(),v,order);
        else
          det_ops::Propagate<MEM,TA>(wtype,npol,Xa,P1s(),v,order);
      }
    }
  }


/*
  template<class WlkSet>
  void Orthogonalize_excited_impl(WlkSet& wset);
*/

};

} // namespace afqmc

} // namespace sfqmc

#include "AFQMCBasePropagator.icc"

