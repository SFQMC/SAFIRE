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
#include <map>
#include <string>
#include <iostream>
#include <tuple>

#include "IO/app_loggers.h"
#include "IO/ptree/ptree_utilities.hpp"
#include "utilities/Random.hpp"
#include "utilities/check.hpp"

#include "AFQMC/config.h"

#include "numerics/shared_array/const_shared_array.hpp"
#include "AFQMC/Wavefunctions/Wavefunction.hpp"
#include "AFQMC/SlaterDeterminantOperations/propagate.hpp"

namespace sfqmc
{
namespace afqmc
{
/*
 * Base AFQMC propagator.
 * For all hamiltonians that only use a dense vHS. For model hamiltonians, use AFQMCModelPropagator.
 */
template<MEMORY_SPACE MEM>
class AFQMCBasePropagator : public AFQMCInfo
{

public:
  AFQMCBasePropagator() {
    utils::check(false, "Error: Reached disabled AFQMCBasePropagator default constructor.");
  }

  AFQMCBasePropagator(AFQMCInfo& info,
                      ptree pt_in,
                      std::shared_ptr<utils::mpi_context_t<mpi3::communicator>> mpi_,
                      Wavefunction<MEM>& wfn_,
                      std::shared_ptr<utils::RandomGenerator_t<MEM>> r)
      : AFQMCInfo(info),
        mpi(mpi_),
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
    std::tie(nspins_in_vHS, npol_in_vHS) = wfn->vHS_dims();
    app_log(1," vHS dimensions: nspins = {}, npol = {}", nspins_in_vHS, npol_in_vHS);
    // convert user input to verbose input
    ptree pt = interpret_inputs(pt_in);
    app_log(2,"\nBasePropagator input:\n\n{}\n",io::to_string(pt));
    // initialize using verbose input
    int i_, a_;
    std::string external_field, excited_file;
    double external_field_scale;
    i_ = pt.get<int>("i");
    a_ = pt.get<int>("a");
    order                = pt.get<int>("taylor_n");
    vbias_bound          = pt.get<double>("vbias_bound");
    external_field_scale = pt.get<double>("external_field_scale");
    upper_cutoff_scale = pt.get<double>("upper_cutoff_scale");
    lower_cutoff_scale = pt.get<double>("lower_cutoff_scale");
    apply_constrain     = pt.get<bool>("apply_constrain");
    importance_sampling = pt.get<bool>("importance_sampling");
    substractMF          = pt.get<bool>("substractMF");
    hybrid              = pt.get<bool>("hybrid");
    external_field      = pt.get<std::string>("external_field");
    printP1eV           = pt.get<bool>("printP1eigval");
    if(not mpi->comm.root()) printP1eV = false;
    free_projection     = pt.get<bool>("free_projection");
    denseP1             = pt.get<bool>("denseP1");
    denseP2             = pt.get<bool>("denseP2");
    excited_file        = pt.get<std::string>("excited");
    debug_verbosity     = pt.get<bool>("debug_verbosity");
    natural_shift       = pt.get<bool>("natural_shift");
    symmetric_split     = pt.get<bool>("symmetric_split");
    use_cp_constraint   = pt.get<bool>("use_cp_constraint");
    use_real_vbias      = pt.get<bool>("use_real_vbias");
    auto hamtype(wfn->getHamType());
    utils::check(denseP2 or hamtype == ModelHamiltonian, "denseP2=false only allowed with ModelHamiltonian.");

    if ((hamtype == KPFactorized || hamtype == KPTHC) && denseP1)
    {
      app_error("dense Ham. with kpoints");
      utils::check(false,"BasePropagator: set denseP1 to false");
    }

    app_log(1,"\n\n --------------- Constructing Propagator ------------------ \n");

    app_log(1," vbias_bound: {}", vbias_bound); 
    if(denseP1)
      app_log(1," Using dense 1-body propagator");
    else
      app_log(1," Using sparse 1-body propagator");
    if(denseP2)
      app_log(1," Using dense 2-body propagator (vHS)");
    else
      app_log(1," Using sparse 2-body propagator (vHS)");

    if(nspins_in_vHS>1) 
      app_log(1, " Using a spin-dependent vHS.");
    if(npol_in_vHS>1) 
      app_log(1, " Using a polarization-dependent vHS.");

    if (!importance_sampling && !free_projection)
      app_log(1," WARNING: importance_sampling=no without free projection does not make sense. ");

    if (hybrid)
      app_log(1," Using hybrid method to calculate the weights during the propagation.");
    else
      app_log(1," Using local energy method to calculate the weights during the propagation.");
    if(natural_shift)
      app_log(1, " Using natural shifts with discrete propagators. ");
    if(symmetric_split)
      app_log(1, " Using symmetric split of walker weight update.");

    if (debug_verbosity)
    {
      app_log(1,"[WARNING] Using debug verbosity. THIS WILL GENERATE A LOT OF OUTPUT.");
      app_log(1,"Intended for debugging purposes with a few walkers only.");
    }

    // read orbital matrix if excited state propagator
    excitedState = false;
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
    if (external_field != std::string(""))
    {
      //    read_external_field(H1ext);
      auto walker_type = wfn->getWalkerType();
      int npol  = ( walker_type == NONCOLLINEAR or walker_type == NONCOLLINEAR_FT ? 2 : 1 );
      int nspin = ( walker_type == COLLINEAR  or walker_type == COLLINEAR_FT  ? 2 : 1 );
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

  static ptree interpret_inputs(const ptree pt0)
  {
    // read inputs with default options
    int i_ = pt0.get<int>("i", -1);
    int a_ = pt0.get<int>("a", -1);
    auto taylor_n               = pt0.get<double>("taylor_n",6);
    double vbias_bound = pt0.get<double>("vbias_bound", 50.0);
    double external_field_scale = pt0.get<double>("external_field_scale", 1.0);
    double upper_cutoff_scale = pt0.get<double>("upper_cutoff_scale", 10.0);
    double lower_cutoff_scale = pt0.get<double>("lower_cutoff_scale", 1.0);
    bool apply_constrain        = pt0.get<bool>("apply_constrain", true);
    bool importance_sampling    = pt0.get<bool>("importance_sampling", true);
    bool substractMF            = pt0.get<bool>("substractMF", true);
    bool hybrid                 = pt0.get<bool>("hybrid", true);
    bool printP1eigval          = pt0.get<bool>("printP1eigval", false);
    bool free_projection        = pt0.get<bool>("free_projection", false);
    bool denseP1                = pt0.get<bool>("denseP1", false);
    bool denseP2                = pt0.get<bool>("denseP2", true);
    bool debug_verbosity        = pt0.get<bool>("debug_verbosity", false);
    auto natural_shift          = pt0.get<bool>("natural_shift",true);
    auto symmetric_split        = pt0.get<bool>("symmetric_split",false);
    auto use_cp_constraint      = pt0.get<bool>("use_cp_constraint", false);
    auto use_real_vbias         = pt0.get<bool>("use_real_vbias", false);
    std::string external_field  = pt0.get<std::string>("external_field", "");
    std::string excited_file    = pt0.get<std::string>("excited", "");
    // validate inputs
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
    // create verbose internal inputs
    ptree pt1;
    pt1.put("taylor_n", taylor_n);
    pt1.put("i", i_);
    pt1.put("a", a_);
    pt1.put("vbias_bound", vbias_bound);
    pt1.put("external_field_scale", external_field_scale);
    pt1.put("upper_cutoff_scale", upper_cutoff_scale);
    pt1.put("lower_cutoff_scale", lower_cutoff_scale);
    pt1.put("apply_constrain", apply_constrain);
    pt1.put("use_cp_constraint", use_cp_constraint);
    pt1.put("use_real_vbias", use_real_vbias);
    pt1.put("importance_sampling", importance_sampling);
    pt1.put("substractMF", substractMF);
    pt1.put("hybrid", hybrid);
    pt1.put("printP1eigval", printP1eigval);
    pt1.put("free_projection", free_projection);
    pt1.put("denseP1", denseP1);
    pt1.put("denseP2", denseP2);
    pt1.put("external_field", external_field);
    pt1.put("excited", excited_file);
    pt1.put("natural_shift",natural_shift);
    pt1.put("symmetric_split",symmetric_split);
    pt1.put("debug_verbosity", debug_verbosity);
    std::unordered_set<std::string> pass_through_keys = {
      "system",
      "name",
      "debug"
    };
    io::compare_known_keys("Propagator",pt1, pt0,pass_through_keys);
    return pt1;
  }

  ~AFQMCBasePropagator() {}

  AFQMCBasePropagator(AFQMCBasePropagator const& other) = delete;
  AFQMCBasePropagator& operator=(AFQMCBasePropagator const& other) = delete;
  AFQMCBasePropagator(AFQMCBasePropagator&& other)                 = default;
  AFQMCBasePropagator& operator=(AFQMCBasePropagator&& other) = delete;

  template<class WlkSet>
  void Propagate(WlkSet& wset, RealType E1, RealType dt, int nt = 0);

  // Phase 3c-i (StochasticWfn): drive ONE conditioned field-sampling step on the inner ensemble.
  // Seeds the auxiliary fields with an externally supplied per-walker bias Xbias(nwalk,nCV) (the
  // walker-conditioned force bias x_bar(phi_w)), then runs assemble_X -> vHS -> apply_propagators.
  // Deliberately omits the local-energy/overlap and walker-weight update: the resulting determinants
  // are field samples whose weights are not consumed by the StochasticWfn reductions. Requires the
  // propagator to be built with free_projection = false so assemble_X applies the supplied bias.
  template<class WlkSet>
  void Propagate_conditioned(WlkSet& wset, nda::MemoryArrayOfRank<2> auto const& Xbias, RealType dt, int nt = 0);

  // Free-projection field-sampling step, INDEPENDENT of the propagator's build mode. Draws bare
  // auxiliary fields Y ~ p_T(Y) (no force bias, no vMF shift -- toggles free_projection around
  // assemble_X) and applies B_T(Y) = exp(vHS) to the walkers, then stops. Like Propagate_conditioned it
  // skips the energy/overlap and walker-weight update: the results are field samples, not weighted
  // walkers. Used by StochasticWfn to draw walker-INDEPENDENT free-projection trial samples
  // {psi_p = B_T(Y^[p])|phi_T>} for back-propagation references, regardless of whether the forward inner
  // propagator was built for free projection (3b) or importance sampling (3c conditioning/leapfrog).
  template<class WlkSet>
  void Propagate_free(WlkSet& wset, RealType dt, int nt = 0);

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
  bool symmetric_split = false;
  bool use_cp_constraint = false;
  bool use_real_vbias = false;

  int nspins_in_vHS = 1;
  int npol_in_vHS   = 1;

  bool debug_verbosity = false;

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

