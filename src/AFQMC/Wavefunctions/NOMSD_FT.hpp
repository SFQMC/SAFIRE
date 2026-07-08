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

#include <iostream>
#include <vector>
#include <string>
#include <tuple>

#include "AFQMC/config.h"
#include "numerics/shared_array/const_shared_array.hpp"
#include "utilities/check.hpp"
#include "utilities/check_strides.hpp"
#include "IO/ptree/ptree_utilities.hpp"
#include "AFQMC/Utilities/AFQMCTimer.h"
#include "AFQMC/Walkers/WalkerConfig.hpp"

#include "AFQMC/HamiltonianOperations/HamiltonianOperations.h"
//#include "AFQMC/SlaterDeterminantOperations/

namespace sfqmc
{
namespace afqmc
{
/*
 * Class that implements a multi-Slater determinant trial wave-function.
 * Single determinant wfns are also allowed. 
 * No relation between different determinants in the expansion is assumed.
 * Designed for non-orthogonal MSD expansions. 
 * For particle-hole orthogonal MSD wfns, use FastMSD.
 */
template<MEMORY_SPACE MEM, class devPsiT>
class NOMSD_FT
{

public:

  NOMSD_FT() {
    utils::check(false,"Default constructor for NOMSD disabled.");
  }

  NOMSD_FT(ptree pt_in,
        int NMO_, int ntau_,
        WALKER_TYPES wlk,
        std::shared_ptr<utils::mpi_context_t<mpi3::communicator>> mpi_,
        HamiltonianOperations<MEM>&& hop_,
        nda::array<ComplexType,1>&& ci_,
        nda::array<devPsiT,3>&& orbs_,
        ComplexType nce,
        [[maybe_unused]] int targetNW = 1)
      : mpi(mpi_),
        walker_type(wlk),
        NMO{NMO_}, ntau{ntau_},
        HamOp(std::move(hop_)),
        ci(std::move(ci_)),
        OrbMats(std::move(orbs_)),
        RefOrbMats(0, 0, 0, 0),
        NuclearCoulombEnergy(nce),
        sclL(walker_type == COLLINEAR ? 2: 1)
  {

    //std::cout<<"OrbMats(0,1,1)"<<std::endl;

    utils::check(OrbMats.extent(0) == ci.size(), "Size mismatch");
    // convert user input to verbose input
    ptree pt = interpret_inputs(pt_in);
    app_log(2,"\nNOMSD input:\n{}\n",io::to_string(pt));
    // initialize using verbose input
    bool rediag;
    rediag = pt.get<bool>("rediag");

    if (rediag) {
      utils::check(false,"finish");
      //recompute_ci();
    }
    
    resetLogScale();

  }

  static ptree interpret_inputs(const ptree pt0)
  {
    // read inputs with default options
    auto rediag      = pt0.get<bool>("rediag", false);
    auto dense_trial = pt0.get<bool>("dense_trial", true);
    // create verbose internal inputs
    ptree pt1;
    pt1.put("rediag", rediag);
    pt1.put("dense_trial", dense_trial);
    std::unordered_set<std::string> pass_through_keys = {
      "system",
      "name",
      "ndets_to_read",
      "filename",
      "compute",
      "dense_trial"
    };
    io::compare_known_keys("Non-orthogonal multi-Slater det. (NOMSD) Wavefunction",pt1, pt0,pass_through_keys);
    return pt1;
  }

  ~NOMSD_FT() = default; 

  NOMSD_FT(NOMSD_FT const& other) = delete;
  NOMSD_FT& operator=(NOMSD_FT const& other) = delete;
  NOMSD_FT(NOMSD_FT&& other)                 = default;
  NOMSD_FT& operator=(NOMSD_FT&& other) = delete;

  int number_of_cholesky_vectors() const { return HamOp.number_of_cholesky_vectors(); }

  WALKER_TYPES getWalkerType() const { return walker_type; }

  bool isFiniteTemperature() const { return true; }

  /*
   * Returns the memory space.
   */
  constexpr auto get_memory_space() const { return MEM; }

  /*
   *  Performs runtime optimizations.
   */       
  template<class WlkSet>
  void runtime_optimization(WlkSet& wset)
  {
    const int nw   = wset.size();
    const int nspin = walker_type==COLLINEAR ? 2 : 1;
    const int npol = walker_type==NONCOLLINEAR ? 2 : 1;
    memory::array<MEM,ComplexType,2> G(nw,nspin*npol*NMO*npol*NMO);
    // don't use buffered_array!!!
    HamOp.runtime_optimization(G);
  }

  /*
   * Expectation value of Hubbard-Stratonovich potential with respect to trial wave-function.
   */
  void vMF(nda::MemoryVector auto&& v, double dt);

  /*
   * Green function of the trial wave-funtion. 
   */
  auto G_MF();

  template<class... Args>
  void generalizedFockMatrix(Args&&... args)
  {
    HamOp.generalizedFockMatrix(std::forward<Args>(args)...); 
  }

  HamiltonianTypes getHamType() const { return HamOp.getHamType(); }

  auto getFieldTypes()
  {
    return HamOp.getFieldTypes();
  }

  template<class... Args>
  void update_potentials(Args&&... args)
  {
    HamOp.update_potentials(std::forward<Args>(args)...);
  }

  auto vHS_dims() const { return HamOp.vHS_dims(); }

  template<class... Args>
  auto getOneBodyPropagatorMatrix(Args&&... args)
  {
    return HamOp.getOneBodyPropagatorMatrix(std::forward<Args>(args)...);
  }

  template<class... Args>
  auto vHS_sparse(Args&&... args)
  {
    return HamOp.vHS_sparse(std::forward<Args>(args)...);
  }

  /*
   * Calculates the bias potential.
   */
  
  template<class WlkSet>
  void vbias(WlkSet& wset, nda::MemoryMatrix auto && v, double dt, int nt = -1)
  {
    memory::check_memory_space<MEM>(v);
    AFQMCTimer.start(G_for_vbias_timer);
    int nspin = walker_type==COLLINEAR ? 2 : 1;
    int npol  = walker_type==NONCOLLINEAR ? 2 : 1;
    int nw = wset.size();
    nt = nt < 0? wset.getTauStep() : nt; // if time-slice is not given, use current time-slice
    int nc = nspin*npol*NMO*npol*NMO;
    utils::check(v.shape() == std::array<long,2>{nw,HamOp.number_of_cholesky_vectors()}, 
                 "Shape mismatch");
    memory::buffered_array<MEM,ComplexType,2> G(nw,nc);
    memory::buffered_array<MEM,ComplexType,1> ovlp(nw);
    MixedDensityMatrix(wset, G, ovlp, nt);
    AFQMCTimer.stop(G_for_vbias_timer);
    AFQMCTimer.start(vbias_timer);
    v() = ComplexType(0.0);
    HamOp.vbias(G, v, dt);
    AFQMCTimer.stop(vbias_timer);
  }
  

  /*
   * Returns the Hubbard-Stratonovoch potential. 
   *  Input:
   *    - X: [# chol vecs][nW]
   *  Output:
   *    - vHS
   */
  template<nda::MemoryMatrix X>
  auto vHS(X && x, double dt )
  {
    utils::check(x.extent(1) == HamOp.number_of_cholesky_vectors(), "Shape mismatch");
    return HamOp.vHS(std::forward<X>(x), dt);
  }

  /*
   * Calculates the local energy and overlaps of all the walkers in the set and stores
   * them in the wset data
   */
  template<class WlkSet>
  void Energy(WlkSet& wset, int nt = -1)
  {
    auto all = nda::range::all;
    int nw = wset.size();
    memory::buffered_array<MEM,ComplexType,1> ovlp(nw,ComplexType(0.0));
    memory::buffered_array<MEM,ComplexType,2> eloc(nw,3);
    eloc() = ComplexType(0.0);
    Energy(wset, eloc(), ovlp(), nt);
    wset.setProperty(OVLP, ovlp);
    wset.setProperty(E1_, eloc(all, 0));
    wset.setProperty(EXX_, eloc(all, 1));
    wset.setProperty(EJ_, eloc(all, 2));
  }

  /*
   * Calculates the local energy and overlaps of all the walkers in the set and 
   * returns them in the appropriate data structures
   */
  template<class WlkSet,  nda::MemoryMatrix TMat, nda::MemoryVector TVec>
  void Energy(const WlkSet& wset, TMat&& E, TVec&& Ov, int nt = -1);

  /*
   * Calculates the mixed density matrix for all walkers in the walker set. 
   * Options:
   *  - compact:   If true (default), returns compact form with Dim: [NEL*NMO], 
   *                 otherwise returns full form with Dim: [NMO*NMO]. 
   */ 
  template<class WlkSet, nda::MemoryMatrix MatG>
  void MixedDensityMatrix(const WlkSet& wset, MatG&& G, int nt)
  {
    int nw = wset.size();
    memory::buffered_array<MEM,ComplexType,1> ovlp(nw,ComplexType(0.0));
    MixedDensityMatrix(wset, std::forward<MatG>(G), ovlp, nt);
  }

  template<class WlkSet, nda::MemoryMatrix MatG, nda::MemoryVector TVec>
  void MixedDensityMatrix(const WlkSet& wset, MatG&& G, TVec&& Ov, int nt);

  /*
   * Calculates the density matrix with respect to a given Reference
   * for all walkers in the walker set. 
   */
  template<class WlkSet, nda::MemoryMatrix RVec, nda::MemoryMatrix MatG, nda::MemoryVector TVec>
  void DensityMatrix(const WlkSet& wset, RVec&& Ref, MatG&& G, TVec&& Ov, int nt);

  /*
   * Calculates the overlaps of all walkers in the set. Returns values in arrays. 
   */
  template<class WlkSet, nda::MemoryArrayOfRank<1> TVec>
  void Log_Overlap(const WlkSet& wset, TVec && Ov, int nt = -1);

  /*
   * Calculates the overlaps of all walkers in the set. Updates values in wset. 
   */
  template<class WlkSet>
  void Log_Overlap(WlkSet& wset, int nt = -1)
  {
    int nw = wset.size();
    memory::buffered_array<MEM,ComplexType,1> ovlp(nw,ComplexType(0.0));
    Log_Overlap(wset, ovlp, nt);
    wset.setProperty(OVLP, ovlp);
  }

  /*
   * Calculates Green functions and calls Observables.
   */
  template<class WlkSet, class Observable>
  void accumulate_estimators(int iav, WlkSet& wset, nda::MemoryVector auto const& wgt,
        std::vector<Observable>& properties_1body, std::vector<Observable>& properties, 
        nda::MemoryArrayOfRank<4> auto* X, nda::MemoryArrayOfRank<4> auto* Yc, 
        nda::MemoryArrayOfRank<4> auto* M, bool time_evolved, bool importanceSampling=true);
  
  /*
   * Calculates Green functions and calls Observables.
   */
  template<class WlkSet, class Observable>
  void accumulate_estimators(int iav, WlkSet& wset, nda::MemoryVector auto const& wgt,
        std::vector<Observable>& properties_1body,
        std::vector<Observable>& properties, bool importanceSampling = true)
  {
    memory::buffered_array<MEM,ComplexType,4> *X = nullptr;
    accumulate_estimators(iav,wset,wgt,properties_1body,properties,X,X,X,false,importanceSampling);//,nt);
  }

  /*
   * Calculates Green functions and calls Observables.
   */
  /*
  template<class WlkSet, class Observable>
  void accumulate_estimators(int iav, WlkSet& wset, nda::MemoryVector auto const& wgt,
        std::vector<Observable>& properties_1body, std::vector<Observable>& properties, 
        bool importanceSampling=true, int nt=0);
  */
  ComplexType getReferenceWeight(int i) const { 
    utils::check(false, "back propagation not implemented for finite-T");
    return 0;
  }

  int total_number_of_references() const { 
    utils::check(false, "back propagation not implemented for finite-T");
    return 0;
  }

  void getReferences(int number_of_references, nda::MemoryArrayOfRank<3> auto&& Refs) const
  {
    utils::check(false, "back propagation not implemented for finite-T");
  }

  // FIX: multi-determinant finiteT wfns?
  void updateLogScale(auto scl_new, SpinTypes s)
  {
    if(s==Alpha)
      sclL(0) = scl_new;
    else
      sclL(1) = scl_new;
  }

  auto getLogScale(SpinTypes s){
    if(MEM==DEVICE_MEMORY){
      auto scl_h = nda::to_host(sclL);
      if(s==Alpha)
        return scl_h(0);
      else
        return scl_h(1);
    }
    else{
      if(s==Alpha)
        return sclL(0);
      else
        return sclL(1);
    }
  }

  void resetLogScale(){
    sclL() = ComplexType(0.0);
  }

  void setLogPT0(nda::MemoryArrayOfRank<1> auto&& v)
  {
    LogPT0 = v;
  }

  auto const& getLogPT0() const {return LogPT0;}
  
protected:
  std::shared_ptr<utils::mpi_context_t<mpi3::communicator>> mpi;

  // type of walker/wfn
  WALKER_TYPES walker_type;
  int NMO{};
  int ntau{};

  HamiltonianOperations<MEM> HamOp;

  nda::array<ComplexType, 1> ci;

  // OrbMats[ndet][nspin][3](nel,NMO)
  nda::array<devPsiT,3> OrbMats;

  // RefOrbMats[ndet][nspin][nel][NMO]
  memory::array<MEM,ComplexType,4> RefOrbMats;

  memory::array<MEM,ComplexType,1> LogPT0;

  ComplexType NuclearCoulombEnergy;

  // log scale for DL
  nda::array<ComplexType, 1> sclL;

/*
  void recompute_ci();

  template<class WlkSet, class TVec, class Mat1, class Mat2, class Mat3, class Observable>
  void accumulate_estimators_general_impl(int iav, WlkSet& wset, TVec& wgt, 
        std::vector<Observable>& properties_1body, std::vector<Observable>& properties,
        Mat1 const& X, Mat2 const& Y, Mat3 const& M, bool time_evolved, bool importanceSampling);

  template<class WlkSet, class TVec, class Mat1, class Mat2, class Mat3, class Observable>
  void accumulate_estimators_single_ref_impl(int iav, WlkSet& wset, TVec& wgt, 
        std::vector<Observable>& properties_1body, std::vector<Observable>& properties,
        Mat1 const& X, Mat2 const& Y, Mat3 const& M, bool time_evolved, bool importanceSampling);
*/
};

} // namespace afqmc

} // namespace sfqmc

#include "AFQMC/Wavefunctions/NOMSD_FT.icc"

