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
#include "AFQMC/parameters.hpp"
#include "AFQMC/Utilities/AFQMCTimer.h"
#include "AFQMC/Walkers/WalkerConfig.hpp"

#include "AFQMC/HamiltonianOperations/HamiltonianOperations.h"
#include "AFQMC/SlaterDeterminantOperations/density_matrix.hpp"
#include "AFQMC/Walkers/WalkerSet.hpp"
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

  NOMSD_FT() = delete;

  NOMSD_FT(const WavefunctionParameters& params,
        int NMO_, int ntau_,
        WALKER_TYPES wlk,
        std::shared_ptr<utils::mpi_context_t<mpi3::communicator>> mpi_,
        HamiltonianOperations<MEM>&& hop_,
        nda::array<ComplexType,1>&& ci_,
        nda::array<devPsiT,3>&& orbs_,
        [[maybe_unused]] int targetNW = 1)
      : mpi(mpi_),
        walker_type(wlk),
        NMO{NMO_}, ntau{ntau_},
        HamOp(std::move(hop_)),
        ci(std::move(ci_)),
        OrbMats(std::move(orbs_)),
        RefOrbMats(0, 0, 0, 0),
        sclL(walker_type == COLLINEAR ? 2: 1)
  {

    //std::cout<<"OrbMats(0,1,1)"<<std::endl;

    utils::check(OrbMats.extent(0) == ci.size(), "Size mismatch");
    
    resetLogScale();

  }

  NOMSD_FT(NOMSD_FT const& other) = delete;
  NOMSD_FT& operator=(NOMSD_FT const& other) = delete;
  NOMSD_FT(NOMSD_FT&& other)                 = default;
  NOMSD_FT& operator=(NOMSD_FT&& other) = default;

  int number_of_cholesky_vectors() const { return HamOp.number_of_cholesky_vectors(); }

  RealType energy_offset() const { return HamOp.energy_offset(); }

  WALKER_TYPES getWalkerType() const { return walker_type; }

  bool isFiniteTemperature() const { return true; }

  int getNMO() const { return NMO; }

  /*
   * Returns the memory space.
   */
  constexpr auto get_memory_space() const { return MEM; }

  /*
   *  Performs runtime optimizations.
   */       
  void runtime_optimization(WalkerSet<MEM>& wset);

  /*
   * Expectation value of Hubbard-Stratonovich potential with respect to trial wave-function.
   */
  void vMF(memory::array_view<MEM,ComplexType,1> v, double dt);

  /*
   * Green function of the trial wave-funtion. 
   */
  memory::const_shared_array<HOST_MEMORY,ComplexType,3> G_MF();

  HamiltonianType getHamType() const { return HamOp.getHamType(); }

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
  
  void vbias(WalkerSet<MEM>& wset, memory::array_view<MEM,ComplexType,2> v, double dt, int nt = -1);
  

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
  void Energy(WalkerSet<MEM>& wset, int nt = -1);

  /*
   * Calculates the local energy and overlaps of all the walkers in the set and 
   * returns them in the appropriate data structures
   */
  void Energy(WalkerSet<MEM> const& wset, memory::array_view<MEM,ComplexType,2> E, memory::array_view<MEM,ComplexType,1> Ov, int nt = -1);

  /*
   * Calculates the mixed density matrix for all walkers in the walker set. 
   * Options:
   *  - compact:   If true (default), returns compact form with Dim: [NEL*NMO], 
   *                 otherwise returns full form with Dim: [NMO*NMO]. 
   */ 
  void MixedDensityMatrix(WalkerSet<MEM> const& wset, memory::array_view<MEM,ComplexType,2> G, int nt);

  void MixedDensityMatrix(WalkerSet<MEM> const& wset, memory::array_view<MEM,ComplexType,2> G, memory::array_view<MEM,ComplexType,1> Ov, int nt);

  /*
   * Calculates the density matrix with respect to a given Reference
   * for all walkers in the walker set. 
   */
  template<class WlkSet, nda::MemoryMatrix RVec, nda::MemoryMatrix MatG, nda::MemoryVector TVec>
  void DensityMatrix(const WlkSet& wset, RVec&& Ref, MatG&& G, TVec&& Ov, int nt);

  /*
   * Calculates the overlaps of all walkers in the set. Returns values in arrays. 
   */
  void Log_Overlap(WalkerSet<MEM> const& wset, memory::array_view<MEM,ComplexType,1> Ov, int nt = -1);

  /*
   * Calculates the overlaps of all walkers in the set. Updates values in wset. 
   */
  void Log_Overlap(WalkerSet<MEM>& wset, int nt = -1);

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

  void getReferences(nda::MemoryArrayOfRank<3> auto&& Refs) const
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

/* 
 * Computes the density matrix for a given reference. 
 * G and Ov are expected to be in shared memory.
 * Simple round-robin is used. 
 */
template<MEMORY_SPACE MEM, class devPsiT>
template<class WlkSet, nda::MemoryMatrix RMat, nda::MemoryMatrix MatG, nda::MemoryVector TVec>
void NOMSD_FT<MEM,devPsiT>::DensityMatrix(const WlkSet& wset, RMat&& Ref, MatG&& G, TVec&& Ov, int nt)
{
  // RVec is a vector of const_shared_array or CSRMatrix
  auto all = nda::range::all;
  memory::check_memory_space<MEM>(G,Ov);
  const int nw   = wset.size();
  const int nspin = walker_type==COLLINEAR ? 2 : 1;
  const int npol = walker_type==NONCOLLINEAR ? 2 : 1;
  utils::check(Ov.size() == nw, "Size mismatch");
  utils::check(Ref.extent(0) == nspin,"Size mismatch");
  utils::check(Ref(0,0).shape() == std::array<long,2>{npol*NMO,npol*NMO}, "Size mismatch");
  utils::check(Ref(0,1).shape() == std::array<long,2>{ntau,npol*NMO}, "Size mismatch");
  utils::check(Ref(0,2).shape() == std::array<long,2>{npol*NMO,npol*NMO}, "Size mismatch");
  if(nspin>1){
    utils::check(Ref(1,0).shape() == std::array<long,2>{npol*NMO,npol*NMO}, "Size mismatch");
    utils::check(Ref(1,1).shape() == std::array<long,2>{ntau,npol*NMO}, "Size mismatch");
    utils::check(Ref(1,2).shape() == std::array<long,2>{npol*NMO,npol*NMO}, "Size mismatch");
  }

  G() = ComplexType(0.0);
  Ov() = ComplexType(0.0);

  utils::check(G.shape() == std::array<long,2>{nw,nspin*npol*NMO*npol*NMO}, "Size mismatch");
  auto G3d = nda::reshape(G,std::array<long,3>{nw,nspin*npol*NMO,npol*NMO});

  memory::buffered_array<MEM,ComplexType,2> sclR(nspin,nw);
  wset.getProperty(LOGSCL_UP, sclR(0,nda::range::all));

  memory::buffered_array<MEM,ComplexType,1> sclL_up(1,sclL(0));

  using T = typename std::decay<decltype(Ref(0,1))>::type;
  // MixedDensityMatrix does not accept sparse trial wavefunctions
  if constexpr (math::sparse::CSRMatrix<T>){
    auto ULup_dense(math::sparse::to_array<'N'>(Ref(0,0)()));
    auto DLup_dense(math::sparse::to_array<'N'>(Ref(0,1)()));
    auto VLup_dense(math::sparse::to_array<'N'>(Ref(0,2)()));

    det_ops::MixedDensityMatrix(ULup_dense, DLup_dense(nt,nda::range::all), VLup_dense, 
         wset.UMatrices(Alpha), wset.DMatrices(Alpha), wset.VMatrices(Alpha),
                  G3d(all,nda::range(npol*NMO),all), Ov, sclL_up, sclR(0,nda::range::all));


  } else { 
    auto DLup_dense(Ref(0,1)());

    det_ops::MixedDensityMatrix(Ref(0,0)(), DLup_dense(nt,nda::range::all), Ref(0,2)(), 
         wset.UMatrices(Alpha), wset.DMatrices(Alpha), wset.VMatrices(Alpha),
                  G3d(all,nda::range(npol*NMO),all), Ov, sclL_up, sclR(0,nda::range::all));


  }

  if (walker_type == COLLINEAR){

    wset.getProperty(LOGSCL_DN, sclR(1,nda::range::all));
    memory::buffered_array<MEM,ComplexType,1> sclL_dn(1,sclL(1));

    // MixedDensityMatrix does not accept sparse trial wavefunctions
    if constexpr (math::sparse::CSRMatrix<T>){
      auto ULdn_dense(math::sparse::to_array<'N'>(Ref(1,0)()));
      auto DLdn_dense(math::sparse::to_array<'N'>(Ref(1,1)()));
      auto VLdn_dense(math::sparse::to_array<'N'>(Ref(1,2)()));

      det_ops::MixedDensityMatrix(ULdn_dense, DLdn_dense(nt,nda::range::all), VLdn_dense, 
         wset.UMatrices(Beta), wset.DMatrices(Beta), wset.VMatrices(Beta),
                  G3d(all,nda::range(npol*NMO,nspin*npol*NMO),all), Ov, sclL_dn, sclR(1,nda::range::all));

    } else { 
      auto DLdn_dense(Ref(1,1)());

      det_ops::MixedDensityMatrix(Ref(1,0)(), DLdn_dense(nt,nda::range::all), Ref(1,2)(), 
         wset.UMatrices(Beta), wset.DMatrices(Beta), wset.VMatrices(Beta),
                  G3d(all,nda::range(npol*NMO,nspin*npol*NMO),all), Ov, sclL_dn, sclR(1,nda::range::all));
    }

  }

}

} // namespace afqmc

} // namespace sfqmc

