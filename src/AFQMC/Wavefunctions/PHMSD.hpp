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

#include "AFQMC/config.h"
#include "numerics/shared_array/const_shared_array.hpp"
#include "IO/ptree/ptree_utilities.hpp"
#include "AFQMC/Utilities/readWfn.h"
#include "AFQMC/Utilities/type_conversion.hpp"

#include "AFQMC/HamiltonianOperations/HamiltonianOperations.h"

#include "AFQMC/Wavefunctions/phmsd_helpers.hpp"
#include "AFQMC/Wavefunctions/Excitations.hpp"

namespace sfqmc
{
namespace afqmc
{
/*
 * Class that implements particle-hole multi-Slater determinant expansions trial wave-functions.
 * All determinants in the expansion are related to the reference determinant 
 * by a list of single particle excitations.
 */
template<MEMORY_SPACE MEM>
class PHMSD
{

public:
  // temporary
  PHMSD(ptree pt_in,
        WALKER_TYPES wlk,
        int NMO_, int nup_, int ndown_,
        std::shared_ptr<utils::mpi_context_t<mpi3::communicator>> mpi_,
        HamiltonianOperations<MEM>&& hop_)
      : mpi(mpi_),
        walker_type(wlk),
        NMO{NMO_}, nup{nup_}, ndown{ndown_},
        HamOp(std::move(hop_))
  {}

  template<class csrM>
  PHMSD(ptree pt_in,
        WALKER_TYPES wlk,
        int NMO_, int nup_, int ndown_,
        std::shared_ptr<utils::mpi_context_t<mpi3::communicator>> mpi_,
        HamiltonianOperations<MEM>&& hop_,
        ph_excitations<int, ComplexType, MEM>&& abij_,
        nda::array<csrM,1>&& op_spin_det_coupling_,
        nda::array<csrM,1>&& orbs_,
        ComplexType nce,
        [[maybe_unused]] int targetNW = 1)
      : mpi(mpi_),
        walker_type(wlk),
        NMO{NMO_}, nup{nup_}, ndown{ndown_},
        HamOp(std::move(hop_)),
        abij(std::move(abij_)),
        OpSpinDetCouplings(std::move(op_spin_det_coupling_)),
        OrbMats(std::move(orbs_)),
        NuclearCoulombEnergy(nce)
  {
    /* To me, PHMSD is not compatible with walker_type=CLOSED unless
     * the MSD expansion is symmetric with respect to spin. For this, 
     * it is better to write a specialized class that assumes either spin symmetry
     * or e.g. Perfect Pairing.
     */
    if (walker_type == CLOSED)
      APP_ABORT("Error: PHMSD requires walker_type != CLOSED. walker_type: {}", walkerTypeToString(walker_type));

    // FINISH!!!
    if (walker_type == NONCOLLINEAR)
      APP_ABORT("PHMSD has not yet been implemented for NONCOLLINEAR walkers.");

    if (walker_type == FULLYPOLARIZED)
      APP_ABORT("PHMSD has not yet been implemented for FULLYPOLARIZED walkers.");

    const int nspin = (walker_type==COLLINEAR ? 2 : 1 );
    utils::check(OrbMats.extent(0)==1 or OrbMats.extent(0)==nspin, "PHMSD: Invalid size of OrbMats");

    // setup device structures

    // convert user input to verbose input
    ptree pt = interpret_inputs(pt_in);
    app_log(2,"\nPHMSD input:\n{}\n",io::to_string(pt));
    // initialize using verbose input

    // optional
    if( auto val = pt.get_optional<int>("algorithm") ) {
      energy_algorithm = *val;
    } else {
      if(HamOp.getHamType() == RealDenseFactorized)
        energy_algorithm=1;  // add others as they get implemented...
      else 
        energy_algorithm=0;
    }

    if(energy_algorithm==0)
      app_log(1, " Using default (slow) energy algorithm. ");
    else if(energy_algorithm==1)
      app_log(1, " Using energy algorithm 1. ");
    else if(energy_algorithm==2)
      app_log(1, " Using energy algorithm 2. ");
    else
      APP_ABORT(" Error in PHMSD constructor: Unknown algorithm. \n\n");
    // check that refc is appropriate for the selected algorithm
    if(energy_algorithm==1) {
      auto refc=abij.reference_configuration();
      for(int i=0; i<nup; i++)
        if( refc[i] != i ) 
        utils::check(refc[i] == i, " Error: PHMSD algorithm=1 requires refc[i]==i.\n\n");
      for(int i=0; i<ndown; i++)
        utils::check(refc[nup+i] == i, " Error: PHMSD algorithm=1 requires refc[i]==i.\n\n");
    }
  }

  static ptree interpret_inputs(const ptree pt0)
  {
    // read inputs with default options
    // create verbose internal inputs
    ptree pt1;
    // leave as a true optional, to bypass issue with default value
    if( auto val = pt0.get_optional<int>("algorithm") )
      pt1.put("algorithm", *val);
    std::unordered_set<std::string> pass_through_keys = {
      "system",
      "name",
      "ndets_to_read",
      "restart_file",
      "filename",
      "rediag"
    };
    io::compare_known_keys("particle-hole multi-Slater det. (PHMSD) Wavefunction", pt1, pt0,pass_through_keys);
    return pt1;
  }

  int number_of_cholesky_vectors() const { return HamOp.number_of_cholesky_vectors(); }

  WALKER_TYPES getWalkerType() const { return walker_type; }

  /*
   *  Performs runtime optimizations.
   */
  template<class WlkSet>
  void runtime_optimization(WlkSet& wset)
  {
    const int nw   = wset.size();
    const int nel = (walker_type==COLLINEAR ? nup+ndown : nup );
    const int npol = (walker_type==NONCOLLINEAR ? 2 : 1 );
    memory::array<MEM,ComplexType,2> G(nw,nel*npol*NMO);
    // don't use buffered_array!!!
// This needs to depend on algorithm!!!
    HamOp.runtime_optimization(G);
  }

  /*
   * Returns the memory space.
   */
  constexpr auto get_memory_space() const { return MEM; }

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
  void vbias(WlkSet& wset, nda::MemoryMatrix auto && v, double dt, int nt = 0)
  { 
    memory::check_memory_space<MEM>(v);
    AFQMCTimer.start(G_for_vbias_timer);
    int nact  = OrbMats(0).extent(0) + (walker_type==COLLINEAR ? OrbMats(OrbMats.extent(0)-1).extent(0) : 0);
    int npol  = (walker_type==NONCOLLINEAR ? 2 : 1);
    int nw = wset.size();
    utils::check(v.shape() == std::array<long,2>{nw,HamOp.number_of_cholesky_vectors()},
                 "Shape mismatch");
    memory::buffered_array<MEM,ComplexType,2> G(nw,nact*npol*NMO);
    memory::buffered_array<MEM,ComplexType,1> ovlp(nw);
    MixedDensityMatrix(wset, G, ovlp);
    AFQMCTimer.stop(G_for_vbias_timer);
    AFQMCTimer.start(vbias_timer);
    v() = ComplexType(0.0);
    HamOp.vbias(G, v, dt);
    AFQMCTimer.stop(vbias_timer);
  }

  /*
   * Returns the Hubbard-Stratonovich potential. 
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
  void Energy(WlkSet& wset, int nt = 0)
  {
    auto all = nda::range::all;
    int nw = wset.size();
    memory::buffered_array<MEM,ComplexType,1> ovlp(nw,ComplexType(0.0));
    memory::buffered_array<MEM,ComplexType,2> eloc(nw,3);
    eloc() = ComplexType(0.0);
    Energy(wset, eloc(), ovlp());
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
  void Energy(const WlkSet& wset, TMat&& E, TVec&& Ov, int nt = 0)
  {
    if(energy_algorithm==0)
      energy_alg0(wset,E,Ov);
    else if(energy_algorithm==1)
      energy_alg1(wset,E,Ov);
    else if(energy_algorithm==2)
      energy_alg2(wset,E,Ov);
    else
      utils::check(false," Error: Unknown energy_algorithm. \n\n");
  }

  /*
   * Calculates the mixed density matrix for all walkers in the walker set. 
   * Options:
   *  - compact:   If true (default), returns compact form with Dim: [NEL*NMO], 
   *                 otherwise returns full form with Dim: [NMO*NMO]. 
   */
  template<class WlkSet, nda::MemoryMatrix MatG>
  void MixedDensityMatrix(const WlkSet& wset, MatG&& G, bool compact = true)
  {
    int nw = wset.size();
    memory::buffered_array<MEM,ComplexType,1> ovlp(nw,ComplexType(0.0));
    MixedDensityMatrix(wset, std::forward<MatG>(G), ovlp, compact);
  }

  template<class WlkSet, nda::MemoryMatrix MatG, nda::MemoryVector TVec>
  void MixedDensityMatrix(const WlkSet& wset, MatG&& G, TVec&& Ov, bool compact = true);

  /*
   * Calculates the density matrix with respect to a given Reference
   * for all walkers in the walker set. 
   */
  template<class WlkSet, nda::MemoryVector RVec, nda::MemoryMatrix MatG, nda::MemoryVector TVec>
  void DensityMatrix(const WlkSet& wset, RVec&& Ref, MatG&& G, TVec&& Ov,
                     bool compact = true, bool herm = true)
  {}

  /*
   * Calculates the overlaps of all walkers in the set. Returns values in arrays. 
   */
  template<class WlkSet, nda::MemoryArrayOfRank<1> TVec>
  void Log_Overlap(const WlkSet& wset, TVec && Ov, int nt = 0);

  /*
   * Calculates the overlaps of all walkers in the set. Updates values in wset. 
   */
  template<class WlkSet>
  void Log_Overlap(WlkSet& wset)
  {
    int nw = wset.size();
    memory::buffered_array<MEM,ComplexType,1> ovlp(nw,ComplexType(0.0));
    Log_Overlap(wset, ovlp);
    wset.setProperty(OVLP, ovlp);
  }

  template<class WlkSet, class Observable>
  void accumulate_estimators(int iav, WlkSet& wset, nda::MemoryVector auto const& wgt,
        std::vector<Observable>& properties_1body, std::vector<Observable>& properties, 
        nda::MemoryArrayOfRank<4> auto* X, nda::MemoryArrayOfRank<4> auto* Yc, 
        nda::MemoryArrayOfRank<4> auto* M, bool time_evolved, bool importanceSampling=true)
  {
    utils::check(false,"finish");
  }

  /*
   * Calculates Green functions and calls Observables.
   */
  template<class WlkSet, class Observable>
  void accumulate_estimators(int iav, WlkSet& wset, nda::MemoryVector auto const& wgt,
        std::vector<Observable>& properties_1body,
        std::vector<Observable>& properties, bool importanceSampling = true)
  {
    memory::buffered_array<MEM,ComplexType,4> *X = nullptr;
    accumulate_estimators(iav,wset,wgt,properties_1body,properties,X,X,X,false,importanceSampling);
  }

  ComplexType getReferenceWeight(int i) const { return std::get<2>(*abij.configuration(i)); }

  int total_number_of_references() const { return abij.number_of_configurations(); } 

  /*
   * Returns the reference Slater Matrices needed for back propagation.  
   */
  void getReferences(int number_of_references, nda::MemoryArrayOfRank<3> auto&& Refs) 
  {
    using nda::range;
    auto all = range::all;
    memory::check_memory_space<MEM>(Refs);
    int nel = nup + (walker_type == COLLINEAR ? ndown : 0);
    int nspin = walker_type == COLLINEAR ? 2 : 1;
    int nspin_in_wfn = OrbMats.extent(0);
    int npol = (walker_type == NONCOLLINEAR ? 2 : 1);
    if(number_of_references==0) return;
    if(number_of_references < 0) number_of_references = total_number_of_references(); 
    utils::check(number_of_references > 0 and
                 number_of_references <= Refs.extent(0),
                 "Invalid number_of_references: {} should fulfill 0 < n <= {}!", number_of_references, Refs.extent(0));
    utils::check(Refs.extent(1) == npol*NMO and Refs.extent(2) == nel, "Size mismatch");
    if (RefOrbMats.extent(0) < number_of_references)
    {
      RefOrbMats = memory::share_from_root(*mpi, [&] {
        nda::array<ComplexType,3> R(number_of_references,npol*NMO,nel);
        R() = ComplexType(0.0);
        std::array nels = {nup, ndown};
        std::array spin_offset = {0, nup};
        for(int spin = 0; spin < nspin; spin++) {
          int spin_ = spin % nspin_in_wfn;
          auto psi = nda::to_host(math::sparse::to_array<'N'>(OrbMats(spin_)));
          nda::vector<int> Ac(nels[spin]);
          for (int i_det = 0; i_det < number_of_references; ++i_det) {
            auto c=abij.configuration(i_det);
            abij.get_configuration(spin, std::get<0>(*c), Ac);
            for (int a = 0; a < nels[spin]; ++a) {
              R(i_det,all,spin_offset[spin]+a) = nda::conj(psi(Ac(a),all));
            }
          }
        }
        return R;
      });
    }
    utils::check(RefOrbMats.extent(0) >= number_of_references and
                 RefOrbMats.extent(1) == npol*NMO and RefOrbMats.extent(2) == nel,
                 "Problems with RefOrbMats");
    // this is slow and uses too much memory. Improve!!!
    Refs() = RefOrbMats()(range(number_of_references),all,all);
  }

  void updateLogScale(auto scl_new, SpinTypes s)
  {
    utils::check(false, "updateLogScale is not compatible with ground state calculations");
  }

  void resetLogScale()
  {
    utils::check(false, "resetLogScale is not implemented for ground state calculations");
  }

  ComplexType getLogScale(SpinTypes s)
  {
    utils::check(false, "getLogScale is not implemented for ground state calculations");
    return ComplexType(0.0);
  }

  void setLogPT0(nda::MemoryArrayOfRank<1> auto&& v)
  {
    utils::check(false, "setLogPT0 is not implemented for ground state calculations");
  }

  auto getLogPT0() const {
    utils::check(false, "getLogPT0 is not implemented for ground state calculations");
    return memory::array<MEM,ComplexType,1>{};
  }

protected:

  std::shared_ptr<utils::mpi_context_t<mpi3::communicator>> mpi;

  // type of walker/wfn
  WALKER_TYPES walker_type{};
  int NMO{};
  int nup{};
  int ndown{};

  HamiltonianOperations<MEM> HamOp;

  // MAM: use enum when this is settled...
  // 0: loop over unique configurations, calculate G and evaluate E from scratch
  // 1: use ph_reference_energy and ph_excited_energy, which requires compact R matrix
  // 2: calculate Fapbq and call ph_energy_Fapbq
  int energy_algorithm = 0;

  ph_excitations<int, ComplexType, MEM> abij; 

  // sparse matrix with opposite spin determinant couplings
  nda::array<PsiT_Matrix<MEM>,1> OpSpinDetCouplings;

  nda::array<PsiT_Matrix<MEM>,1> OrbMats;

  // store references for back propagation
  memory::const_shared_array<HOST_MEMORY,ComplexType,3> RefOrbMats;

  ComplexType NuclearCoulombEnergy;

  /*
   * Node-shared dense (daggered) copies of the orbital matrices.
   */
  auto dense_orbs()
  {
    std::vector<memory::host_const_shared_array<ComplexType,2>> Orbs;
    Orbs.reserve(OrbMats.size());
    for(int i=0; i<OrbMats.size(); ++i) {
      Orbs.emplace_back(memory::share_from_root(*mpi, [&] {
        return nda::to_host(math::sparse::to_array<'H'>(OrbMats(i)));
      }));
    }
    return Orbs;
  }

  /* Implementation of various energy evaluation algorithms. */
  template<class WlkSet,  nda::MemoryMatrix Mat, nda::MemoryVector TVec>
  void energy_alg0(const WlkSet& wset, Mat&& E, TVec&& Ov);

  template<class WlkSet,  nda::MemoryMatrix Mat, nda::MemoryVector TVec>
  void energy_alg1(const WlkSet& wset, Mat&& E, TVec&& Ov);

  template<class WlkSet,  nda::MemoryMatrix Mat, nda::MemoryVector TVec>
  void energy_alg2(const WlkSet& wset, Mat&& E, TVec&& Ov)
  { utils::check(false, "not implemented"); }

};

} // namespace afqmc

} // namespace sfqmc

#include "AFQMC/Wavefunctions/PHMSD.icc"

