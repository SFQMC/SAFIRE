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
#include "AFQMC/parameters.hpp"
#include "AFQMC/Utilities/readWfn.h"
#include "AFQMC/Utilities/type_conversion.hpp"

#include "AFQMC/HamiltonianOperations/HamiltonianOperations.h"

#include "AFQMC/Wavefunctions/phmsd_helpers.hpp"
#include "AFQMC/Wavefunctions/Excitations.hpp"
#include "AFQMC/Walkers/WalkerSet.hpp"

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
  PHMSD(const WavefunctionParameters& params,
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
  PHMSD(const WavefunctionParameters& params,
        WALKER_TYPES wlk,
        int NMO_, int nup_, int ndown_,
        std::shared_ptr<utils::mpi_context_t<mpi3::communicator>> mpi_,
        HamiltonianOperations<MEM>&& hop_,
        ph_excitations<int, ComplexType, MEM>&& abij_,
        nda::array<csrM,1>&& op_spin_det_coupling_,
        nda::array<csrM,1>&& orbs_,
        [[maybe_unused]] int targetNW = 1)
      : mpi(mpi_),
        walker_type(wlk),
        NMO{NMO_}, nup{nup_}, ndown{ndown_},
        HamOp(std::move(hop_)),
        abij(std::move(abij_)),
        OpSpinDetCouplings(std::move(op_spin_det_coupling_)),
        OrbMats(std::move(orbs_))
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

    const int nspin = (walker_type==COLLINEAR ? 2 : 1 );
    utils::check(OrbMats.extent(0)==1 or OrbMats.extent(0)==nspin, "PHMSD: Invalid size of OrbMats");

    // setup device structures

    energy_algorithm = resolved(params.algorithm, "algorithm");

    app_log(1, "Using the {} energy algorithm. ", nlohmann::json(energy_algorithm).get<std::string>());
    // check that refc is appropriate for the selected algorithm
    if(energy_algorithm == PHMSDEnergyAlgorithm::woodbury) {
      auto refc=abij.reference_configuration();
      for(int i=0; i<nup; i++)
        utils::check(refc[i] == i, "Error: PHMSD woodbury algorithm requires refc[i]==i.\n\n");
      for(int i=0; i<ndown; i++)
        utils::check(refc[nup+i] == i, "Error: PHMSD woodbury algorithm requires refc[i]==i.\n\n");
    }

    nwalk_block_size = params.nwalk_block_size;
    ndet_block_size  = params.ndet_block_size;
    utils::check(nwalk_block_size > 0, " Error: PHMSD nwalk_block_size must be > 0.");
    utils::check(ndet_block_size  > 0, " Error: PHMSD ndet_block_size must be > 0.");
    app_log(1, "PHMSD energy batching: nwalk_block_size={}, ndet_block_size={}.",
            nwalk_block_size, ndet_block_size);
  }

  int number_of_cholesky_vectors() const { return HamOp.number_of_cholesky_vectors(); }

  WALKER_TYPES getWalkerType() const { return walker_type; }

  bool isFiniteTemperature() const { return false; }

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
  void vbias(WalkerSet<MEM>& wset, memory::array_view<MEM,ComplexType,2> v, double dt, int nt = 0);

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
  void Energy(WalkerSet<MEM>& wset, int nt = 0);

  /*
   * Calculates the local energy and overlaps of all the walkers in the set and 
   * returns them in the appropriate data structures
   */
  void Energy(WalkerSet<MEM> const& wset, memory::array_view<MEM,ComplexType,2> E, memory::array_view<MEM,ComplexType,1> Ov, int nt = 0);

  /*
   * Calculates the mixed density matrix for all walkers in the walker set. 
   * Options:
   *  - compact:   If true (default), returns compact form with Dim: [NEL*NMO], 
   *                 otherwise returns full form with Dim: [NMO*NMO]. 
   */
  void MixedDensityMatrix(WalkerSet<MEM> const& wset, memory::array_view<MEM,ComplexType,2> G, bool compact = true);

  void MixedDensityMatrix(WalkerSet<MEM> const& wset, memory::array_view<MEM,ComplexType,2> G, memory::array_view<MEM,ComplexType,1> Ov, bool compact = true);

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
  void Log_Overlap(WalkerSet<MEM> const& wset, memory::array_view<MEM,ComplexType,1> Ov, int nt = 0);

  /*
   * Calculates the overlaps of all walkers in the set. Updates values in wset. 
   */
  void Log_Overlap(WalkerSet<MEM>& wset);

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

  int getNMO() const { return NMO; }

  /*
   * Returns the reference Slater Matrices needed for back propagation.  
   */
  void getReferences(memory::buffered_array<MEM,ComplexType,3>& Refs) 
  {
    using nda::range;
    auto all = range::all;
    memory::check_memory_space<MEM>(Refs);
    int nel = nup + (walker_type == COLLINEAR ? ndown : 0);
    int nspin = walker_type == COLLINEAR ? 2 : 1;
    int nspin_in_wfn = OrbMats.extent(0);
    int npol = (walker_type == NONCOLLINEAR ? 2 : 1);

    int number_of_references = abij.number_of_configurations();
    Refs.resize(number_of_references, npol*NMO, nel);
    
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
            int conf_idx = (spin == 0) ? std::get<0>(*c) : std::get<1>(*c);
            abij.get_configuration(spin, conf_idx, Ac);
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

  PHMSDEnergyAlgorithm energy_algorithm{};

  // energy_shared_alg1 batching (optional wavefunction inputs; see interpret_inputs):
  //   nwalk_block_size : walkers processed per energy batch  (bounds KEright/Tdn)
  //   ndet_block_size  : determinants per excitation-shell block (bounds R/KEl)
  int nwalk_block_size = 8;
  int ndet_block_size  = 4096;

  ph_excitations<int, ComplexType, MEM> abij;

  // sparse matrix with opposite spin determinant couplings
  nda::array<PsiT_Matrix<MEM>,1> OpSpinDetCouplings;

  nda::array<PsiT_Matrix<MEM>,1> OrbMats;

  // store references for back propagation
  memory::const_shared_array<HOST_MEMORY,ComplexType,3> RefOrbMats;

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
  void energy_alg0(WalkerSet<MEM> const& wset, memory::array_view<MEM,ComplexType,2> E, memory::array_view<MEM,ComplexType,1> Ov);

  void energy_alg1(WalkerSet<MEM> const& wset, memory::array_view<MEM,ComplexType,2> E, memory::array_view<MEM,ComplexType,1> Ov);

};

} // namespace afqmc

} // namespace sfqmc

