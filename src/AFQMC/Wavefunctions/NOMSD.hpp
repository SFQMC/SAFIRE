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

#include "AFQMC/config.h"
#include "numerics/shared_array/const_shared_array.hpp"
#include "utilities/check.hpp"
#include "utilities/check_strides.hpp"
#include "AFQMC/parameters.hpp"
#include "numerics/sparse/csr_utils.hpp"
#include "AFQMC/Utilities/AFQMCTimer.h"
#include "AFQMC/Walkers/WalkerConfig.hpp"
#include "AFQMC/Walkers/WalkerSet.hpp"

#include "AFQMC/HamiltonianOperations/HamiltonianOperations.h"
#include "AFQMC/SlaterDeterminantOperations/density_matrix.hpp"
#include "numerics/operations/product.hpp"

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
class NOMSD
{

public:

  NOMSD() = delete;

  NOMSD(const WavefunctionParameters& params,
        int NMO_, int nup_, int ndown_,
        WALKER_TYPES wlk,
        std::shared_ptr<utils::mpi_context_t<mpi3::communicator>> _mpi,
        HamiltonianOperations<MEM>&& hop_,
        nda::array<ComplexType,1>&& ci_,
        nda::array<devPsiT,2>&& orbs_,
        [[maybe_unused]] int targetNW = 1)
      : mpi(_mpi),
        walker_type(wlk),
        NMO{NMO_}, nup{nup_}, ndown{ndown_},
        HamOp(std::move(hop_)),
        ci(std::move(ci_)),
        OrbMats(std::move(orbs_))
  {
    utils::check(OrbMats.extent(0) == ci.size(), "Size mismatch");
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
  void vbias(WalkerSet<MEM>& wset, memory::array_view<MEM,ComplexType,2> v,
             double dt, int nt = 0);

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
  void Energy(WalkerSet<MEM> const& wset, memory::array_view<MEM,ComplexType,2> E,
              memory::array_view<MEM,ComplexType,1> Ov, int nt = 0);

  /*
   * Calculates the mixed density matrix for all walkers in the walker set. 
   * Options:
   *  - compact:   If true (default), returns compact form with Dim: [NEL*NMO], 
   *                 otherwise returns full form with Dim: [NMO*NMO]. 
   */ 
  void MixedDensityMatrix(WalkerSet<MEM> const& wset, memory::array_view<MEM,ComplexType,2> G,
                          bool compact = true);

  void MixedDensityMatrix(WalkerSet<MEM> const& wset, memory::array_view<MEM,ComplexType,2> G,
                          memory::array_view<MEM,ComplexType,1> Ov, bool compact = true);

  /*
   * Calculates the density matrix with respect to a given Reference
   * for all walkers in the walker set. 
   */
  template<class WlkSet, nda::MemoryVector RVec, nda::MemoryMatrix MatG, nda::MemoryVector TVec>
  void DensityMatrix(const WlkSet& wset, RVec&& Ref, MatG&& G, TVec&& Ov, 
                     bool compact = true, bool herm = true);

  /*
   * Calculates the overlaps of all walkers in the set. Returns values in arrays. 
   */
  void Log_Overlap(WalkerSet<MEM> const& wset, memory::array_view<MEM,ComplexType,1> Ov,
                   int nt = 0);

  /*
   * Calculates the overlaps of all walkers in the set. Updates values in wset. 
   */
  void Log_Overlap(WalkerSet<MEM>& wset);

  ComplexType getReferenceWeight(int i) const { return ci[i]; }

  int total_number_of_references() const { return OrbMats.extent(0); }

  int getNMO() const { return NMO; }

  /*
   * Returns the reference Slater Matrices needed for back propagation.  
   */ 
  void getReferences(memory::buffered_array<MEM,ComplexType,3>& Refs) const;

  void updateLogScale(auto scl_new, SpinTypes s)
  {
    utils::check(false, "updateLogScale is not implemented for ground state calculations");
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
  WALKER_TYPES walker_type;

  int NMO{};
  int nup{};
  int ndown{};

  HamiltonianOperations<MEM> HamOp;

  nda::array<ComplexType, 1> ci;

  // OrbMats[ndet][nspin](nel,NMO)
  nda::array<devPsiT,2> OrbMats;

/*
  void recompute_ci();
*/
};

/* 
 * Computes the density matrix for a given reference. 
 * G and Ov are expected to be in shared memory.
 * Simple round-robin is used. 
 */
template<MEMORY_SPACE MEM, class devPsiT>
template<class WlkSet, nda::MemoryVector RVec, nda::MemoryMatrix MatG, nda::MemoryVector TVec>
void NOMSD<MEM,devPsiT>::DensityMatrix(const WlkSet& wset, RVec&& Ref, MatG&& G, TVec&& Ov, 
                     bool compact, bool herm)
{
  // RVec is a vector of const_shared_array or CSRMatrix
  auto all = nda::range::all;
  memory::check_memory_space<MEM>(G,Ov);
  const int nw   = wset.size();
  const int nel = (walker_type==COLLINEAR ? nup+ndown : nup );
  const int nspin = (walker_type==COLLINEAR ? 2 : 1 );
  const int npol = (walker_type==NONCOLLINEAR ? 2 : 1 );
  utils::check(Ov.size() == nw, "Size mismatch");
  utils::check(Ref.size() == nspin,"Size mismatch");
  if(herm) {
    utils::check_shape(Ref(0), "Ref(0)", nup,npol*NMO);
    if(nspin>1) {
      utils::check_shape(Ref(1), "Ref(1)", ndown,npol*NMO);
    }
  } else {
    utils::check_shape(Ref(0), "Ref(0)", npol*NMO, nup);
    if(nspin>1) {
      utils::check_shape(Ref(1), "Ref(1)", npol*NMO, ndown);
    }
  }
  G() = ComplexType(0.0);
  Ov() = ComplexType(0.0);

  if(compact) {

    utils::check(G.shape() == std::array<long,2>{nw,nel*npol*NMO}, "Size mismatch");
    auto G3d = nda::reshape(G,std::array<long,3>{nw,nel,npol*NMO});
    det_ops::MixedDensityMatrix(Ref(0)(), wset.SlaterMatrices(Alpha), G3d(all,nda::range(nup),all), Ov, compact); 
    if(walker_type == CLOSED) {
      nda::tensor::scale(ComplexType(2.0),Ov);
    } else if (walker_type == COLLINEAR)  {
      det_ops::MixedDensityMatrix(Ref(1)(), wset.SlaterMatrices(Beta), G3d(all,nda::range(nup,nel),all), Ov, compact); 
    }

  } else {

    utils::check(G.shape() == std::array<long,2>{nw,nspin*npol*NMO*npol*NMO}, "Size mismatch");
    auto G3d = nda::reshape(G,std::array<long,3>{nw,nspin*npol*NMO,npol*NMO});
    det_ops::MixedDensityMatrix(Ref(0)(), wset.SlaterMatrices(Alpha), G3d(all,nda::range(npol*NMO),all), Ov, compact);
    if(walker_type == CLOSED) {
      nda::tensor::scale(ComplexType(2.0),Ov);
    } else if (walker_type == COLLINEAR)  {
      det_ops::MixedDensityMatrix(Ref(1)(), wset.SlaterMatrices(Beta), G3d(all,nda::range(npol*NMO,nspin*npol*NMO),all), Ov, compact);
    }

  }
}

/*
template<MEMORY_SPACE MEM, class devPsiT>
inline void NOMSD<MEM,devPsiT>::recompute_ci()
{
  if (walker_type == NONCOLLINEAR)
    APP_ABORT(" Error: NOMSD::recompute_ci not implemented with NONCOLLINEAR.");
  double LogOverlapFactor(0.0);
  int ndets = ci.size();
  if (ndets == 1)
  {
    ci[0] = ComplexType(1.0, 0.0);
    return;
  }

  if (TG.getNGroupsPerTG() > 1)
  {
    APP_ABORT(" Error: rediag not implemented with distributed wavefunction.");
  }
  else
  {
    size_t nt = (1 + dm_size(false) + NMO * nup);
    StaticSHMVector shmbuff(iextensions<1u>{nt}, ComplexType(0.0),
                            shm_buffer_manager.get_generator().template get_allocator<ComplexType>());

    boost::multi::array<ComplexType, 2, shared_allocator<ComplexType>> H({ndets, ndets}, TG.Node());
    boost::multi::array<ComplexType, 2, shared_allocator<ComplexType>> S({ndets, ndets}, TG.Node());
    boost::multi::array<ComplexType, 1> energy(iextensions<1u>{2});
    CMatrix Psia({0, 0});
    CMatrix Psib({0, 0});
    if (TG.TG_local().root())
    {
      // only TG_local().root() calculates G for now
      Psia = CMatrix({NMO, nup});
      if (walker_type == COLLINEAR)
        Psib = CMatrix({NMO, ndown});
    }
    using std::fill_n;
    fill_n(H.origin(), H.num_elements(), ComplexType(0.0));
    fill_n(S.origin(), S.num_elements(), ComplexType(0.0));
    fill_n(energy.origin(), energy.num_elements(), ComplexType(0.0));

    //ComplexType zero(0.0);
    auto Gsize = dm_size(false);
    int nr = Gsize, nc = 1;
    if (transposed_G_for_E_)
      std::swap(nr, nc);
    CVector_ref ov_(make_device_ptr(shmbuff.origin()), iextensions<1u>{1});
    CMatrix_ref G(ov_.origin() + 1, {nr, nc});
    StaticMatrix eloc2({1, 3},  ComplexType(0.0),
		       buffer_manager.get_generator().template get_allocator<ComplexType>());

    for (int jdet = 0, ji = 0; jdet < ndets; jdet++)
    {
      ComplexType cjdet = ci[jdet];
      for (int idet = jdet; idet < ndets; idet++, ji++)
      {
        if (ji % TG.getNumberOfTGs() == TG.getTGNumber())
        {
          ComplexType cidet = ci[idet];
          if (TG.TG_local().root())
          {
            CMatrix_ref Ga(G.origin(), {nup, NMO});
            ma::Matrix2MA('H', OrbMats[nspins * jdet], Psia);
            ov_[0] = SDetOp.MixedDensityMatrix(OrbMats[nspins * idet], Psia, Ga, LogOverlapFactor, true);
            if (walker_type == COLLINEAR)
            {
              CMatrix_ref Gb(Ga.origin() + Ga.num_elements(), {ndown, NMO});
              ma::Matrix2MA('H', OrbMats[nspins * jdet + 1], Psib);
              ov_[0] *= SDetOp.MixedDensityMatrix(OrbMats[nspins * idet + 1], Psib, Gb, LogOverlapFactor, true);
            }
            else if (walker_type == CLOSED)
            {
              ov_[0] *= ComplexType(ov_[0]);
            }
          }
          TG.TG_local().barrier();
          HamOp.energy(eloc2, G, idet, TG.TG_local().root());
          if (TG.TG_local().size() > 1)
            TG.TG_local().all_reduce_in_place_n(raw_pointer_cast(eloc2.origin()), 3, std::plus<>());
          if (TG.TG_local().root())
          {
            if (idet != jdet)
            {
              //ComplexType ovij = static_cast<ComplexType>(ov_[0]);
              H[idet][jdet] = ComplexType(ov_[0]) * (ComplexType(eloc2[0][0]) + ComplexType(eloc2[0][1]) + ComplexType(eloc2[0][2]));
              S[idet][jdet] = ComplexType(ov_[0]);
              S[idet][jdet] = ComplexType(ov_[0]);
              H[jdet][idet] = ma::conj(ComplexType(H[idet][jdet]));
              S[jdet][idet] = ma::conj(ComplexType(S[idet][jdet]));
              energy[0] += ma::conj(cidet) * cjdet * ComplexType(H[idet][jdet]) 
			+ ma::conj(cjdet) * cidet * ComplexType(H[jdet][idet]);
              energy[1] += ma::conj(cidet) * cjdet * ComplexType(S[idet][jdet]) 
			+ ma::conj(cjdet) * cidet * ComplexType(S[jdet][idet]);
            }
            else
            {
              H[idet][idet] =
                  ComplexType(real(ComplexType(ov_[0]) * (ComplexType(eloc2[0][0]) + ComplexType(eloc2[0][1]) + ComplexType(eloc2[0][2]))), 
                              0.0);
              S[idet][idet] = ComplexType(real(ComplexType(ov_[0])), 0.0);
              energy[0] += ma::conj(cidet) * cjdet * ComplexType(H[idet][jdet]);
              energy[1] += ma::conj(cidet) * cjdet * ComplexType(S[idet][jdet]);
            }
          }
          TG.TG_local().barrier();
        }
      }
    }
    TG.Global().barrier();
    if (TG.Node().root())
      TG.Cores().all_reduce_in_place_n(raw_pointer_cast(H.origin()), H.num_elements(), std::plus<>());
    if (TG.Node().root())
      TG.Cores().all_reduce_in_place_n(raw_pointer_cast(S.origin()), S.num_elements(), std::plus<>());
    TG.Global().all_reduce_in_place_n(energy.origin(), 2, std::plus<>());

    app_log(1," - Variational energy of trial wavefunction: {}",  energy[0] / energy[1]);
    app_log(1," - Diagonalizing CI matrix.");
    using RVector = boost::multi::array<RealType, 1>;
    #if defined(ENABLE_DEVICE)
    using CMatrix = boost::multi::array<ComplexType, 2>;
    #endif
    // Want a "unique" solution for all cores/nodes.
    if (TG.Global().rank() == 0)
    {
      boost::multi::array_ref<ComplexType, 2> H_(raw_pointer_cast(H.origin()), {ndets, ndets});
      boost::multi::array_ref<ComplexType, 2> S_(raw_pointer_cast(S.origin()), {ndets, ndets});
      std::pair<RVector, CMatrix> Sol = ma::genEigSelect<RVector, CMatrix>(H_, S_, 1);
      app_log(1," - Updating CI coefficients. ");
      app_log(1," - Recomputed coefficient of first determinant: {}", Sol.second[0][0]); 
      for (int idet = 0; idet < ndets; idet++)
      {
        ComplexType ci_ = Sol.second[0][idet];
        // Do we want this much output?
        app_log(1, "{} old: {}, new: {} ",idet,ci[idet],ci_);
        ci[idet] = ci_;
      }
      app_log(1," - Recomputed variational energy of trial wavefunction: {}",Sol.first[0]);
    }
    if (TG.Global().size() > 1)
      TG.Global().broadcast_n(raw_pointer_cast(ci.data()), ci.size(), 0);
  }
}
*/

} // namespace afqmc

} // namespace sfqmc

