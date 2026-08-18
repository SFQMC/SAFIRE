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

#include "AFQMC/config.h"
#include "utilities/mpi_context.h"
#include "numerics/operations/product.hpp"
#include "numerics/nda_functions.hpp"
#include "nda/blas.hpp"
#include "nda/tensor.hpp"
#if defined(ENABLE_DEVICE)
#include "numerics/device_kernels/kernels.h" // kernels::device::{accumulate,diag_trace_accumulate} (device only)
#endif

namespace sfqmc
{
namespace afqmc
{
/**
 * @brief Local-energy kernels contracting a FULL (un-rotated) density matrix against a bare Cholesky.
 *
 * @details The compact path half-rotates the Cholesky against the trial determinant, which a
 * stochastic trial has no single form of: its density matrix is an inner-ensemble reduction and exists
 * only in the full layout. Reached via Real3IndexFactorization::energy_fullG.
 */
namespace full_g
{

/**
 * @brief Accumulate ONE spin's contribution to the un-rotated full-G local energy.
 *
 * @details Adds E1 and EXX for a single spin into E, and the RAW Coulomb vector
 * K_n = sum_a (sum_k G[a][k] L[a][k][n]) into Kl. The caller owns zeroing E, adding E0 and the EJ
 * finalization 0.5*scl^2*|Kl|^2, which is what lets the two COLLINEAR spin calls share one Kl so the
 * Coulomb term contracts the TOTAL (alpha+beta) density.
 *
 * @param E [nwalk, 3] energies (one-body, exchange, Coulomb); accumulated into, not zeroed
 * @param G3 [nwalk][NMO][NMO] view of G_sigma[i][k]. MAY BE STRIDED -- e.g. a COLLINEAR spin block of
 *        a [nwalk][2*NMO*NMO] buffer -- since it is only element-accessed, never reshaped; accepting a
 *        strided view avoids a second per-spin block copy in energy_collinear. The EXX gemm still
 *        repacks it contiguously.
 * @param Lankf [NMO*local_nCV][NMO] bare, spin-independent Cholesky, Lankf(i*local_nCV + n, k) = L(i,k,n)
 * @param hijf [NMO*NMO] bare, spin-independent one-body h_ik
 * @param local_nCV Cholesky vectors held on this rank
 * @param scl spin prefactor: 2 for CLOSED (doubling the single alpha spin), 1 for COLLINEAR
 * @param Kl accumulated Coulomb vector, shared across the spin calls of one walker set
 * @param addH1 include the one-body term
 * @param addEJ accumulate the Coulomb vector; the finalization is the caller's
 * @param addEXX include the exchange term
 */
template<MEMORY_SPACE MEM, class MatE, class MatG, class MatLan, class VecHij, class MatK>
void accumulate_spin_full_g(MatE&& E,
                            MatG const& G3,
                            MatLan const& Lankf,
                            VecHij const& hijf,
                            int local_nCV,
                            ComplexType scl,
                            MatK&& Kl,
                            bool addH1,
                            bool addEJ,
                            bool addEXX)
{
  using nda::range;
  auto all = range::all;
  memory::check_memory_space<MEM>(E, G3, Lankf, hijf);

  int const nwalk = int(G3.extent(0));
  int const NMO   = int(Lankf.extent(1));

  utils::check(G3.extent(1) == NMO && G3.extent(2) == NMO, "full_g::accumulate_spin_full_g: G shape mismatch");
  utils::check(Lankf.extent(0) == long(NMO) * local_nCV, "full_g::accumulate_spin_full_g: Lankf shape mismatch");
  utils::check(hijf.extent(0) == long(NMO) * NMO, "full_g::accumulate_spin_full_g: hijf shape mismatch");

  if (addH1)
  {
    // E[w][0] += scl * sum_ik h_ik G[w][ik]
    auto hij2 = nda::reshape(hijf, std::array<long, 2>{NMO, NMO});
    nda::tensor::contract(scl, hij2, "ik", G3, "wik", ComplexType(1.0), E(all, 0), "w");
  }

  if (not addEXX)
    return;

  // GF[(n,i)][k] = G[n][i][k] -- repack into contiguous [nwalk*NMO][NMO] for the gemm (handles a
  // strided G3, e.g. a collinear spin block).
  memory::buffered_array<MEM, ComplexType, 2> GF(nwalk * NMO, NMO);
  if constexpr (MEM == HOST_MEMORY)
  {
    for (int n = 0; n < nwalk; ++n)
      for (int i = 0; i < NMO; ++i)
        for (int k = 0; k < NMO; ++k)
          GF(n * NMO + i, k) = G3(n, i, k);
  }
#if defined(ENABLE_DEVICE)
  else
  {
    // GF viewed as [nwalk, NMO, NMO] is exactly G3; the nda copy handles a strided G3 (collinear block).
    auto GF3d = nda::reshape(GF, std::array<long, 3>{nwalk, NMO, NMO});
    GF3d() = G3();
  }
#else
  else
  {
    static_assert(MEM == HOST_MEMORY, "Device memory requires ENABLE_DEVICE");
  }
#endif

  // Twban[(n,i)][(i',nc)] = sum_k GF[(n,i)][k] * Lankf[(i',nc)][k], over the FULL (i',nc) range.
  memory::buffered_array<MEM, ComplexType, 2> Twban(nwalk * NMO, NMO * local_nCV);
  nda::blas::gemm(ComplexType(1.0), GF, nda::transpose(Lankf), ComplexType(0.0), Twban);

  auto T4D = nda::reshape(Twban, std::array<long, 4>{nwalk, NMO, NMO, local_nCV});

  // EXX[w] = sum_{a,b,nc} T4D[w,a,b,nc] * T4D[w,b,a,nc] (non-conjugating); E[w,1] -= 0.5*scl*EXX[w].
  if constexpr (MEM == HOST_MEMORY)
  {
    for (int n = 0; n < nwalk; ++n)
    {
      ComplexType exx(0.0);
      for (int a = 0; a < NMO; ++a)
        for (int b = 0; b < NMO; ++b)
          // non-conjugating dot: EXX = sum_{ij,nc} T[i][j][nc] T[j][i][nc] (matches energy_impl).
          exx += static_cast<ComplexType>(nda::blas::dot(T4D(n, a, b, all), T4D(n, b, a, all)));
      E(n, 1) -= ComplexType(0.5) * scl * exx;
    }
  }
#if defined(ENABLE_DEVICE)
  else
  {
    // CuTENSOR contraction with the (a<->b) index swap; the two operands are the same T4D relabeled.
    memory::buffered_array<MEM, ComplexType, 1> exx(nwalk);
    nda::tensor::contract(ComplexType(1.0), T4D, "wabc", T4D, "wbac", ComplexType(0.0), exx, "w");
    kernels::device::accumulate(ComplexType(-0.5) * scl, exx, E(all, 1));
  }
#else
  else
  {
    static_assert(MEM == HOST_MEMORY, "Device memory requires ENABLE_DEVICE");
  }
#endif

  if (addEJ)
  {
    // RAW per-spin Coulomb vector (no scl): Kl[n][nc] += sum_a T4D[n][a][a][nc]. For COLLINEAR both
    // spins accumulate into the same Kl, so the caller's 0.5*scl^2*|Kl|^2 acts on the total density.
    if constexpr (MEM == HOST_MEMORY)
    {
      for (int n = 0; n < nwalk; ++n)
        for (int a = 0; a < NMO; ++a)
          Kl(n, all) += T4D(n, a, a, all);
    }
#if defined(ENABLE_DEVICE)
    else
    {
      kernels::device::diag_trace_accumulate(T4D, Kl);
    }
#else
    else
    {
      static_assert(MEM == HOST_MEMORY, "Device memory requires ENABLE_DEVICE");
    }
#endif
  }
  else
  {
    utils::check(false, "full_g::accumulate_spin_full_g: addEXX without addEJ not implemented");
  }
}

/**
 * @brief Un-rotated full-G local energy for CLOSED (RHF) trials.
 *
 * @details Computes the COMPLETE E for ALL walkers on EVERY rank, redundantly; the caller does NOT
 * reduce. That matches the compact energy_impl path, which iterates the full nCV with no MPI reduction
 * -- the reason NOMSD::Energy needs no external all_reduce. Distributing the walker loop or the (i,nc)
 * index without a reduction is correct only at -np 1: beyond that EXX/EJ come out walker-incomplete
 * and CV-partial while E1 is already complete, so no caller-side all_reduce can repair it.
 *
 * @param mpi MPI context; unused, the kernel is replicated rather than distributed
 * @param E [nwalk, 3] energies (one-body, exchange, Coulomb)
 * @param Gfull [nwalk][NMO*NMO] full density matrix, row-major G[i][k]
 * @param Lankf [NMO*local_nCV][NMO] bare Cholesky
 * @param hijf [NMO*NMO] bare one-body
 * @param local_nCV Cholesky vectors held on this rank
 * @param E0 zero of energy, added once per walker
 * @param addH1 include the one-body term
 * @param addEJ include the Coulomb term
 * @param addEXX include the exchange term
 */
template<MEMORY_SPACE MEM, class MatE, class MatG, class MatLan, class VecHij>
void energy_closed(std::shared_ptr<utils::mpi_context_t<mpi3::communicator>> const& mpi,
                   MatE&& E,
                   MatG const& Gfull,
                   MatLan const& Lankf,
                   VecHij const& hijf,
                   int local_nCV,
                   ComplexType E0,
                   bool addH1  = true,
                   bool addEJ  = true,
                   bool addEXX = true)
{
  using nda::range;
  auto all = range::all;
  memory::check_memory_space<MEM>(E, Gfull, Lankf, hijf);

  int const NMO = int(Lankf.extent(1));
  int const nwalk = int(Gfull.extent(0));
  ComplexType const scl(2.0, 0.0); // CLOSED: single alpha spin, doubled.

  utils::check(Gfull.extent(1) == long(NMO) * NMO, "full_g::energy_closed: G shape mismatch");

  // Cholesky (Lankf) is replicated, so each rank computes the COMPLETE E for ALL walkers (no walker
  // round-robin, no (i,nc) FairDivide, no all_reduce). mpi is unused here.
  (void)mpi;

  E() = ComplexType(0.0);
  if (addH1)
    E(all, 0) = E0;

  memory::buffered_array<MEM, ComplexType, 2> Kl((addEJ ? nwalk : 0), (addEJ ? local_nCV : 0));
  if (addEJ)
    Kl() = ComplexType(0.0);

  auto G3 = nda::reshape(Gfull, std::array<long, 3>{nwalk, NMO, NMO});
  accumulate_spin_full_g<MEM>(E, G3, Lankf, hijf, local_nCV, scl, Kl(), addH1, addEJ, addEXX);

  if (addEXX and addEJ)
  {
    // EJ[w] = sum_nc Kl[w,nc]*Kl[w,nc] (non-conjugating); E[w,2] += 0.5*scl^2*EJ[w].
    if constexpr (MEM == HOST_MEMORY)
      for (int n = 0; n < nwalk; ++n)
        E(n, 2) += ComplexType(0.5) * scl * scl *
                   static_cast<ComplexType>(nda::blas::dot(Kl(n, all), Kl(n, all)));
#if defined(ENABLE_DEVICE)
    else
    {
      memory::buffered_array<MEM, ComplexType, 1> ej(nwalk);
      nda::tensor::contract(ComplexType(1.0), Kl(), "wc", Kl(), "wc", ComplexType(0.0), ej, "w");
      kernels::device::accumulate(ComplexType(0.5) * scl * scl, ej, E(all, 2));
    }
#else
    else
    {
      static_assert(MEM == HOST_MEMORY, "Device memory requires ENABLE_DEVICE");
    }
#endif
  }
}

// StochasticWfn: un-rotated full-G local-energy contraction for COLLINEAR (UHF) trials (inner_nsteps > 0).
/**
 * @brief Un-rotated full-G local energy for COLLINEAR (UHF) trials.
 *
 * @details Mirrors the compact COLLINEAR energy: per-spin E1 and EXX with no closed-shell doubling,
 * and a SINGLE EJ contracted on the TOTAL (alpha+beta) density. Both spins reuse the same bare
 * Cholesky and one-body, so a spin-dependent Hamiltonian would score beta with the alpha integrals;
 * Real3IndexFactorization::energy_fullG rejects that upstream. Replicated per rank, no all_reduce.
 *
 * @param mpi MPI context; unused, the kernel is replicated rather than distributed
 * @param E [nwalk, 3] energies (one-body, exchange, Coulomb)
 * @param Gfull [nwalk][2*NMO*NMO]: alpha block then beta block, each row-major G_sigma[i][k] -- the
 *        layout the inner-ensemble cross reduction builds for COLLINEAR
 * @param Lankf [NMO*local_nCV][NMO] bare, spin-independent Cholesky
 * @param hijf [NMO*NMO] bare, spin-independent one-body
 * @param local_nCV Cholesky vectors held on this rank
 * @param E0 zero of energy, added once per walker
 * @param addH1 include the one-body term
 * @param addEJ include the Coulomb term
 * @param addEXX include the exchange term
 */
template<MEMORY_SPACE MEM, class MatE, class MatG, class MatLan, class VecHij>
void energy_collinear(std::shared_ptr<utils::mpi_context_t<mpi3::communicator>> const& mpi,
                      MatE&& E,
                      MatG const& Gfull,
                      MatLan const& Lankf,
                      VecHij const& hijf,
                      int local_nCV,
                      ComplexType E0,
                      bool addH1  = true,
                      bool addEJ  = true,
                      bool addEXX = true)
{
  using nda::range;
  auto all = range::all;
  memory::check_memory_space<MEM>(E, Gfull, Lankf, hijf);

  int const NMO = int(Lankf.extent(1));
  int const nwalk = int(Gfull.extent(0));
  ComplexType const scl(1.0, 0.0); // COLLINEAR: per-spin, no closed doubling.

  utils::check(Gfull.extent(1) == 2L * NMO * NMO,
               "full_g::energy_collinear: G shape mismatch (expected [nwalk][2*NMO*NMO])");

  (void)mpi;

  E() = ComplexType(0.0);
  if (addH1)
    E(all, 0) = E0;

  memory::buffered_array<MEM, ComplexType, 2> Kl((addEJ ? nwalk : 0), (addEJ ? local_nCV : 0));
  if (addEJ)
    Kl() = ComplexType(0.0);

  // View the alpha block [0, NMO*NMO) and beta block [NMO*NMO, 2*NMO*NMO) of each contiguous Gfull row
  // as STRIDED [nwalk][NMO][NMO] arrays (row stride 2*NMO*NMO) and pass them straight to the per-spin
  // kernel -- no per-block copy (the kernel element-accesses G3 and repacks into its own gemm buffer).
  // E0 is added once above; both spin calls accumulate one-body/exchange into E and Coulomb into shared Kl.
  utils::check(Gfull.is_contiguous(), "full_g::energy_collinear: Gfull must be contiguous");
  std::array<long, 3> const shp{nwalk, NMO, NMO};
  std::array<long, 3> const strd{2L * NMO * NMO, long(NMO), 1L};
  nda::idx_map<3, 0, nda::C_stride_order<3>, nda::layout_prop_e::none> const idxm(shp, strd);

  auto Ga3 = memory::array_view<MEM, const ComplexType, 3>(idxm, Gfull.data());
  accumulate_spin_full_g<MEM>(E, Ga3, Lankf, hijf, local_nCV, scl, Kl(), addH1, addEJ, addEXX);

  auto Gb3 = memory::array_view<MEM, const ComplexType, 3>(idxm, Gfull.data() + long(NMO) * NMO);
  accumulate_spin_full_g<MEM>(E, Gb3, Lankf, hijf, local_nCV, scl, Kl(), addH1, addEJ, addEXX);

  if (addEXX and addEJ)
  {
    // EJ[w] = sum_nc Kl[w,nc]*Kl[w,nc] (non-conjugating); E[w,2] += 0.5*scl^2*EJ[w].
    if constexpr (MEM == HOST_MEMORY)
      for (int n = 0; n < nwalk; ++n)
        E(n, 2) += ComplexType(0.5) * scl * scl *
                   static_cast<ComplexType>(nda::blas::dot(Kl(n, all), Kl(n, all)));
#if defined(ENABLE_DEVICE)
    else
    {
      memory::buffered_array<MEM, ComplexType, 1> ej(nwalk);
      nda::tensor::contract(ComplexType(1.0), Kl(), "wc", Kl(), "wc", ComplexType(0.0), ej, "w");
      kernels::device::accumulate(ComplexType(0.5) * scl * scl, ej, E(all, 2));
    }
#else
    else
    {
      static_assert(MEM == HOST_MEMORY, "Device memory requires ENABLE_DEVICE");
    }
#endif
  }
}

} // namespace full_g
} // namespace afqmc
} // namespace sfqmc
