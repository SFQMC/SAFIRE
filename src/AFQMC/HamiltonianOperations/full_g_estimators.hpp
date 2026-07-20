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
namespace full_g
{

// StochasticWfn: per-spin contribution of the un-rotated full-G local energy (inner_nsteps > 0).
// Accumulates the one-body (E1) and exchange (EXX) energies for ONE spin's full density matrix into
// E, and accumulates the RAW per-spin Coulomb vector K_n = sum_a (sum_k G[a][k] L[a][k][n]) into Kl.
// The caller owns zeroing E, adding E0, and the EJ finalization 0.5*scl^2*|Kl|^2 -- so for COLLINEAR
// the two spin calls share one Kl and the EJ finalization contracts the TOTAL (alpha+beta) density.
//   scl : 2 (CLOSED, doubles the single alpha spin) or 1 (COLLINEAR, per spin).
//   G3  : [nwalk][NMO][NMO] view of G_sigma[i][k]. MAY BE STRIDED -- e.g. a COLLINEAR spin block of a
//         [nwalk][2*NMO*NMO] buffer -- since it is only element-accessed (contract + the GF repack),
//         never reshaped. The EXX gemm still needs the contiguous [nwalk*NMO][NMO] GF, so one repack is
//         unavoidable; this avoids a *separate* per-spin block copy in energy_collinear.
//   Lankf : [NMO*local_nCV][NMO], Lankf(i*local_nCV + n, k) = L(i,k,n) (bare, spin-independent Cholesky).
//   hijf  : [NMO*NMO], bare (spin-independent) one-body h_ik.
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

// StochasticWfn: un-rotated full-G local-energy contraction for CLOSED (RHF) trials (inner_nsteps > 0).
// G layout: [nwalk][NMO*NMO]. The Cholesky (Lankf) is REPLICATED on every rank -- exactly as the compact
// Real3IndexFactorization::energy_impl path is (it iterates the full nCV with no MPI reduction, which is
// why NOMSD::Energy needs no external all_reduce). So this kernel computes the COMPLETE E for ALL walkers
// on every rank (redundant across ranks) and the caller does NOT reduce. (Earlier it distributed BOTH the
// walker loop [n % comm.size()] AND the (i,nc) index [FairDivide] across the comm with no reduction --
// correct only at -np 1; at -np > 1 that left EXX/EJ walker-incomplete and CV-partial -> garbage, while
// the ungated E1 was already complete, so no single caller all_reduce could fix it.)
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
// G layout: [nwalk][2*NMO*NMO] = alpha block [0, NMO*NMO) followed by beta block [NMO*NMO, 2*NMO*NMO),
// each row-major G_sigma[i][k] (the layout reduce_inner_cross_dm builds for COLLINEAR). Mirrors the
// compact COLLINEAR energy: per-spin E1 and EXX with scl=1 (no closed doubling), and a single EJ on the
// TOTAL (alpha+beta) density. Both spins reuse the bare spin-independent Cholesky/one-body from
// ensure_full_cholesky() (Likn(0)); spin-dependent integrals (Likn.extent(0)>1) are rejected upstream.
// Replicated per rank, no all_reduce.
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
