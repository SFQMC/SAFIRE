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

namespace sfqmc
{
namespace afqmc
{
namespace full_g
{

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

  int const nwalk      = int(Gfull.extent(0));
  int const NMO        = int(Lankf.extent(1));
  ComplexType const scl(2.0, 0.0);

  utils::check(Gfull.extent(1) == long(NMO) * NMO, "full_g::energy_closed: G shape mismatch");
  utils::check(Lankf.extent(0) == long(NMO) * local_nCV, "full_g::energy_closed: Lankf shape mismatch");
  utils::check(hijf.extent(0) == long(NMO) * NMO, "full_g::energy_closed: hijf shape mismatch");

  E() = ComplexType(0.0);
  if (addH1)
    E(all, 0) = E0;

  if (addH1)
  {
    // E[w][0] += scl * sum_ik h_ik G[w][ik]
    // hijf is stored flat [NMO*NMO] and Gfull flat [nwalk][NMO*NMO]; tensor::contract needs the
    // index ranks to match the labels, so view them as h[i][k] and G[w][i][k].
    auto hij2 = nda::reshape(hijf, std::array<long, 2>{NMO, NMO});
    auto G3   = nda::reshape(Gfull, std::array<long, 3>{nwalk, NMO, NMO});
    nda::tensor::contract(scl, hij2, "ik", G3, "wik", ComplexType(1.0), E(all, 0), "w");
  }

  if (not addEXX)
    return;

  // Cholesky (Lankf) is replicated, so each rank computes the COMPLETE EXX/EJ for ALL walkers (no
  // walker round-robin, no (i,nc) FairDivide, no all_reduce). mpi is unused here.
  (void)mpi;

  memory::buffered_array<MEM, ComplexType, 2> GF(nwalk * NMO, NMO);
  for (int n = 0; n < nwalk; ++n)
  {
    auto Gn = Gfull(n, all);
    for (int i = 0; i < NMO; ++i)
      for (int k = 0; k < NMO; ++k)
        GF(n * NMO + i, k) = Gn(i * NMO + k);
  }

  // Twban[(n,i)][(i',nc)] = sum_k GF[(n,i)][k] * Lankf[(i',nc)][k], over the FULL (i',nc) range.
  memory::buffered_array<MEM, ComplexType, 2> Twban(nwalk * NMO, NMO * local_nCV);
  nda::blas::gemm(ComplexType(1.0), GF, nda::transpose(Lankf), ComplexType(0.0), Twban);

  auto T4D = nda::reshape(Twban, std::array<long, 4>{nwalk, NMO, NMO, local_nCV});

  for (int n = 0; n < nwalk; ++n)
  {
    ComplexType exx(0.0);
    for (int a = 0; a < NMO; ++a)
      for (int b = 0; b < NMO; ++b)
        // non-conjugating dot: EXX = sum_{ij,nc} T[i][j][nc] T[j][i][nc] (matches energy_impl).
        exx += static_cast<ComplexType>(nda::blas::dot(T4D(n, a, b, all), T4D(n, b, a, all)));
    E(n, 1) -= ComplexType(0.5) * scl * exx;
  }

  if (addEJ)
  {
    memory::buffered_array<MEM, ComplexType, 2> Kl(nwalk, local_nCV);
    Kl() = ComplexType(0.0);
    for (int n = 0; n < nwalk; ++n)
      for (int a = 0; a < NMO; ++a)
        Kl(n, all) += T4D(n, a, a, all);
    for (int n = 0; n < nwalk; ++n)
      E(n, 2) += ComplexType(0.5) * scl * scl *
                 static_cast<ComplexType>(nda::blas::dot(Kl(n, all), Kl(n, all)));
  }
  else
  {
    utils::check(false, "full_g::energy_closed: addEXX without addEJ not implemented");
  }
}

} // namespace full_g
} // namespace afqmc
} // namespace sfqmc
