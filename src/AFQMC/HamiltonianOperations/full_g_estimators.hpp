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
#include "utilities/FairDivide.hpp"
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

// Phase 3b (StochasticWfn): un-rotated full-G local-energy contraction for CLOSED (RHF) trials.
// G layout: [nwalk][NMO*NMO]. E is partial per MPI rank; the caller all-reduces.
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

  memory::buffered_array<MEM, ComplexType, 2> GF(nwalk * NMO, NMO);
  for (int n = 0; n < nwalk; ++n)
  {
    if (n % mpi->comm.size() != mpi->comm.rank())
      continue;
    auto Gn = Gfull(n, all);
    for (int i = 0; i < NMO; ++i)
      for (int k = 0; k < NMO; ++k)
        GF(n * NMO + i, k) = Gn(i * NMO + k);
  }
  mpi->comm.barrier();

  memory::buffered_array<MEM, ComplexType, 2> Twban(nwalk * NMO, NMO * local_nCV);
  long i0, iN;
  std::tie(i0, iN) = FairDivideBoundary(long(mpi->comm.rank()), long(NMO) * local_nCV, long(mpi->comm.size()));
  if (iN > i0)
  {
    // Lankf is [NMO*local_nCV][NMO] indexed at row (i*local_nCV + nc); the (i,nc) combined index
    // (== Twban's column index) is what FairDivide partitions, so slice Lankf's ROWS, not its columns.
    auto Lslice = Lankf(range(i0, iN), all);
    auto Tslice = Twban(all, range(i0, iN));
    nda::blas::gemm(ComplexType(1.0), GF, nda::transpose(Lslice), ComplexType(0.0), Tslice);
  }
  mpi->comm.barrier();

  auto T4D = nda::reshape(Twban, std::array<long, 4>{nwalk, NMO, NMO, local_nCV});

  for (int n = 0; n < nwalk; ++n)
  {
    if (n % mpi->comm.size() != mpi->comm.rank())
      continue;
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
    {
      if (n % mpi->comm.size() != mpi->comm.rank())
        continue;
      for (int a = 0; a < NMO; ++a)
        Kl(n, all) += T4D(n, a, a, all);
    }
    mpi->comm.barrier();
    for (int n = 0; n < nwalk; ++n)
    {
      if (n % mpi->comm.size() != mpi->comm.rank())
        continue;
      E(n, 2) += ComplexType(0.5) * scl * scl *
                 static_cast<ComplexType>(nda::blas::dot(Kl(n, all), Kl(n, all)));
    }
  }
  else
  {
    utils::check(false, "full_g::energy_closed: addEXX without addEJ not implemented");
  }

  mpi->comm.barrier();
}

} // namespace full_g
} // namespace afqmc
} // namespace sfqmc
