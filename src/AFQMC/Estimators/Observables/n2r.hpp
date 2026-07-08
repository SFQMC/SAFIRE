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

#ifndef SFQMC_AFQMC_N2R_HPP
#define SFQMC_AFQMC_N2R_HPP

#include "AFQMC/config.h"
#include <vector>
#include <string>
#include <iostream>

#include "hdf/hdf_multi.h"
#include "hdf/hdf_archive.h"

#include "AFQMC/Walkers/WalkerSet.hpp"
#include "Numerics/detail/utilities.hpp"
#include "Numerics/ma_operations.hpp"
#include "Numerics/batched_operations.hpp"

namespace sfqmc
{
namespace afqmc
{
/* 
 * Observable class that calculates the walker averaged on-top pair density 
 * Alloc defines the allocator type used to store the orbital and temporary tensors.
 * In a device compilation, this would need out-of-card gemm available.
 */
template<class Alloc>
class n2r : public AFQMCInfo
{
  // allocators
  using Allocator = device_allocator<ComplexType>;

  // type defs
  using pointer       = typename std::allocator_traits<Allocator>::pointer;
  using const_pointer = typename std::allocator_traits<Allocator>::const_pointer;

  // allocators
  using aux_Allocator = Alloc;

  using aux_pointer       = typename std::allocator_traits<aux_Allocator>::pointer;
  using const_aux_pointer = typename std::allocator_traits<aux_Allocator>::const_pointer;

  using auxCVector     = boost::multi::array<ComplexType, 1, aux_Allocator>;
  using auxCMatrix     = boost::multi::array<ComplexType, 2, aux_Allocator>;
  using auxCVector_ref = boost::multi::array_ref<ComplexType, 1, aux_pointer>;
  using auxCMatrix_ref = boost::multi::array_ref<ComplexType, 2, aux_pointer>;

  using CVector_ref    = boost::multi::array_ref<ComplexType, 1, pointer>;
  using CMatrix_ref    = boost::multi::array_ref<ComplexType, 2, pointer>;
  using CVector        = boost::multi::array<ComplexType, 1, Allocator>;
  using CMatrix        = boost::multi::array<ComplexType, 2, Allocator>;
  using stdCVector     = boost::multi::array<ComplexType, 1>;
  using stdIVector     = boost::multi::array<int, 1>;
  using stdCVector_ref = boost::multi::array_ref<ComplexType, 1>;
  using stdCMatrix_ref = boost::multi::array_ref<ComplexType, 2>;
  using mpi3CVector    = boost::multi::array<ComplexType, 1, shared_allocator<ComplexType>>;
  using mpi3CMatrix    = boost::multi::array<ComplexType, 2, shared_allocator<ComplexType>>;
  using mpi3CTensor    = boost::multi::array<ComplexType, 3, shared_allocator<ComplexType>>;
  using mpi3C4Tensor   = boost::multi::array<ComplexType, 4, shared_allocator<ComplexType>>;

public:
  n2r(afqmc::TaskGroup_& tg_,
      AFQMCInfo& info,
      ptree pt,
      WALKER_TYPES wlk,
      bool host_mem,
      aux_Allocator alloc_,
      aux_Allocator orb_alloc_,
      int nave_ = 1,
      int bsize = 1)
      : AFQMCInfo(info),
        aux_alloc(alloc_),
        block_size(bsize),
        nave(nave_),
        counter(0),
        TG(tg_),
        walker_type(wlk),
        dm_size(0),
        writer(false),
        use_host_memory(host_mem),
        hdf_walker_output(""),
        Orbitals({0, 0}, orb_alloc_),
        DMAverage({0, 0}, shared_allocator<ComplexType>{TG.TG_local()}),
        Buff(iextensions<1u>{0}, alloc_),
        Buff2(iextensions<1u>{0})
  {
    app_log(1,"  --  Adding on-top pair density (N2R) estimator. -- \n ");
    std::string orb_file = pt.get<std::string>("orbitals");
    if (not file_exists(orb_file))
    {
      app_error(" Error: orbitals file does not exist: {}", orb_file);
      APP_ABORT("");
    }

    stdIVector norbs(iextensions<1u>{0});
    stdIVector grid_dim(iextensions<1u>{3});
    dm_size = 0;

    // read orbitals
    hdf_archive dump;
    if (TG.Node().root())
    {
      if (!dump.open(orb_file, H5F_ACC_RDONLY))
        APP_ABORT(" Error opening orbitals file for n2r estimator. ");
      if (dump.push("OrbsR", false)<0)
        APP_ABORT(" Error in n2r: Group OrbsR not found.");
      if (!dump.readEntry(grid_dim, "grid_dim"))
        APP_ABORT(" Error in n2r: Problems reading grid_dim. ");
      RUNTIME_CHECK(grid_dim.size() == 3, "");
      dm_size = grid_dim[0] * grid_dim[1] * grid_dim[2];
      if (dm_size < 1)
        APP_ABORT(" Error in n2r: prod(grid_dim) < 1. ");
      TG.Node().broadcast_n(&dm_size, 1, 0);
      if (!dump.readEntry(norbs, "number_of_orbitals"))
        APP_ABORT(" Error in n2r: Problems reading number_of_orbitals. ");
      int M = 0, nk = norbs.size();
      for (int i = 0; i < nk; i++)
        M += norbs[i];
      if (M != NMO) {
        app_error(" Error in n2r: Inconsistent number of orbitals in file: M={}, expected={}",
		      M,NMO);
        APP_ABORT("");
      }
      Orbitals = auxCMatrix({NMO, dm_size}, orb_alloc_);
      for (int k = 0, kn = 0; k < nk; k++)
      {
        for (int i = 0; i < norbs[k]; i++, kn++)
        {
          stdCVector orb(iextensions<1u>{dm_size});
          if (!dump.readEntry(orb, "kp" + std::to_string(k) + "_b" + std::to_string(i)))
          {
            app_error(" Error in n2r: Problems reading orbital: {} {}",k,i);
            APP_ABORT("");
          }
          using std::copy_n;
          copy_n(orb.origin(), dm_size, ma::pointer_dispatch(Orbitals[kn].origin()));
        }
      }
      dump.pop();
      dump.close();

      app_log(1," Number of grid points: {}", dm_size);
    }
    else
    {
      TG.Node().broadcast_n(&dm_size, 1, 0);
      Orbitals = auxCMatrix({NMO, dm_size}, orb_alloc_);
    }
    TG.Node().barrier();

    using std::fill_n;
    writer = (TG.Global().rank() == 0);

    DMAverage = mpi3CMatrix({nave, dm_size}, shared_allocator<ComplexType>{TG.TG_local()});
    fill_n(DMAverage.origin(), DMAverage.num_elements(), ComplexType(0.0, 0.0));
  }

/*******   Interface for sum over references, e.g. NOMSD ********/
  template<class MatG, class MatG_host, class HostCVec1>
  void accumulate(int iav, MatG&& G, [[maybe_unused]] MatG_host&& G_host, 
                  HostCVec1&& Xw, [[maybe_unused]] bool impsamp)
  {
    static_assert(std::decay<MatG>::type::dimensionality == 4, "Wrong dimensionality");
    static_assert(std::decay<MatG_host>::type::dimensionality == 4, "Wrong dimensionality");
    using std::copy_n;
    using std::fill_n;
    // assumes G[nwalk][spin][M][M]
    RUNTIME_CHECK(G.size(0) == Xw.size(0), "");

    int nw(G.size(0));
    int nsp = ( walker_type == COLLINEAR ? 2 : 1 );

    int i0, iN;
    std::tie(i0, iN) = FairDivideBoundary(TG.TG_local().rank(), dm_size, TG.TG_local().size());

    // MAM: Uses out-of-card GEMM on device!!!
    // T1[i][r] = sum_j G[spin][i][j] * Psi(j,r)
    // G[spin][r] = sum_ij conj( Psi(i,r) ) * G[spin][i][j] * Psi(j,r) = sum_i conj( Psi(i,r) ) T1[i][r]
    int N = dm_size * NMO + dm_size;
    set_buffer(N);
    auxCMatrix_ref T(Buff.origin(), {NMO, dm_size});
    auxCVector_ref Gr(Buff.origin() + T.num_elements(), iextensions<1u>{dm_size});

    int N2 = nsp * (iN - i0);
    set_buffer2(N2);
    stdCMatrix_ref Gr_(Buff2.origin(), {nsp, (iN - i0)});

    // change batched_dot to a ** interface to make it more general and useful

    for (int iw = 0; iw < nw; iw++)
    {
      auto&& Gu = G[iw][0];
      auto&& Orb0N = Orbitals(Orbitals.extension(0), {i0, iN});
      auto&& T0N = T(T.extension(0), {i0, iN});
      ma::product(Gu, Orb0N, T0N);
      ma::dot('H', 'T', Orb0N, T0N, Gr.range({i0,iN}));	

      using std::copy_n;
      copy_n(ma::pointer_dispatch(Gr.origin()) + i0, (iN - i0), Gr_[0].origin());

      if (walker_type == CLOSED)
      {
        auto Gur = Gr_[0].origin();
        auto DM = raw_pointer_cast(DMAverage[iav].origin()) + i0;
        auto X_ = 2.0 * Xw[iw];
        for (int ir = i0; ir < iN; ir++, Gur++, DM++)
          (*DM) += X_ * (*Gur) * (*Gur);
      }
      else if (walker_type == COLLINEAR)
      {
        auto&& Gd = G[iw][1];
        ma::product(Gd, Orb0N, T0N);
        ma::dot('H', 'T', Orb0N, T0N, Gr.range({i0,iN}));	
        using std::copy_n;
        copy_n(ma::pointer_dispatch(Gr.origin()) + i0, (iN - i0), Gr_[1].origin());

        {
          auto Gur = Gr_[0].origin();
          auto Gdr = Gr_[1].origin();
          auto DM = raw_pointer_cast(DMAverage[iav].origin()) + i0;
          auto X_ = 2.0 * Xw[iw];
          for (int ir = i0; ir < iN; ir++, Gur++, DM++)
            (*DM) += X_ * (*Gur) * (*Gdr);
        }
      }
      else if (walker_type == NONCOLLINEAR)
      {
        APP_ABORT(" Error: NONCOLLINEAR not yet implemented in n2r. ");
      }
    }
    TG.TG_local().barrier();
  }

/*******   Interface for PHMSD-like wfns: Reference + excited configurations  *******/
  template<class... Args>
  void accumulate_reference_configuration([[maybe_unused]] Args&&... args)
  {
    APP_ABORT(" Finish: accumulate_reference_configuration ");
  }

  template<class... Args>
  void accumulate_excited_configuration_first([[maybe_unused]] Args&&... args)
  {
    APP_ABORT(" Finish: accumulate_excited_configuration_first ");
  }

  template<class... Args>
  void accumulate_excited_configuration_second([[maybe_unused]] Args&&... args)
  {
    APP_ABORT(" Finish: accumulate_excited_configuration_second ");
  }

  template<class HostCVec>
  void print(int iblock, hdf_archive& dump, HostCVec&& Wsum)
  {
    using std::fill_n;
    const int n_zero = 9;

    if (TG.TG_local().root())
    {
      ma::scal(ComplexType(1.0 / block_size), DMAverage);
      TG.TG_heads().reduce_in_place_n(raw_pointer_cast(DMAverage.origin()), DMAverage.num_elements(), std::plus<>(), 0);
      if (writer)
      {
        dump.push(std::string("N2R"));
        for (int i = 0; i < nave; ++i)
        {
          dump.push(std::string("Average_") + std::to_string(i));
          std::string padded_iblock =
              std::string(n_zero - std::to_string(iblock).length(), '0') + std::to_string(iblock);
          stdCVector_ref DMAverage_(raw_pointer_cast(DMAverage[i].origin()), {dm_size});
          dump.write(DMAverage_, "n2r_" + padded_iblock);
          dump.write(Wsum[i], "denominator_" + padded_iblock);
          dump.pop();
        }
        dump.pop();
      }
    }
    TG.TG_local().barrier();
    fill_n(DMAverage.origin(), DMAverage.num_elements(), ComplexType(0.0, 0.0));
  }

private:
  aux_Allocator aux_alloc;

  int block_size;

  int nave;

  int counter;

  TaskGroup_& TG;

  WALKER_TYPES walker_type;

  int dm_size;

  bool writer;

  bool use_host_memory;

  std::string hdf_walker_output;

  auxCMatrix Orbitals;

  // DMAverage (nave, spin*spin*x*NMO*(x*NMO-1)/2 ), x=(1:CLOSED/COLLINEAR, 2:NONCOLLINEAR)
  mpi3CMatrix DMAverage;

  // buffer space
  auxCVector Buff;
  stdCVector Buff2;

  void set_buffer(size_t N)
  {
    if (Buff.num_elements() < N)
      Buff = auxCVector(iextensions<1u>{N}, aux_alloc);
    using std::fill_n;
    fill_n(Buff.origin(), N, ComplexType(0.0));
  }
  void set_buffer2(size_t N)
  {
    if (Buff2.num_elements() < N)
      Buff2 = stdCVector(iextensions<1u>{N});
    using std::fill_n;
    fill_n(Buff2.origin(), N, ComplexType(0.0));
  }
};

} // namespace afqmc
} // namespace sfqmc

#endif
