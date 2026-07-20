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

#ifndef SFQMC_AFQMC_GENERALIZEDFOCKMATRIX_HPP
#define SFQMC_AFQMC_GENERALIZEDFOCKMATRIX_HPP

#include "AFQMC/config.h"
#include <vector>
#include <string>
#include <iostream>

#include "hdf/hdf_multi.h"
#include "hdf/hdf_archive.h"

#include "AFQMC/Walkers/WalkerSet.hpp"
#include "Numerics/ma_operations.hpp"
#include "Memory/buffer_managers.h"

#include "AFQMC/Wavefunctions/Wavefunction.hpp"

namespace sfqmc
{
namespace afqmc
{
/* 
 * Observable class that calculates the walker averaged "full" 1 RDM.
 * In this context, "full" means that no contraction over the RDM is
 * being performed. The resulting RDM will be [spin][x*NMO][x*NMO],
 * where x:2 for NONCOLLINEAR and 1 for everything else.
 */
class generalizedFockMatrix
{
  // allocators
  using Allocator        = device_allocator<ComplexType>;
  using Allocator_shared = localTG_allocator<ComplexType>;

  // type defs
  using pointer              = typename std::allocator_traits<Allocator>::pointer;
  using const_pointer        = typename std::allocator_traits<Allocator>::const_pointer;
  using pointer_shared       = typename std::allocator_traits<Allocator_shared>::pointer;
  using const_pointer_shared = typename std::allocator_traits<Allocator_shared>::const_pointer;

  using CVector_ref    = boost::multi::array_ref<ComplexType, 1, pointer>;
  using CMatrix_ref    = boost::multi::array_ref<ComplexType, 2, pointer>;
  using CVector        = boost::multi::array<ComplexType, 1, Allocator>;
  using CMatrix        = boost::multi::array<ComplexType, 2, Allocator>;
  using sharedCMatrix  = boost::multi::array<ComplexType, 2, Allocator_shared>;
  using sharedCTensor  = boost::multi::array<ComplexType, 3, Allocator_shared>;
  using stdCVector_ref = boost::multi::array_ref<ComplexType, 1>;
  using stdCMatrix_ref = boost::multi::array_ref<ComplexType, 2>;
  using stdCVector     = boost::multi::array<ComplexType, 1>;
  using stdCMatrix     = boost::multi::array<ComplexType, 2>;
  using stdIMatrix     = boost::multi::array<int, 2>;
  using mpi3CVector    = boost::multi::array<ComplexType, 1, shared_allocator<ComplexType>>;
  using mpi3IMatrix    = boost::multi::array<int, 2, shared_allocator<int>>;
  using mpi3CMatrix    = boost::multi::array<ComplexType, 2, shared_allocator<ComplexType>>;
  using mpi3CTensor    = boost::multi::array<ComplexType, 3, shared_allocator<ComplexType>>;
  using mpi3C4Tensor   = boost::multi::array<ComplexType, 4, shared_allocator<ComplexType>>;

  using stack_alloc_type = LocalTGBufferManager::template allocator_t<ComplexType>;
  using Static3Tensor    = boost::multi::static_array<ComplexType, 3, stack_alloc_type>;

  using host_stack_alloc_type = HostBufferManager::template allocator_t<ComplexType>;
  using HostStatic3Tensor     = boost::multi::static_array<ComplexType, 3, host_stack_alloc_type>;

public:
  generalizedFockMatrix(afqmc::TaskGroup_& tg_,
                        int NMO_,
                        [[maybe_unused]] ptree pt,
                        WALKER_TYPES wlk,
                        Wavefunction& wfn_, 
                        int nave_ = 1,
                        int bsize = 1)
      : NMO{NMO_},
        TG(tg_),
        walker_type(wlk),
	wfn(wfn_),
        writer(false),
        block_size(bsize),
        nave(nave_),
        DMAverage({0, 0, 0}, shared_allocator<ComplexType>{TG.TG_local()})
  {
    using std::copy_n;
    using std::fill_n;

    app_log(1,"  --  Adding generalized Fock matrix (genFock) estimator. -- ");

    dm_size = NMO * NMO;
    if (walker_type == COLLINEAR)
      dm_size *= 2;
    else if (walker_type == NONCOLLINEAR)
      dm_size *= 4;

    using std::fill_n;
    writer = (TG.Global().rank() == 0);

    DMAverage = mpi3CTensor({2, nave, dm_size}, shared_allocator<ComplexType>{TG.TG_local()});
    fill_n(DMAverage.origin(), DMAverage.num_elements(), ComplexType(0.0, 0.0));
  }

/*******   Interface for sum over references, e.g. NOMSD ********/
  template<class MatG, class MatG_host, class HostCVec1>
  void accumulate(int iav, MatG&& G, [[maybe_unused]] MatG_host&& G_host, 
                  HostCVec1&& Xw, [[maybe_unused]] bool impsamp)
  {
    static_assert(std::decay<MatG>::type::dimensionality == 4, "Wrong dimensionality");
    static_assert(std::decay<MatG_host>::type::dimensionality == 4, "Wrong dimensionality");
    using std::fill_n;
    // assumes G[nwalk][spin][M][M]
    int nw(G.size(0));
    RUNTIME_CHECK(G.size(0) == Xw.size(0), "");
    RUNTIME_CHECK(G[0].num_elements() == dm_size, "");

    LocalTGBufferManager buffer_manager;
    Static3Tensor gFock({2, nw, dm_size}, buffer_manager.get_generator().template get_allocator<ComplexType>());

    wfn.generalizedFockMatrix(G, gFock[0], gFock[1]);

    int i0, iN;
    std::tie(i0, iN) = FairDivideBoundary(TG.TG_local().rank(), dm_size, TG.TG_local().size());

#if defined(ENABLE_CUDA) || defined(ENABLE_HIP)
    HostBufferManager host_buffer_manager;
    HostStatic3Tensor Buff({2,nw,dm_size},
                    host_buffer_manager.get_generator().template get_allocator<ComplexType>());
    copy_n(gFock.origin(), gFock.num_elements(), Buff.origin());
    for (int ic = 0; ic < 2; ic++)
      ma::product(ComplexType(1.0), ma::T(Buff[ic]), Xw, ComplexType(1.0), DMAverage[ic][iav]);
#else
    for (int ic = 0; ic < 2; ic++)
    {
        ma::product(ComplexType(1.0), ma::T(gFock[ic].rotated().sliced(i0,iN).unrotated()), Xw, 
                    ComplexType(1.0), DMAverage[ic][iav].sliced(i0, iN));
    }
#endif
    TG.TG_local().barrier();
  }

/*******   Interface for PHMSD-like wfns: Reference + excited configurations  *******/
  template<class... Args>
  void accumulate_reference_configuration([[maybe_unused]] Args&&... args)
  {
    APP_ABORT(" Finish: accumulate_reference_configuration ");
  }

  template<class... Args>
  void accumulate_excited_configuration_first([[maybe_unused]]Args&&... args)
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
        dump.push(std::string("GenFockPlus"));
        for (int i = 0; i < nave; ++i)
        {
          dump.push(std::string("Average_") + std::to_string(i));
          std::string padded_iblock =
              std::string(n_zero - std::to_string(iblock).length(), '0') + std::to_string(iblock);
          stdCVector_ref DMAverage_(raw_pointer_cast(DMAverage[0][i].origin()), {dm_size});
          dump.write(DMAverage_, "gfockp_" + padded_iblock);
          dump.write(Wsum[i], "denominator_" + padded_iblock);
          dump.pop();
        }
        dump.pop();
        dump.push(std::string("GenFockMinus"));
        for (int i = 0; i < nave; ++i)
        {
          dump.push(std::string("Average_") + std::to_string(i));
          std::string padded_iblock =
              std::string(n_zero - std::to_string(iblock).length(), '0') + std::to_string(iblock);
          stdCVector_ref DMAverage_(raw_pointer_cast(DMAverage[1][i].origin()), {dm_size});
          dump.write(DMAverage_, "gfockm_" + padded_iblock);
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
  TaskGroup_& TG;

  WALKER_TYPES walker_type;

  int NMO = -1;

  Wavefunction& wfn;

  bool writer;

  int block_size;

  int nave;

  int dm_size;

  // DMAverage (2, nave, spin*x*NMO*x*NMO), x=(1:CLOSED/COLLINEAR, 2:NONCOLLINEAR)
  mpi3CTensor DMAverage;
};

} // namespace afqmc
} // namespace sfqmc

#endif
