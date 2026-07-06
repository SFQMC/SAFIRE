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

#ifndef SFQMC_AFQMC_REALSPACE_CORRELATORS_HPP
#define SFQMC_AFQMC_REALSPACE_CORRELATORS_HPP

#include "AFQMC/config.h"
#include <vector>
#include <string>

#include "hdf/hdf_archive.h"

#include "Numerics/detail/utilities.hpp"
#include "Numerics/ma_operations.hpp"

namespace sfqmc
{
namespace afqmc
{
/* 
 */
class realspace_correlators : public AFQMCInfo
{
  // allocators
  using Allocator = localTG_allocator<ComplexType>;

  // type defs
  using pointer = device_ptr<ComplexType>;

  using CVector_ref    = boost::multi::array_ref<ComplexType, 1, pointer>;
  using CMatrix_ref    = boost::multi::array_ref<ComplexType, 2, pointer>;
  using CTensor_ref    = boost::multi::array_ref<ComplexType, 3, pointer>;
  using C4Tensor_ref   = boost::multi::array_ref<ComplexType, 4, pointer>;
  using CVector        = boost::multi::array<ComplexType, 1, Allocator>;
  using CMatrix        = boost::multi::array<ComplexType, 2, Allocator>;
  using stdCVector     = boost::multi::array<ComplexType, 1>;
  using stdCMatrix     = boost::multi::array<ComplexType, 2>;
  using stdIVector     = boost::multi::array<int, 1>;
  using stdCVector_ref = boost::multi::array_ref<ComplexType, 1>;
  using stdCMatrix_ref = boost::multi::array_ref<ComplexType, 2>;
  using mpi3CVector    = boost::multi::array<ComplexType, 1, shared_allocator<ComplexType>>;
  using mpi3CMatrix    = boost::multi::array<ComplexType, 2, shared_allocator<ComplexType>>;
  using mpi3CTensor    = boost::multi::array<ComplexType, 3, shared_allocator<ComplexType>>;
  using mpi3C4Tensor   = boost::multi::array<ComplexType, 4, shared_allocator<ComplexType>>;

  using shm_stack_alloc_type = LocalTGBufferManager::template allocator_t<ComplexType>;
  using StaticMatrix         = boost::multi::static_array<ComplexType, 2, shm_stack_alloc_type>;
  using Static4Tensor        = boost::multi::static_array<ComplexType, 4, shm_stack_alloc_type>;

  using host_stack_alloc_type = HostBufferManager::template allocator_t<ComplexType>;
  using HostStaticMatrix      = boost::multi::static_array<ComplexType, 2, host_stack_alloc_type>;
  using HostStatic4Tensor     = boost::multi::static_array<ComplexType, 4, host_stack_alloc_type>;

public:
  realspace_correlators(afqmc::TaskGroup_& tg_,
                        AFQMCInfo& info,
                        ptree pt,
                        WALKER_TYPES wlk,
                        int nave_ = 1,
                        int bsize = 1)
      : AFQMCInfo(info),
        alloc(make_localTG_allocator<ComplexType>(tg_)),
        block_size(bsize),
        nave(nave_),
        TG(tg_),
        walker_type(wlk),
        dm_size(0),
        writer(false),
        Orbitals({0, 0}, alloc),
        DMAverage({0, 0, 0}, shared_allocator<ComplexType>{TG.TG_local()}),
        Orrp({0, 0}, shared_allocator<ComplexType>{TG.TG_local()})
  {
    app_log(1,"  --  Adding real-space off diagonal 2RDM (realspace_correlators). -- \n ");
    std::string orb_file = pt.get<std::string>("orbitals");
    if (not file_exists(orb_file))
    {
      app_error(" Error: orbitals file does not exist: {}", orb_file);
      APP_ABORT("");
    }

    stdIVector norbs(iextensions<1u>{0});
    dm_size     = 0;
    int npoints = 0;

    // read orbitals
    hdf_archive dump;
    if (TG.Node().root())
    {
      if (!dump.open(orb_file, H5F_ACC_RDONLY))
        APP_ABORT(" Error opening orbitals file for realspace_correlators estimator. ");
      if (dump.push("OrbsR", false)<0)
        APP_ABORT(" Error in realspace_correlators: Group OrbsR not found.");
      // read one orbital to check size and later corroborate all orbitals have same size
      stdCVector orb(iextensions<1u>{1});
      if (!dump.readEntry(orb, "kp0_b0"))
        APP_ABORT(" Error in realspace_correlators: Problems reading orbital: 0  0");
      npoints = orb.size(0);
      if (npoints < 1)
        APP_ABORT(" Error in realspace_correlators: npoints < 1. ");
      TG.Node().broadcast_n(&npoints, 1, 0);
      if (!dump.readEntry(norbs, "number_of_orbitals"))
	APP_ABORT(" Error in realspace_correlators: Problems reading number_of_orbitals. ");
      int M = 0, nk = norbs.size();
      for (int i = 0; i < nk; i++)
        M += norbs[i];
      if (M != NMO)
      {
        app_error(" Error in realspace_correlators: Inconsistent number of orbitals in file: M:{} expected:{}",M,NMO);
        APP_ABORT("");
      }
      TG.Node().broadcast_n(&npoints, 1, 0);
      Orbitals = CMatrix({NMO, npoints}, alloc);
      // host copy to calculate Orrp
      stdCMatrix host_orb({NMO, npoints});
      Orrp = mpi3CMatrix({npoints, npoints}, shared_allocator<ComplexType>{TG.TG_local()});
      for (int k = 0, kn = 0; k < nk; k++)
      {
        for (int i = 0; i < norbs[k]; i++, kn++)
        {
          if (!dump.readEntry(orb, "kp" + std::to_string(k) + "_b" + std::to_string(i)))
          {
            app_error(" Error in realspace_correlators: Inconsistent orbital size: {} {} ",k,i);
            APP_ABORT("");
          }
          if (orb.size(0) != npoints)
          {
            app_error(" Error in realspace_correlators: Inconsistent orbital size: {} {} ",k,i);
            APP_ABORT("");
          }
          using std::copy_n;
          copy_n(orb.origin(), npoints, ma::pointer_dispatch(Orbitals[kn].origin()));
          copy_n(orb.origin(), npoints, host_orb[kn].origin());
        }
      }
      dump.pop();
      dump.close();

      // Or(r,r') = sum_j conj(psi(j,r)) * psi(j,r')
      ma::product(ma::H(host_orb), host_orb, Orrp);

      app_log(1," Number of grid points: {}",npoints); 
    }
    else
    {
      TG.Node().broadcast_n(&npoints, 1, 0);
      Orbitals = CMatrix({NMO, npoints}, alloc);
      Orrp     = mpi3CMatrix({npoints, npoints}, shared_allocator<ComplexType>{TG.TG_local()});
    }
    dm_size = npoints * npoints;
    TG.Node().barrier();

    using std::fill_n;
    writer = (TG.Global().rank() == 0);

    if (writer)
    {
      type_id.reserve(3);
      type_id.emplace_back("CC");
      type_id.emplace_back("SS");
      type_id.emplace_back("CS");
    }

    DMAverage = mpi3CTensor({nave, 3, dm_size}, shared_allocator<ComplexType>{TG.TG_local()});
    fill_n(DMAverage.origin(), DMAverage.num_elements(), ComplexType(0.0, 0.0));
  }

/*******   Interface for sum over references, e.g. NOMSD ********/
  template<class MatG, class MatG_host, class HostCVec1>
  void accumulate(int iav, MatG&& G, [[maybe_unused]] MatG_host&& G_host, HostCVec1&& Xw,[[maybe_unused]] bool impsamp)
  {
    static_assert(std::decay<MatG>::type::dimensionality == 4, "Wrong dimensionality");
    static_assert(std::decay<MatG_host>::type::dimensionality == 4, "Wrong dimensionality");
    using ma::gemmBatched;
    using std::copy_n;
    using std::fill_n;
    RUNTIME_CHECK(G.size(0) == Xw.size(0), "");

    // assumes G[nwalk][spin][M][M]
    int nw(G.size(0));
    int npts(Orbitals.size(1));
    int nsp = ( walker_type == COLLINEAR or walker_type == COLLINEAR_FT ? 2 : 1 );

    LocalTGBufferManager buffer_manager;
#if defined(ENABLE_CUDA) || defined(ENABLE_HIP)
    // on CPU memory on GPU build
    HostBufferManager host_buffer_manager;
    HostStatic4Tensor Gr_host({nw, nsp, npts, npts}, 
                    host_buffer_manager.get_generator().template get_allocator<ComplexType>());
#else
    // in shared memory on CPU build
    Static4Tensor Gr_host({nw, nsp, npts, npts}, 
                    buffer_manager.get_generator().template get_allocator<ComplexType>());
#endif

    // calculate green functions in real space and send to host for processing/accumulation
    // if memory becomes a problem, then batch over walkers
    {
      StaticMatrix T({nw * nsp * NMO, npts}, buffer_manager.get_generator().template get_allocator<ComplexType>());
      StaticMatrix Gr({nsp * nw, npts * npts}, buffer_manager.get_generator().template get_allocator<ComplexType>());
      CTensor_ref Gr3D(make_device_ptr(Gr.origin()), {nw, nsp, npts * npts});
      CTensor_ref T3D(make_device_ptr(T.origin()), {nw, nsp, NMO * npts});
      CMatrix_ref G2D(make_device_ptr(G.origin()), {nw * nsp * NMO, NMO});

      // T1[iw][ispin][i][r] = sum_j G[iw][ispin][i][j] * Psi(j,r)
      int i0, iN;
      std::tie(i0, iN) = FairDivideBoundary(TG.TG_local().rank(), int(G2D.size(0)), TG.TG_local().size());
      ma::product(G2D.sliced(i0, iN), Orbitals, T.sliced(i0, iN));
      TG.TG_local().barrier();

      // G[iw][ispin][r][r'] = sum_ij conj( Psi(i,r) ) * G[iw][ispin][i][j] * Psi(j,r')
      //                     = sum_i conj( Psi(i,r) ) T1[iw][ispin][i][r'] = H(Psi) * T1
      std::vector<decltype(ma::pointer_dispatch(Orbitals.origin()))> Aarray;
      Aarray.reserve(nw * nsp);
      std::vector<decltype(ma::pointer_dispatch(Orbitals.origin()))> Barray;
      Barray.reserve(nw * nsp);
      std::vector<decltype(ma::pointer_dispatch(Orbitals.origin()))> Carray;
      Carray.reserve(nw * nsp);
      for (int iw = 0, p = 0; iw < nw; ++iw)
        for (int ispin = 0; ispin < nsp; ++ispin, ++p)
          if (p % TG.TG_local().size() == TG.TG_local().rank())
          {
            Aarray.emplace_back(ma::pointer_dispatch(Orbitals.origin()));
            Barray.emplace_back(ma::pointer_dispatch(T3D[iw][ispin].origin()));
            Carray.emplace_back(ma::pointer_dispatch(Gr3D[iw][ispin].origin()));
          }
      // careful with fortran ordering
      gemmBatched('N', 'C', npts, npts, NMO, ComplexType(1.0), Barray.data(), npts, Aarray.data(), npts,
                  ComplexType(0.0), Carray.data(), npts, int(Aarray.size()));
      TG.TG_local().barrier();
      std::tie(i0, iN) = FairDivideBoundary(TG.TG_local().rank(), int(Gr.num_elements()), TG.TG_local().size());
      copy_n(Gr.origin() + i0, (iN - i0), Gr_host.origin() + i0);
    }
    TG.TG_local().barrier();

    // : Z(type,r,r') = < (nup_r OP1 ndn_r) (nup_r' OP2 ndn_r') >
    // where type= 0: charge-charge --> ++
    //             1: spin-spin     --> --
    //             2: charge-spin   --> +-
    // THIS COULD BE WRITTEN WITH OPENMP TO ENABLE GPU EASILY!!!
    for (int iw = 0; iw < nw; iw++)
    {
      if (walker_type == CLOSED)
      {
        auto&& Gur = Gr_host[iw][0];
        stdCMatrix_ref DMcc(raw_pointer_cast(DMAverage[iav][0].origin()), {npts, npts});
        stdCMatrix_ref DMss(raw_pointer_cast(DMAverage[iav][1].origin()), {npts, npts});
        stdCMatrix_ref DMcs(raw_pointer_cast(DMAverage[iav][2].origin()), {npts, npts});
        auto X_ = 2.0 * Xw[iw];
        for (int i = 0; i < npts; i++) {
          if (i % TG.TG_local().size() != TG.TG_local().rank())
            continue;
          for (int j = 0; j < npts; j++)
          {
            auto g1 = Gur[i][i] * Gur[j][j];
            auto g2 = Gur[i][j] * Gur[j][i];
            auto o1 = Gur[i][j] * Orrp[j][i];
            DMcc[i][j] += X_ * (2.0 * g1 - g2 + o1);
            DMss[i][j] += X_ * (-g2 + o1);
            // no cs in RHF
          }
        }
      }
      else if (walker_type == COLLINEAR or walker_type == COLLINEAR_FT)
      {
        auto&& Gur = Gr_host[iw][0];
        auto&& Gdr = Gr_host[iw][1];
        stdCMatrix_ref DMcc(raw_pointer_cast(DMAverage[iav][0].origin()), {npts, npts});
        stdCMatrix_ref DMss(raw_pointer_cast(DMAverage[iav][1].origin()), {npts, npts});
        stdCMatrix_ref DMcs(raw_pointer_cast(DMAverage[iav][2].origin()), {npts, npts});
        auto X_ = Xw[iw];
        for (int i = 0; i < npts; i++) {
          if (i % TG.TG_local().size() != TG.TG_local().rank())
            continue;
          for (int j = 0; j < npts; j++)
          {
            auto guu = Gur[i][i] * Gur[j][j] - Gur[i][j] * Gur[j][i] + Gur[i][j] * Orrp[j][i];
            auto gdd = Gdr[i][i] * Gdr[j][j] - Gdr[i][j] * Gdr[j][i] + Gdr[i][j] * Orrp[j][i];
            auto gud = Gur[i][i] * Gdr[j][j];
            auto gdu(Gdr[i][i] * Gur[j][j]);
            DMcc[i][j] += X_ * (guu + gud + gdu + gdd);
            DMss[i][j] += X_ * (guu - gud - gdu + gdd);
            DMcs[i][j] += X_ * (guu - gud + gdu - gdd);
          }
        }
      }
      else if (walker_type == NONCOLLINEAR or walker_type == NONCOLLINEAR_FT)
      {
        APP_ABORT(" Noncollinear not implemented \n\n");
        auto&& Gur = Gr_host[iw][0];
        auto&& Gdr = Gr_host[iw][1];
        stdCMatrix_ref DMcc(raw_pointer_cast(DMAverage[iav][0].origin()), {npts, npts});
        stdCMatrix_ref DMss(raw_pointer_cast(DMAverage[iav][1].origin()), {npts, npts});
        stdCMatrix_ref DMcs(raw_pointer_cast(DMAverage[iav][2].origin()), {npts, npts});
        auto X_ = Xw[iw];
        for (int i = 0; i < npts; i++) {
          if (i % TG.TG_local().size() != TG.TG_local().rank())
            continue;
          for (int j = 0; j < npts; j++)
          {
            auto guu = Gur[i][i] * Gur[j][j] - Gur[i][j] * Gur[j][i];
            auto gdd = Gdr[i][i] * Gdr[j][j] - Gdr[i][j] * Gdr[j][i];
            auto gud = Gur[i][i] * Gdr[j][j] - Gur[i][j] * Gdr[j][i];
            auto gdu = Gdr[i][i] * Gur[j][j] - Gdr[i][j] * Gur[j][i];
            DMcc[i][j] += X_ * (guu + gud + gdu + gdd);
            DMss[i][j] += X_ * (guu - gud - gdu + gdd);
            DMcs[i][j] += X_ * (guu - gud + gdu - gdd);
          }
        }
      }
    }
    TG.TG_local().barrier();
  }

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
        dump.push(std::string("OD2RDM"));
        for (int t = 0; t < 3; ++t)
        {
          dump.push(type_id[t]);
          for (int i = 0; i < nave; ++i)
          {
            dump.push(std::string("Average_") + std::to_string(i));
            std::string padded_iblock =
                std::string(n_zero - std::to_string(iblock).length(), '0') + std::to_string(iblock);
            stdCVector_ref DMAverage_(raw_pointer_cast(DMAverage[i][t].origin()), {dm_size});
            dump.write(DMAverage_, "od2rdm_" + type_id[t] + padded_iblock);
            dump.write(Wsum[i], "denominator_" + padded_iblock);
            dump.pop();
          }
          dump.pop();
        }
        dump.pop();
      }
    }
    TG.TG_local().barrier();
    fill_n(DMAverage.origin(), DMAverage.num_elements(), ComplexType(0.0, 0.0));
  }

private:
  Allocator alloc;

  int block_size;

  int nave;

  TaskGroup_& TG;

  WALKER_TYPES walker_type;

  int dm_size;

  bool writer;

  std::vector<std::string> type_id;

  // unfolding of orbitals to super cell (in KP case) is assumed done
  // as a preprocessing step
  CMatrix Orbitals;

  // type:0-cc, 1-ss, 2-cs (c=charge, s=spin)
  // np = # of points in real space.
  // DMAverage (nave, type, np*np )
  mpi3CTensor DMAverage;

  mpi3CMatrix Orrp;
};

} // namespace afqmc
} // namespace sfqmc

#endif
