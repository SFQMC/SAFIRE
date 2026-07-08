/*
 * This file is distributed under the Apache License, Version 2.0 License.
 * See LICENSE file in top directory for details.
 *
 * Copyright (c) 2021-2025 The Simons Foundation, Inc.
 *
 * You may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *      http://www.apache.org/licenses/LICENSE-2.0
 *
 */

#ifndef SFQMC_AFQMC_SK_HPP
#define SFQMC_AFQMC_SK_HPP

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
#include "Memory/buffer_managers.h"

namespace sfqmc
{
namespace afqmc
{
/* 
 */
template<bool SP>
class sk : public AFQMCInfo
{
  using VType = typename to_working_precision<SP,ComplexType>::type;
  using SPRealType = typename to_working_precision<SP,RealType>::type;

  // allocators
  template<typename T>
  using Allocator = localTG_allocator<T>;

  // type defs
  template<typename T>
  using pointer = device_ptr<T>;

  using IVector        = boost::multi::array<int, 1, Allocator<int>>;
  using CVector        = boost::multi::array<VType, 1, Allocator<VType>>;
  using CMatrix        = boost::multi::array<VType, 2, Allocator<VType>>;
  using C3Tensor       = boost::multi::array<VType, 3, Allocator<VType>>;

  using stdIVector        = boost::multi::array<int, 1>;

  using mpiMatrix    = boost::multi::array<VType, 2, shared_allocator<VType>>;
  using mpi3Tensor    = boost::multi::array<VType, 3, shared_allocator<VType>>;
  using mpi4Tensor    = boost::multi::array<VType, 4, shared_allocator<VType>>;

  using shm_stack_alloc_type = LocalTGBufferManager::template allocator_t<VType>;
  using Ishm_stack_alloc_type = LocalTGBufferManager::template allocator_t<int>;
  using StaticIVector        = boost::multi::static_array<int, 1, Ishm_stack_alloc_type>;
  using StaticVector         = boost::multi::static_array<VType, 1, shm_stack_alloc_type>;
  using StaticMatrix         = boost::multi::static_array<VType, 2, shm_stack_alloc_type>;
  using Static3Tensor        = boost::multi::static_array<VType, 3, shm_stack_alloc_type>;
  using Static4Tensor        = boost::multi::static_array<VType, 4, shm_stack_alloc_type>;

  using host_stack_alloc_type = HostBufferManager::template allocator_t<VType>;
  using HostStaticMatrix      = boost::multi::static_array<VType, 2, host_stack_alloc_type>;
  using HostStatic4Tensor     = boost::multi::static_array<VType, 4, host_stack_alloc_type>;

public:
  sk(afqmc::TaskGroup_& tg_,
                        AFQMCInfo& info,
                        ptree pt,
                        WALKER_TYPES wlk,
                        int nave_ = 1,
                        int bsize = 1)
      : AFQMCInfo(info),
        alloc(make_localTG_allocator<VType>(tg_)),
        block_size(bsize),
        nave(nave_),
        TG(tg_),
        walker_type(wlk),
        writer(false),
        SKAverage({0, 0, 0, 0}, shared_allocator<VType>{TG.TG_local()})
  {
    using std::fill_n;
    app_log(1,"  --  Adding structure factor estimator (real orbitals). -- \n ");
    std::string filename = pt.get<std::string>("filename");
    if (not file_exists(filename))
      APP_ABORT(" Error: Pair densities file does not exist: " + filename);
    stdIVector norbs(iextensions<1u>{0});

    // read pair densities 
    hdf_archive dump;
    if (TG.Node().root())
    {
      if (!dump.open(filename, H5F_ACC_RDONLY))
        APP_ABORT(" Error opening pair density file for sk estimator. ");
      if (dump.push("PikG", false)<0)
        APP_ABORT(" Error in sk: Group PikG not found.");
      
      // read data structures from hdf5 file
      std::vector<int> Idata(2);
      if (TG.Node().root())
        if (!dump.readEntry(Idata, "dims")) 
          APP_ABORT(" Error in sk: Problems reading dims. ");
      TG.Node().broadcast_n(Idata.begin(), 2, 0);
      nkpts = Idata[0];
      ngvecs = Idata[1];

      nmo_per_kp = Vector<int>(iextensions<1u>{nkpts},0);
      kminus = Vector<int>(iextensions<1u>{nkpts},0);
      QKtok2 = Matrix<int>({nkpts,nkpts},0);
      if (!dump.readEntry(nmo_per_kp, "NMOPerKP"))
        APP_ABORT(" Error in sk: Problems reading NMOPerKP. ");
      if (!dump.readEntry(kminus, "MinusK"))
        APP_ABORT(" Error in sk: Problems reading MinusK. "); 
      if (!dump.readEntry(QKtok2, "QKTok2"))
        APP_ABORT(" Error in sk: Problems reading QKtok2. ");

      // broadcast host data  
      TG.Node().broadcast_n(nmo_per_kp.origin(), nkpts, 0);
      TG.Node().broadcast_n(kminus.origin(), nkpts, 0);
      TG.Node().broadcast_n(QKtok2.origin(), nkpts*nkpts, 0);

      nmo_max = *std::max_element(nmo_per_kp.begin(), nmo_per_kp.end());

      // not using possible symmetries yet 
      // LQKGik is padded to nmo_max.
      LQKGik.reserve(nkpts);
      for(int k=0; k<nkpts; k++) 
        LQKGik.emplace_back(mpi3Tensor({nkpts, ngvecs, nmo_max*nmo_max}, 
                                        shared_allocator<VType>{TG.TG_local()}));

      Matrix<VType> LQ({nkpts, nmo_max*nmo_max*ngvecs});
      auto LQ3D = LQ.rotated().partitioned(nmo_max*nmo_max).unrotated();  
      RUNTIME_CHECK(LQ3D.size(0) == nkpts, "");
      RUNTIME_CHECK(LQ3D.size(1) == nmo_max*nmo_max, "");
      RUNTIME_CHECK(LQ3D.size(2) == ngvecs, "");
      for (int Q = 0; Q < nkpts; Q++)
      {
        using ma::conj;
        // no symmetry or distribution yet!
        {
          if (!dump.readEntry(LQ, std::string("L") + std::to_string(Q)))
            APP_ABORT(" Error in sk: Problems reading PikG/L" + std::to_string(Q)); 
          if (LQ.size(0) != nkpts || LQ.size(1) != nmo_max * nmo_max * ngvecs)
            APP_ABORT(" Error in sk: Problems reading PikG/L: Unexpected dimensions. "); 
          for(int k=0; k<nkpts; k++)
            ma::transpose(LQ3D[k],LQKGik[Q][k]);
        }
      }

      dump.pop();   // PikG
      dump.close();

      app_log(1," Number of G vectors: {}",ngvecs);
    }
    else
    {
      std::vector<int> Idata(2);
      TG.Node().broadcast_n(Idata.begin(), 2, 0);
      nkpts = Idata[0];
      ngvecs = Idata[1];

      // broadcast host data  
      TG.Node().broadcast_n(nmo_per_kp.origin(), nkpts, 0);
      TG.Node().broadcast_n(kminus.origin(), nkpts, 0);
      TG.Node().broadcast_n(QKtok2.origin(), nkpts*nkpts, 0);

      nmo_max = *std::max_element(nmo_per_kp.begin(), nmo_per_kp.end());

      // not using possible symmetries yet 
      // LQKGik is padded to nmo_max.
      LQKGik.reserve(nkpts);
      for(int k=0; k<nkpts; k++) 
        LQKGik.emplace_back(mpi3Tensor({nkpts, ngvecs, nmo_max*nmo_max},
                                        shared_allocator<VType>{TG.TG_local()}));
    }
    TG.Node().barrier();

    writer = (TG.Global().rank() == 0);

    SKAverage = mpi4Tensor({nave, 2, nkpts, ngvecs}, 
                                        shared_allocator<VType>{TG.TG_local()});
    fill_n(SKAverage.origin(), SKAverage.num_elements(), VType(0.0, 0.0));

#if defined(ENABLE_CUDA) || defined(ENABLE_HIP)
    nmo_per_kp_dev = nmo_per_kp;
    kminus_dev = kminus;
    QKtok2_dev = QKtok2;
#endif
  }

/*******   Interface for sum over references, e.g. NOMSD ********/
/*
 * Following approach in KP3Index.
 * S(w,Q,G) = sum_Ki_Kl sum_ikjl sum_ss' LQKGik[Q][Ki][i][k][G] * conj(LQKGik[Q][Kl][l][j][G]) *
 *       ( G[Ki][Kk][s]_ik * G[Kj][Kl][s']_jl - delta(s,s') * G[Ki][Kl][s]_il * G[Kj][Kk][s']_jk )
 * S(Q, G) = sum_w Xw S(w,Q,G)
 * (rough algorithm...) 
 * Using intermediate:
 *  T[Q][Ki]_Giwj = sum_l   LQKGik[Q][Ki][i][l][G] * G[Ki][Kl][s]_jl 
 * S(Q, G) = sum_w sum_Ki_Kl sum_ij Xw ( T[Q][Ki]_Giwi * T[Q][Kl]_Gjwj - 
 *                           delta(s,s') T[Q][Ki]_Giwj * T[Q][Kl]_Gjwi )
 * LQKGik is kept in mpi3 memory and moved to the device on demand if needed.
 */
  template<class MatG, class MatG_host, class HostCVec1>
  void accumulate(int iav, MatG&& G, [[maybe_unused]] MatG_host&& G_host, [[maybe_unused]] HostCVec1&& Xw, bool impsamp)
  {
//    if(nkpts == 1)
//      accumulate_impl_single_kp(iav,G,Xw,impsamp);
//    else
      accumulate_impl_multi_kp(iav,G,Xw,impsamp);

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
  void print(int iblock, hdf_archive& dump, [[maybe_unused]] HostCVec&& Wsum)
  {
    using std::fill_n;
    const int n_zero = 9;
    int nQ = SKAverage.size(2);
    int nG = SKAverage.size(3);

    if (TG.TG_local().root())
    {
      ma::scal(VType(SPRealType(1.0 / block_size)), SKAverage);
      TG.TG_heads().reduce_in_place_n(raw_pointer_cast(SKAverage.origin()), 
                                      SKAverage.num_elements(), std::plus<>(), 0);
      if (writer)
      {
        dump.push(std::string("SK"));
        for (int t = 0; t < type_id.size(); ++t)
        {
          dump.push(type_id[t]);
          for (int i = 0; i < nave; ++i)
          {
            dump.push(std::string("Average_") + std::to_string(i));
            std::string padded_iblock =
                std::string(n_zero - std::to_string(iblock).length(), '0') + std::to_string(iblock);
            Matrix_ref<VType> SKAverage_(raw_pointer_cast(SKAverage[i][t].origin()), {nQ,nG});
            dump.write(SKAverage_, "SK_" + type_id[t] + padded_iblock);
            dump.pop();
          }
          dump.pop();
        }
        dump.pop();
      }
    }
    TG.TG_local().barrier();
    fill_n(SKAverage.origin(), SKAverage.num_elements(), VType(0.0, 0.0));
  }

private:

  Allocator<VType> alloc;

  int block_size = 1;

  int nave = 1;

  TaskGroup_& TG;

  WALKER_TYPES walker_type = UNDEFINED_WALKER_TYPE; 

  bool writer = false;

  std::vector<std::string> type_id = {"CC","SS"};

  int default_buffer_size_in_MB = 2048;
  int nmo_max=0;
  int nkpts = 0;
  int ngvecs = 0;

  // on shm or device memory
  Vector<int> nmo_per_kp;
  Vector<int> kminus;
  Matrix<int> QKtok2;

// should be defined(ENABLE_ACCELERATOR)
#if defined(ENABLE_CUDA) || defined(ENABLE_HIP)
  Vector<int, Allocator<int>> nmo_per_kp_dev;
  Vector<int, Allocator<int>> kminus_dev;
  Matrix<int, Allocator<int>> QKtok2_dev;  
#endif

  // keeping large tensor in shared CPU memory. 
  // In GPU builds, moving to device on demand. 
  std::vector<mpi3Tensor> LQKGik;

  // type:0-cc, 1-ss
  // nG = # of points in reciprocal space.
  // SKAverage (nave, 2, nkpts, ngvecs)
  mpi4Tensor SKAverage;

  template<class MatG, class HostCVec1>
  void accumulate_impl_single_kp([[maybe_unused]] int iav, [[maybe_unused]] MatG&& G, [[maybe_unused]] HostCVec1&& Xw, [[maybe_unused]] bool impsamp)
  {
    APP_ABORT(" Error: Specialization not yet implemented in sk::accumulate. \n\n");
  }  

  template<class MatG, class HostCVec1>
  void accumulate_impl_multi_kp([[maybe_unused]] int iav, [[maybe_unused]] MatG&& G, [[maybe_unused]] HostCVec1&& Xw, [[maybe_unused]] bool impsamp)
  {
    if(TG.TG_local().size() > 1)
      APP_ABORT(" Error: ncores>1 not yet implemented in sk::accumulate. \n\n");
    static_assert(std::decay<MatG>::type::dimensionality == 4, "Wrong dimensionality");
    using ma::gemmBatched;
    using std::copy_n;
    using std::fill_n;
    RUNTIME_CHECK(G.size(0) == Xw.size(0), "");

    // assumes G[nwalk][spin][M][M]
    int nwalk(G.size(0));
    int nspin = walker_type == COLLINEAR ? 2 : 1;
    int npol = walker_type == NONCOLLINEAR ? 2 : 1;

    // if needed, write customized version for gamma point/super cell calculations 
    //int buffer_size = std::max(default_buffer_size_in_MB, buffer_manager.available_in_MB());
    int buffer_size = default_buffer_size_in_MB;
    long Bytes = long(buffer_size) * 1024L * 1024L;
    Bytes /= size_t(nwalk * npol * nmo_max * npol * nmo_max * ngvecs * sizeof(VType));
    long bz0 = std::max(2L, Bytes);
    // batch_size includes the factor of 2 from Q/Qm pair
    long batch_size = std::min(bz0, long(2 * nkpts * nkpts));
    // since I need 2 copies of the intermediates
    batch_size = batch_size/2; 

    //int batch_cnt(0);
    using ma::gemmBatched;
    using vptr = pointer<VType>;
    std::vector<vptr> A1array, A2array;
    std::vector<vptr> B1array, B2array;
    std::vector<vptr> C1array, C2array;
    A1array.reserve(batch_size);
    B1array.reserve(batch_size);
    C1array.reserve(batch_size);
    A2array.reserve(batch_size);
    B2array.reserve(batch_size);
    C2array.reserve(batch_size);

    std::vector<VType> scl_factors;
    scl_factors.reserve(batch_size);

    LocalTGBufferManager buffer_manager;
    StaticVector dev_scl_factors(iextensions<1u>{batch_size},
                     buffer_manager.get_generator().template get_allocator<VType>());
    Static4Tensor T1({batch_size, ngvecs, nwalk, npol*nmo_max*npol*nmo_max},
                     buffer_manager.get_generator().template get_allocator<VType>());
    Static4Tensor T2({batch_size, ngvecs, nwalk, npol*nmo_max*npol*nmo_max},
                     buffer_manager.get_generator().template get_allocator<VType>());

    Static3Tensor LQ(LQKGik[0].extensions(),
                     buffer_manager.get_generator().template get_allocator<VType>());

    for (int spin = 0; spin < nspin; ++spin)
    {

      // GKK
      Static3Tensor GKK({nkpts, nkpts, nwalk * npol * nmo_max * npol * nmo_max},
                      buffer_manager.get_generator().template get_allocator<VType>());
//      GKaKjw_to_GKKwaj(G.rotated()[spin].unrotated(), GKK, nmo_max, nmo_per_kp);

      for (int Q = 0; Q < nkpts; ++Q)
      {
        //int Qm      = kminus[Q];

        // simple implementation for now
        A1array.clear(); A2array.clear();
        B1array.clear(); B2array.clear();
        C1array.clear(); C2array.clear();
        //batch_cnt = 0;

        // copy LQKGik from host memory
        copy_n(raw_pointer_cast(LQKGik[Q].origin()), LQ.num_elements(), LQ.origin());

/*
        for (int Ka = 0; Ka < nkpts; ++Ka)
        {
          for (int Kb = 0; Kb < nkpts; ++Kb)
          {
            int Kl = QKToK2[Qm][Kb];
            int Kk  = QKToK2[Q][Ka];

            // T1 = sum_l conj(L[Q][Kl][l][j][G]) * G[Ki][Kl][w][i][l]
            A1array.push_back(vptr(LQ[Kl].origin()));
            B1array.push_back(GKK[Ka][Kl_].origin());
            C1array.push_back(T1[batch_cnt].origin());

            // T2 = sum_k L[Q][Ki][i][k][G] * G[Kj][Kk][w][j][k]
            A2array.push_back(vptr(LQ[Ka].origin()));
            B2array.push_back(GKK[Kb][Kk].origin());
            C2array.push_back(T2[batch_cnt++].origin());

            scl_factors.push_back(VType(-scl));

            if (batch_cnt >= batch_size)
            {
              gemmBatched('T', 'N', nocc_max * nchol_max, nwalk * nocc_max, npol * nmo_max, SPComplexType(1.0),
                          Aarray.data(), npol * nmo_max, Barray.data(), npol * nmo_max, SPComplexType(0.0),
                          Carray.data(), nocc_max * nchol_max, Aarray.size());

              copy_n(scl_factors.data(), scl_factors.size(), dev_scl_factors.origin());
              Apwabn_Apwban_Bw(dev_scl_factors, T1, E.rotated()[1].unrotated());

//              if (addEJ)
//              {
//                int nc0 = Q2vbias[Q] / 2; //std::accumulate(ncholpQ.begin(),ncholpQ.begin()+Q,0);
//                copy_n(kdiag.data(), kdiag.size(), IMats.origin());
//                using ma::batched_Tab_to_Klr;
//                batched_Tab_to_Klr(kdiag.size(), nwalk, nocc_max, nchol_max, local_nCV, ncholpQ[Q], nc0,
//                                   IMats.origin(), T1.origin(), Kl.origin(), Kr.origin());
//              }

              // reset
              A1array.clear(); A2array.clear();
              B1array.clear(); B2array.clear();
              C1array.clear(); C2array.clear();
              batch_cnt = 0;
            }
          }
        }

        if (batch_cnt > 0)
        {
          gemmBatched('T', 'N', nocc_max * nchol_max, nwalk * nocc_max, npol * nmo_max, SPComplexType(1.0),
                      Aarray.data(), npol * nmo_max, Barray.data(), npol * nmo_max, SPComplexType(0.0), Carray.data(),
                      nocc_max * nchol_max, Aarray.size());

          copy_n(scl_factors.data(), scl_factors.size(), dev_scl_factors.origin());
          Apwabn_Apwban_Bw(dev_scl_factors, T1, E.rotated()[1].unrotated());
                                raw_pointer_cast(E[0].origin()) + 1, E.stride(0));

//          if (addEJ)
//          {
//            int nc0 = Q2vbias[Q] / 2; //std::accumulate(ncholpQ.begin(),ncholpQ.begin()+Q,0);
//            copy_n(kdiag.data(), kdiag.size(), IMats.origin());
//            using ma::batched_Tab_to_Klr;
//            batched_Tab_to_Klr(kdiag.size(), nwalk, nocc_max, nchol_max, local_nCV, ncholpQ[Q], nc0, IMats.origin(),
//                               T1.origin(), Kl.origin(), Kr.origin());
//          }
        }        
*/
      } // Q 
    }  // spin
  }

};

} // namespace afqmc
} // namespace sfqmc

#endif
