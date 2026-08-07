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
#include "AFQMC/parameters.hpp"
#include <vector>
#include <string>
#include <iostream>

#include "nda/nda.hpp"
#include "nda/h5.hpp"
#include "nda/tensor.hpp"

#include "AFQMC/Walkers/WalkerSet.hpp"
#include "numerics/shared_array/const_shared_array.hpp"

namespace sfqmc
{
namespace afqmc
{
/* 
 * Observable class that calculates the walker averaged 1 RDM.
 * The resulting RDM will be [spin][x*NMO][x*NMO],
 * where x:2 for NONCOLLINEAR and 1 for everything else.
 */
class full1rdm
{

public:
  full1rdm() {
    utils::check(false, "Error in Observables::full1rdm: Reached disabled default constructor.");
  }

  full1rdm(std::shared_ptr<utils::mpi_context_t<boost::mpi3::communicator>> _mpi, const OneRDMParameters& params, WALKER_TYPES wlk, int NMO_, int nave = 1)
      : mpi(_mpi),
        walker_type(wlk),
        NMO{NMO_},
        apply_rotation(false),
        XRot{},
        print_from_list(false),
        index_list{},
        DMAverage(0, 0)
  {
    app_log(1,"  --  Adding Full 1RDM (OneRDM) estimator. -- ");
    std::string rot_file, h5_path; 
    rot_file = params.rotation;
    h5_path = params.path;
    print_from_list = params.with_index_list;
    int nspin = walker_type == COLLINEAR ? 2 : 1;
    int npol = walker_type == NONCOLLINEAR ? 2 : 1;

    
    if (rot_file != "")
    {
      std::optional<nda::array<ComplexType,3>> R;
      std::optional<nda::array<int,2>> I;
      {
        std::ifstream f(rot_file.c_str());
        utils::check(f.good()," Error: File with rotation matrix does not exist: {}",rot_file);
      }
      apply_rotation  = true;

      if (mpi->comm.root())
      {
        R.emplace();
        h5::file file(rot_file,'r');
        h5::group grp_(file);
        utils::check(grp_.has_key(h5_path), "Missing h5 dataset:{}",h5_path);
        h5::group grp = grp_.open_group(h5_path);
        nda::h5_read(grp,"RotationMatrix",*R);

        utils::check(R->extent(0) == nspin, "Error Wrong dimensions in RotationMatrix.");
        utils::check(R->extent(2) == npol*NMO, "Error Wrong dimensions in RotationMatrix.");
        // conjugate rotation matrix
        (*R)() = nda::conj((*R)());
        if (print_from_list) {
          I.emplace();
          nda::h5_read(grp,"Indices",*I);
          utils::check(I->extent(1) == 2, "Error Wrong dimensions in Indices.");
        }
      }
      XRot = memory::share_from_root(*mpi, [&]() {
        return R.value()();
      });
      
      if (print_from_list) {
        index_list = memory::share_from_root(*mpi, [&]() {
          return I.value()();
        });
      }

      if (print_from_list)
        dm_size = index_list.extent(0);
      else
        dm_size = nspin * XRot().extent(0) * XRot().extent(0);
    }
    else
    {
      // can also add print_from_list option without rotation later on
      dm_size = nspin * npol * NMO * npol * NMO;
    }

    ncalls = 0;
    DMAverage.resize(nave, dm_size);
    DMAverage() = ComplexType(0.0, 0.0);
  }

/*******   Interface for sum over references, e.g. NOMSD ********/
  void accumulate(int iav, nda::MemoryArrayOfRank<4> auto const& G, 
                  nda::MemoryArrayOfRank<4> auto const& G_host, 
                  nda::MemoryVector auto const& Xw, [[maybe_unused]] bool impsamp)
  {
    memory::check_memory_space<HOST_MEMORY>(G_host,Xw);
    // assumes G[nwalk][spin][M][M]
    utils::check(G.extent(0) == Xw.extent(0), "Shape mismatch");
    // increase counter
    ncalls++;
    int nwalk = Xw.extent(0);
    int nspin = walker_type == COLLINEAR ? 2 : 1 ;
    int npol = walker_type == NONCOLLINEAR ? 2 : 1 ;
    utils::check(G.shape() == std::array<long,4>{nwalk,nspin,npol*NMO,npol*NMO}, "Shape mismatch");
    utils::check(G_host.shape() == G.shape(), "Shape mismatch");
    if (apply_rotation) {

       utils::check(walker_type != NONCOLLINEAR,"Error: Not yet implemented: acc_with_rotation && noncollinear.");

      int nX   = XRot().extent(0);
      // Grot = Xc * G * H(Xc)
      memory::buffered_array<HOST_MEMORY,ComplexType,4> T1(nwalk,nspin,nX,npol*NMO); 
      memory::buffered_array<HOST_MEMORY,ComplexType,4> Grot(G_host.shape()); 

      nda::tensor::contract(nda::conj(XRot()),"sai",G_host,"wsij",T1,"wsaj");
      nda::tensor::contract(T1,"wsaj,",XRot(),"sbj",Grot,"wsab");

      if (print_from_list)
      {
        for (int p = 0; p < index_list.extent(0); p++)
        {
          int is = index_list()(p,0)/(npol*NMO); 
          int i = index_list()(p,0)%(npol*NMO); 
          int j = index_list()(p,1);
          utils::check(is >=0 and is < nspin, "index_list out of bounds in full1rdm");
          DMAverage(iav,p) += nda::blas::dot(Xw(nda::range::all),Grot(nda::range::all,is,i,j)); 
        }
      }
      else {
        auto G2D = nda::reshape(Grot,std::array<long,2>{nwalk,nspin*npol*NMO*npol*NMO});
        nda::blas::gemv(ComplexType(1.0), nda::transpose(G2D), Xw, 
                  ComplexType(1.0), DMAverage(iav,nda::range::all));
      }

    } else {

      // keeping on host since it is quick
      // DMAverage[iav][ij] += sum_iw G_host[iw][ij] Xw[iw] = T( G_host ) * Xw 
      auto G2D = nda::reshape(G_host,std::array<long,2>{nwalk,nspin*npol*NMO*npol*NMO});
      nda::blas::gemv(ComplexType(1.0), nda::transpose(G2D), Xw, 
                ComplexType(1.0), DMAverage(iav,nda::range::all));

    }
  }

/*******   Interface for PHMSD-like wfns: Reference + excited configurations  *******/
  template<class... Args>
  void accumulate_reference_configuration([[maybe_unused]] Args&&... args)
  {
    utils::check(false," Finish: accumulate_reference_configuration ");
  }

  template<class... Args>
  void accumulate_excited_configuration_first([[maybe_unused]] Args&&... args)
  {
     utils::check(false," Finish: accumulate_excited_configuration_first ");
  }

  template<class... Args>
  void accumulate_excited_configuration_second([[maybe_unused]] Args&&... args)
  {
     utils::check(false," Finish: accumulate_excited_configuration_second "); 
  }

  void print(int iblock, h5::group *group, nda::MemoryVector auto && Wsum)
  {
    memory::check_memory_space<HOST_MEMORY>(Wsum);
    DMAverage() *= ComplexType(1.0 / double(ncalls));
    mpi->all_reduce(DMAverage, std::plus<>{});
    if (mpi->comm.root())
    {
      assert(group);
      h5::group parent = ( group->has_key("FullOneRDM") ? 
                           group->open_group("FullOneRDM") : 
                           group->create_group("FullOneRDM") );
      for (int i = 0; i < DMAverage.shape(0); ++i)
      {
        std::string avg_name = std::format("Average_{}", i);
        h5::group obs_group = ( parent.has_key(avg_name) ? 
                                parent.open_group(avg_name) :
                                parent.create_group(avg_name) );
        std::string padded_iblock = std::format("{:09}", iblock);
        h5::write(obs_group, "one_rdm_" + padded_iblock, nda::to_host(nda::flatten(DMAverage(i, nda::ellipsis{}))));
        h5::write(obs_group, "denominator_" + padded_iblock, Wsum[i]);
      }
    }
    ncalls=0;
    DMAverage() = ComplexType(0.0, 0.0);
  }

private:
  std::shared_ptr<utils::mpi_context_t<boost::mpi3::communicator>> mpi;

  WALKER_TYPES walker_type;
  int NMO{};

  int ncalls = 0;

  int dm_size;

  bool apply_rotation;

  // currently spin/polarization independent rotation. Can generalize if needed
  memory::const_shared_array<HOST_MEMORY,ComplexType,3> XRot;

  bool print_from_list;

  memory::const_shared_array<HOST_MEMORY,int,2> index_list;

  // DMAverage (nave, nspin*npol*NMO*npol*NMO), x=(1:CLOSED/COLLINEAR, 2:NONCOLLINEAR)
  nda::array<ComplexType,2> DMAverage;

};

} // namespace afqmc
} // namespace sfqmc

