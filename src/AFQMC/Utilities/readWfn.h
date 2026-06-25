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

#include <cstdlib>
#include <iostream>
#include <fstream>
#include <vector>
#include <string>
#include <ctype.h>

#include "AFQMC/config.h"

#include "nda/h5.hpp"
#include "utilities/h5_utils.hpp"
#include "utilities/check_shape.hpp"
#include "numerics/sparse/sparse.hpp"
#include "AFQMC/Wavefunctions/Excitations.hpp"

namespace sfqmc
{
namespace afqmc
{
void read_ph_wavefunction_hdf(h5::group& grp,
                              nda::array<ComplexType,1>& ci_coeff,
                              nda::array<int,2>& occs,
                              int& ndets,
                              WALKER_TYPES walker_type,
                              int NMO,
                              int nup,
                              int ndown,
                              nda::array<PsiT_Matrix<HOST_MEMORY>, 1>& PsiT_MO,
                              std::string& type);

template<MEMORY_SPACE MEM>
ph_excitations<int, ComplexType, MEM> build_ph_struct(nda::array<ComplexType,1> const& ci_coeff,
                                                 nda::array<int, 2>& occs,
                                                 int ndets,
                                                 int NMO,
                                                 int nup,
                                                 int ndown);

int get_number_of_determinants(std::vector<int> const& dims, int requested);

void getCommonInput(h5::group& g,
                    int& ndets_to_read,
                    nda::array<ComplexType,1>& ci,
                    WALKER_TYPES& walker_type);

WALKER_TYPES getWalkerType(std::string filename, std::string type = "any");

WAVEFUNCTION_TYPES getWavefunctionType(std::string filename);
std::tuple<int,int,int,int> getWavefunctionDims(std::string filename);


template<MEMORY_SPACE MEM>
auto read_nomsd_wavefunction(h5::group& grp,int requested_ndets,
                            WALKER_TYPES walker_type, int NMO, int nup, int ndown)
{
  using csr = PsiT_Matrix<MEM>;
  long nspin = (walker_type == COLLINEAR ? 2 : 1);
  long npol = (walker_type == NONCOLLINEAR ? 2 : 1);

  std::vector<int> dims(5);
  h5::h5_read(grp,"dims",dims);
  int ndets = get_number_of_determinants(dims, requested_ndets);

  // keep in on host at first
  nda::array<csr, 2> psi(ndets,nspin);
  WALKER_TYPES wfn_type = afqmc::initWALKER_TYPES(dims[3]);

  utils::check(walkerTypeIsConvertible(wfn_type, walker_type), "{} trial wavefunction is not compatible with {} walkers", walkerTypeToString(wfn_type), walkerTypeToString(walker_type));

  auto nel = std::to_array({nup, ndown});

  if(wfn_type == walker_type) {
    for(int id=0, k=0; id<ndets; ++id) {
      for(int is=0; is<nspin; ++is, ++k) {
        h5::group pgrp = grp.open_group("PsiT_"+std::to_string(k));
        psi(id,is) = std::move(math::sparse::HDF2CSR<ComplexType,MEM,int,int>(pgrp));
        utils::check_shape(psi(id,is), std::format("psi({},{})", id, is), nel[is], npol*NMO);
      } 
    }
  } else if(wfn_type == COLLINEAR) { 
    // upgrade from COLLINEAR to NONCOLLINEAR
    for(int id=0; id<ndets; ++id) {
      h5::group ugrp = grp.open_group("PsiT_"+std::to_string(2*id));
      auto up = math::sparse::HDF2CSR<ComplexType,HOST_MEMORY,int,int>(ugrp);
      h5::group dgrp = grp.open_group("PsiT_"+std::to_string(2*id+1));
      auto dn = math::sparse::HDF2CSR<ComplexType,HOST_MEMORY,int,int>(dgrp);
      // combining, shift dn by NMO 
      psi(id,0) = math::sparse::combine_csr(up,dn,NMO);
      utils::check_shape(psi(id,0), std::format("psi({},0)",id), nup, npol*NMO);
    }
  } else {
    utils::check(wfn_type == CLOSED, "Logic error: wfn_type: {}, walker_type: {}", walkerTypeToString(wfn_type), walkerTypeToString(walker_type)); 
    utils::check(walker_type == COLLINEAR or walker_type == NONCOLLINEAR, "Logic error."); 
    if(walker_type == COLLINEAR) {
      // upgrade from CLOSED to COLLINEAR 
      utils::check(nup==ndown, "Problems upgrading wavefunction: nup != ndown when upgrading to COLLINEAR");
      for(int id=0; id<ndets; ++id) {
        h5::group pgrp = grp.open_group("PsiT_"+std::to_string(id));
        psi(id,0) = std::move(math::sparse::HDF2CSR<ComplexType,MEM,int,int>(pgrp));
        utils::check_shape(psi(id,0), std::format("psi({},0)",id), nup, npol*NMO);
        psi(id,1) = psi(id,0);
        utils::check_shape(psi(id,1), std::format("psi({},1)",id), ndown, npol*NMO);
      }
    } else {
      // upgrade from CLOSED to NONCOLLINEAR 
      for(int id=0; id<ndets; ++id) {
        h5::group ugrp = grp.open_group("PsiT_"+std::to_string(id));
        auto up = math::sparse::HDF2CSR<ComplexType,HOST_MEMORY,int,int>(ugrp);
        utils::check(up.extent(1) == NMO, "Shape mismatch");
        utils::check(2*up.extent(0) == nel[0], "Shape mismatch");
        // combining, shift dn by NMO 
        psi(id,0) = math::sparse::combine_csr(up,up,NMO);
        utils::check_shape(psi(id,0), std::format("psi({},0)",id), nup, npol*NMO);
      }
    }
  }

  return psi;
}

template<MEMORY_SPACE MEM>
auto read_nomsd_wavefunction(h5::group& grp,int ndets,
                            WALKER_TYPES walker_type, int NMO, int ntau)
{
  using csr = PsiT_Matrix<MEM>;
  long nspin = (walker_type == COLLINEAR_FT ? 2 : 1);

  std::vector<int> dims(5);
  h5::read(grp,"dims",dims);
  utils::check(NMO==dims[0], "Inconsistent NMO.");
  utils::check(ntau == dims[1], "Inconsistent  ntau.");
  utils::check(int(walker_type) >= dims[3],
               "Inconsistent walker_type.");
  utils::check(ndets <= dims[4], "Inconsistent  ndets_to_read.");

  // keep in on host at first
  nda::array<csr, 3> psi(ndets,nspin,3);

  WALKER_TYPES wfn_type = afqmc::initWALKER_TYPES(dims[3]);

  if(wfn_type == walker_type) {
    for(int id=0, k=0; id<ndets; ++id) {
      for(int is=0; is<nspin; ++is, ++k) {
        h5::group pgrp = grp.open_group("UL_"+std::to_string(k));
        psi(id,is,0) = std::move(math::sparse::HDF2CSR<ComplexType,MEM,int,int>(pgrp));
        pgrp = grp.open_group("DL_"+std::to_string(k));
        psi(id,is,1) = std::move(math::sparse::HDF2CSR<ComplexType,MEM,int,int>(pgrp));
        pgrp = grp.open_group("VL_"+std::to_string(k));
        psi(id,is,2) = std::move(math::sparse::HDF2CSR<ComplexType,MEM,int,int>(pgrp));
        //utils::check(psi(id,is).shape() == std::array<long,2>{nel[is],NMO}, "Shape mismatch");
      } 
    }
  } else if(wfn_type == COLLINEAR_FT) { 
    utils::check(walker_type == NONCOLLINEAR_FT, "walker_type==COLLINEAR incompatible with wfn_type:{}",int(wfn_type));
    // upgrade from COLLINEAR to NONCOLLINEAR
    for(int id=0; id<ndets; ++id) {
      h5::group ugrp = grp.open_group("UL_"+std::to_string(2*id));
      auto up = math::sparse::HDF2CSR<ComplexType,HOST_MEMORY,int,int>(ugrp);
      h5::group dgrp = grp.open_group("UL_"+std::to_string(2*id+1));
      auto dn = math::sparse::HDF2CSR<ComplexType,HOST_MEMORY,int,int>(dgrp);
      utils::check(up.extent(1) == NMO, "Shape mismatch");
      utils::check(dn.extent(1) == NMO, "Shape mismatch");
      //utils::check(up.extent(0) + dn.extent(1) == nel[0], "Shape mismatch");
      // combining, shift dn by NMO 
      psi(id,0,0) = math::sparse::combine_csr(up,dn,NMO);
      ugrp = grp.open_group("DL_"+std::to_string(2*id));
      up = math::sparse::HDF2CSR<ComplexType,HOST_MEMORY,int,int>(ugrp);
      dgrp = grp.open_group("DL_"+std::to_string(2*id+1));
      dn = math::sparse::HDF2CSR<ComplexType,HOST_MEMORY,int,int>(dgrp);
      utils::check(up.extent(1) == NMO, "Shape mismatch");
      utils::check(dn.extent(1) == NMO, "Shape mismatch");
      //utils::check(up.extent(0) + dn.extent(1) == nel[0], "Shape mismatch");
      // combining, shift dn by NMO 
      psi(id,0,1) = math::sparse::combine_csr(up,dn,NMO);
      ugrp = grp.open_group("VL_"+std::to_string(2*id));
      up = math::sparse::HDF2CSR<ComplexType,HOST_MEMORY,int,int>(ugrp);
      dgrp = grp.open_group("VL_"+std::to_string(2*id+1));
      dn = math::sparse::HDF2CSR<ComplexType,HOST_MEMORY,int,int>(dgrp);
      utils::check(up.extent(1) == NMO, "Shape mismatch");
      utils::check(dn.extent(1) == NMO, "Shape mismatch");
      //utils::check(up.extent(0) + dn.extent(1) == nel[0], "Shape mismatch");
      // combining, shift dn by NMO 
      psi(id,0,2) = math::sparse::combine_csr(up,dn,NMO);
    } 
  } else {
    utils::check(wfn_type == CLOSED, "Closed wavefunction type not implemented for finite-T");
    /*
    utils::check(wfn_type == CLOSED, "Logic error: wfn_type:{}, walker_type:{}",int(wfn_type),int(walker_type)); 
    utils::check(walker_type == COLLINEAR or walker_type == NONCOLLINEAR, "Logic error."); 
    if(walker_type == COLLINEAR) {
      // upgrade from CLOSED to COLLINEAR 
      utils::check(nup==ndown, "Problems upgrading wavefunction: nup!=ndown when upgrading to COLLINEAR");
      for(int id=0; id<ndets; ++id) {
        h5::group pgrp = grp.open_group("PsiT_"+std::to_string(id));
        psi(id,0) = std::move(math::sparse::HDF2CSR<ComplexType,MEM,int,int>(pgrp));
        utils::check(psi(id,0).shape() == std::array<long,2>{nel[0],NMO}, "Shape mismatch");
        psi(id,1) = psi(id,0);
      }
    } else {
      // upgrade from CLOSED to NONCOLLINEAR 
      for(int id=0; id<ndets; ++id) {
        h5::group ugrp = grp.open_group("PsiT_"+std::to_string(id));
        auto up = math::sparse::HDF2CSR<ComplexType,HOST_MEMORY,int,int>(ugrp);
        utils::check(up.extent(1) == NMO, "Shape mismatch");
        utils::check(2*up.extent(0) == nel[0], "Shape mismatch");
        // combining, shift dn by NMO 
        psi(id,0) = math::sparse::combine_csr(up,up,NMO);
      }
    }
    */
  }

  return psi;
}

} // namespace afqmc

} // namespace sfqmc

