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


#include <cstdlib>
#include <iostream>
#include <fstream>
#include <vector>
#include <string>
#include <ctype.h>

#include "AFQMC/config.h"
#include "utilities/check.hpp"
#include "IO/app_loggers.h"
#include "readWfn.h"
#include "utilities/h5_utils.hpp"

#include "nda/nda.hpp"
#include "nda/h5.hpp"

#include "numerics/sparse/sparse.hpp"

namespace sfqmc
{
namespace afqmc
{

std::tuple<int,int,int,int> getWavefunctionDims(std::string filename)
{
  h5::file file(filename,'r');
  h5::group grp(file);
  h5::group wgrp = grp.open_group("Wavefunction");
  std::string name;
  if (wgrp.has_key("NOMSD")) {
    name = std::string("NOMSD");
  } else if (wgrp.has_key("PHMSD")) {
    name = std::string("PHMSD");
  } else {
    utils::check(false, "Missing NOMSD/PHMSD block."); 
  }
  h5::group ngrp = wgrp.open_group(name);
  std::vector<int> dims(5);
  h5::h5_read(ngrp,"dims",dims);
  return std::make_tuple(dims[4],dims[0],dims[1],dims[2]);
}

WAVEFUNCTION_TYPES getWavefunctionType(std::string filename)
{
  std::string type;
  h5::file file(filename,'r');
  h5::group grp(file);
  h5::group wgrp = grp.open_group("Wavefunction");
  if (wgrp.has_key("NOMSD")) {
    return NOMSD_WFN;
  } else if (wgrp.has_key("PHMSD")) {
    return PHMSD_WFN;
  }
  utils::check(false, "Unknown wavefunction type in getWavefunctionType.");
  return NOMSD_WFN;
}

WALKER_TYPES getWalkerType(std::string filename, std::string type)
{
  h5::file file(filename,'r');
  h5::group grp(file);
  h5::group wgrp = grp.open_group("Wavefunction");
  if(type == "any") {
    if( wgrp.has_key("NOMSD") )
      type = "NOMSD";
    else if( wgrp.has_key("PHMSD") )
      type = "PHMSD";
    else
      utils::check(false,"Missing NOMSD/PHMSD datasets in Wavefunction.");
  }
  
  utils::check(wgrp.has_key(type), "Missing wfn type:{}",type);
  h5::group ngrp = wgrp.open_group(type);
  std::vector<int> Idata(5);
  h5::read(ngrp,"dims",Idata);
  
  return initWALKER_TYPES(Idata[3]);
}

void read_ph_wavefunction_hdf(h5::group& grp,
                              nda::array<ComplexType,1>& ci_coeff,
                              nda::array<int,2>& occs,
                              int& ndets,
                              WALKER_TYPES walker_type,
                              int NMO,
                              int nup,
                              int ndown,
                              nda::array<PsiT_Matrix<HOST_MEMORY>, 1>& PsiT,
                              std::string& type)
{
  int npol = (walker_type == NONCOLLINEAR ? 2 : 1);
  utils::check(walker_type != UNDEFINED_WALKER_TYPE, "Undefined walker type.");
  utils::check(walker_type != CLOSED, " walker_type==CLOSED not yet implemented in read_ph_wavefunction_hdf.");
  bool mixed = false;
  int NEL    = nup + (walker_type == COLLINEAR ? ndown : 0);

  /*
   * type:
   *   - occ: All determinants are specified with occupation numbers
   *
   *   - 0: excitations out of a RHF reference
   *          NOTE: Does not mean perfect pairing, means excitations from a single reference
   *   - 1: excitations out of a UHF reference (not yet working)
   */
  WALKER_TYPES wtype;
  getCommonInput(grp, ndets, ci_coeff, wtype);
  // make first coefficient positive (or maybe largest???)
  ci_coeff() *= ( std::real(ci_coeff(0)) < 0.0 ? -1.0 : 1.0 );  
  utils::check(wtype != CLOSED, " walker_type==CLOSED not yet implemented for PHMSD Trial wavefunctions.");
  utils::check(wtype != NONCOLLINEAR, " walker_type==NONCOLLINEAR not yet implemented for PHMSD Trial wavefunctions. Contact developers if you need this feature.");

  // limiting to this for now, kind of irrelevant until we find a FCI code that works in
  // a UHF basis
  utils::check(wtype == walker_type, " walker_type ({}) in wavefunction file differs from input file ({}).", walkerTypeToString(wtype), walkerTypeToString(walker_type));

  int type_;
  h5::h5_read(grp,"type",type_);
  if (type_ == 0) {
    type = "occ";
  } else if(type_ == 1) {
    type = "mixed";
    mixed = true;
  } else if(type_ == 2) {
    type = "mixed";
    mixed = true;
  } else {
    utils::check(false,"Unknown value of dataset type.");
  }

  if (mixed)
  { // read reference
    PsiT.resize( ( type_ == 2 ? 2 : 1) );

    {
      h5::group g = grp.open_group("PsiT_"+ std::to_string(0));
      PsiT(0) = math::sparse::HDF2CSR<ComplexType,HOST_MEMORY,int,int>(g);
      utils::check(PsiT(0).extent(1) == npol*NMO, 
                   "For PHMSD type=mixed, PsiT.size(1) must be npol*NMO");
    }
    if (type_ == 2)
    {
      utils::check(walker_type == COLLINEAR,
                   "walker_type must be COLLINEAR, got {}", walkerTypeToString(walker_type));
      h5::group g = grp.open_group("PsiT_"+ std::to_string(1));
      PsiT(1) = math::sparse::HDF2CSR<ComplexType,HOST_MEMORY,int,int>(g);
      utils::check(PsiT(1).extent(1) == npol*NMO, 
                   "For PHMSD type=mixed, PsiT.size(1) must be npol*NMO");
    }
  }
  nda::array<int,1> buff;
  nda::h5_read(grp,"occs",buff);
  utils::check(buff.size() >= ndets*NEL," occupation array too small." );
  occs.resize(ndets,NEL);
  std::copy_n(buff.data(),ndets*NEL,occs.data());
}

template<MEMORY_SPACE MEM>
ph_excitations<int, ComplexType, MEM> build_ph_struct(nda::array<ComplexType,1> const& ci_coeff,
                                                 nda::array<int, 2>& occs,
                                                 int ndets,
                                                 int NMO,
                                                 int nup,
                                                 int ndown)
{
  using nda::range;
  ComplexType ci;
  // count number of k-particle excitations
  // counts[0] has special meaning, it must be equal to nup+ndown.
  std::vector<long> counts_alpha(nup + 1);
  std::vector<long> counts_beta(ndown + 1);
  // ugly but need dynamic memory allocation
  std::vector<std::vector<int>> unique_alpha(nup + 1);
  std::vector<std::vector<int>> unique_beta(ndown + 1);
  // reference configuration, taken as the first one right now
  nda::array<int,1> refa(nup);
  nda::array<int,1> refb(ndown);
  // space for excitation string identifying the current configuration
  std::vector<int> exct;
  // record file position to come back
  std::vector<int> Iwork; // work arrays for permutation calculation
  std::streampos start;
  {
    Iwork.resize(2 * nup);
    exct.reserve(2 * nup);
    for (int i = 0; i < ndets; i++)
    {
      ci = ci_coeff[i];
      // alpha
      for (int k = 0, q = 0; k < nup; k++)
      {
        q = occs(i,k);
        utils::check(q>=0 and q<NMO, "Bad occupation number " + std::to_string(q) + " in determinant " + std::to_string(i) + " in wavefunction file. ");
      }
      if (i == 0)
      {
        refa() = occs(0,range(nup));
      }
      else
      {
        int np = get_excitation_number(true, refa, occs(i,range(nup)), exct, ci, Iwork);
        push_excitation(exct, unique_alpha[np]);
      }
      if(ndown==0) continue; // NONCOLLINEAR
      // beta
      for (int k = 0, q = 0; k < ndown; k++)
      {
        q = occs(i,nup + k);
        utils::check(q>=NMO and q<2*NMO,"Bad occupation number " + std::to_string(q) + " in determinant " + std::to_string(i) + " in wavefunction file. ");
      }
      if (i == 0)
      {
        refb() = occs(0,range(nup,nup+ndown));
      }
      else
      {
        int np = get_excitation_number(true, refb, occs(i,range(nup,nup+ndown)), exct, ci, Iwork);
        push_excitation(exct, unique_beta[np]);
      }
    }
    // now that we have all unique configurations, count
    for (int i = 1; i <= nup; i++)
      counts_alpha[i] = unique_alpha[i].size();
    for (int i = 1; i <= ndown; i++)
      counts_beta[i] = unique_beta[i].size();
  }
  // using int for now, but should move to short later when everything works well
  // ph_struct stores the reference configuration on the index [0]
  ph_excitations<int, ComplexType, MEM> ph_struct(ndets, nup, ndown, counts_alpha, counts_beta);

  {
    std::map<int, int> refa2loc;
    for (int i = 0; i < nup; i++)
      refa2loc[refa[i]] = i;
    std::map<int, int> refb2loc;
    for (int i = 0; i < ndown; i++)
      refb2loc[refb[i]] = i;
    // add reference
    ph_struct.add_reference(refa, refb);
    // add unique configurations
    // alpha
    for (int n = 1; n < unique_alpha.size(); n++)
      for (std::vector<int>::iterator it = unique_alpha[n].begin(); it < unique_alpha[n].end(); it += (2 * n))
        ph_struct.add_alpha(n, it);
    // beta
    for (int n = 1; n < unique_beta.size(); n++)
      for (std::vector<int>::iterator it = unique_beta[n].begin(); it < unique_beta[n].end(); it += (2 * n))
        ph_struct.add_beta(n, it);
    // read configurations
    int alpha_index;
    int beta_index;
    int np;
    for (int i = 0; i < ndets; i++)
    {
      ci = ci_coeff[i];
      for (int k = 0, q = 0; k < nup; k++)
      {
        q = occs(i,k);
        utils::check(q>=0 and q<NMO,"Bad occupation number " + std::to_string(q) + " in determinant " + std::to_string(i) + " in wavefunction file. ");
      }
      np = get_excitation_number(true, refa, occs(i,range(nup)), exct, ci, Iwork);
      alpha_index =
          ((np == 0) ? (0)
                     : (find_excitation(exct, unique_alpha[np]) + ph_struct.number_of_unique_smaller_than(np)[0]));
      if(ndown==0) {
        ph_struct.add_configuration(alpha_index, 0, ci);
	continue;
      }
      for (int k = 0, q = 0; k < ndown; k++)
      {
        q = occs(i,nup + k);
        utils::check(q>=NMO and q<2*NMO,"Bad occupation number " + std::to_string(q) + " in determinant " + std::to_string(i) + " in wavefunction file. ");
      }
      np = get_excitation_number(true, refb, occs(i,range(nup,nup+ndown)), exct, ci, Iwork);
      beta_index =
          ((np == 0) ? (0) : (find_excitation(exct, unique_beta[np]) + ph_struct.number_of_unique_smaller_than(np)[1]));
      ph_struct.add_configuration(alpha_index, beta_index, ci);
    }
  }
  return ph_struct;
}


int get_number_of_determinants(std::vector<int> const& dims, int requested) {
  if(requested < 1) {
    return dims[4];
  }
  if(requested > dims[4]) {
    app_warning("Found less determinants than requested, adjusting request: requested {} > {} in file.", requested, dims[4]);
    return dims[4];
  }
  return requested;
}

/*
 * Read trial wavefunction information from file.
 */
void getCommonInput(h5::group& grp,
                    int& ndets_to_read,
                    nda::array<ComplexType,1>& ci,
                    WALKER_TYPES& walker_type)
{
  // check for consistency in parameters
  std::vector<int> dims(5);
  h5::read(grp,"dims",dims);
  ndets_to_read = get_number_of_determinants(dims, ndets_to_read);
  app_log(1," - Number of determinants in trial wavefunction: {} ", ndets_to_read);
  ci.resize(ndets_to_read);
  nda::array<ComplexType,1> ci_t(dims[4]);
  utils::h5_read(grp,"ci_coeffs",ci_t);
  walker_type = initWALKER_TYPES(dims[3]);
  ci() = ci_t(nda::range(ndets_to_read)); 
}

// instantiate
 template ph_excitations<int, ComplexType, HOST_MEMORY> 
build_ph_struct<HOST_MEMORY>(nda::array<ComplexType,1> const&,nda::array<int, 2>&,int,int,int,int);

#if defined(ENABLE_DEVICE)
 template ph_excitations<int, ComplexType, DEVICE_MEMORY> 
build_ph_struct<DEVICE_MEMORY>(nda::array<ComplexType,1> const&,nda::array<int, 2>&,int,int,int,int);
#endif


} // namespace afqmc

} // namespace sfqmc
