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
#include "AFQMC/parameters.hpp"
#include "utilities/check.hpp"
#include "IO/app_loggers.h"
#include "readWfn.h"
#include "format_version.hpp"
#include "utilities/h5_utils.hpp"

#include "nda/nda.hpp"
#include "nda/h5.hpp"

#include "numerics/sparse/sparse.hpp"

namespace sfqmc
{
namespace afqmc
{

namespace {

/// Extent `dim` of dataset `name` in `grp`.
int dataset_extent(h5::group grp, std::string const& name, int dim) {
  return h5::array_interface::get_dataset_info(grp, name).lengths[dim];
}

/// The group holding the wavefunction of representation `type` ("NOMSD", "PHMSD", or "any").
h5::group open_wavefunction_group(h5::group wgrp, std::string type) {
  if(type == "any") {
    if(wgrp.has_key("NOMSD")) {
      type = "NOMSD";
    } else if(wgrp.has_key("PHMSD")) {
      type = "PHMSD";
    } else {
      utils::check(false, "Missing NOMSD/PHMSD datasets in Wavefunction.");
    }
  }
  utils::check(wgrp.has_key(type), "Missing wfn type:{}", type);
  return wgrp.open_group(type);
}

} // namespace

WALKER_TYPES read_spin_type(h5::group grp) {
  if(!has_format_version(grp) && grp.has_key("dims")) {
    // CoQuí writes neither a format_version nor a spin_type attribute, only the walker type in
    // slot 3 of a 'dims' array
    std::vector<int> dims;
    h5::h5_read(grp, "dims", dims);
    utils::check(dims.size() == 5, "Wavefunction 'dims' has length {}, expected 5.", dims.size());
    utils::check(dims[3] >= CLOSED && dims[3] <= NONCOLLINEAR, "Wavefunction 'dims' has invalid walker type {}.", dims[3]);
    return static_cast<WALKER_TYPES>(dims[3]);
  }
  check_format_version(grp, "Wavefunction");
  std::string spin_type;
  h5::h5_read_attribute(grp, "spin_type", spin_type);
  auto const type = nlohmann::json(spin_type).get<WALKER_TYPES>();
  utils::check(type != UNDEFINED_WALKER_TYPE, "Wavefunction spin_type is \"{}\".", spin_type);
  return type;
}

WavefunctionInfo read_wavefunction_info(h5::group ngrp) {
  WavefunctionInfo info{.walker_type = read_spin_type(ngrp),
                        .NMO         = 0,
                        .nup         = 0,
                        .ndown       = 0,
                        .ndets       = dataset_extent(ngrp, "ci_coeffs", 0)};
  auto const [nspin, npol] = walkerTypeToDims(info.walker_type);

  if(ngrp.has_key("occa")) { // PHMSD
    // an occupation-only expansion has no array that spans the orbitals, so the count is recorded
    h5::h5_read_attribute(ngrp, "number_of_orbitals", info.NMO);
    info.nup   = dataset_extent(ngrp, "occa", 1);
    info.ndown = dataset_extent(ngrp, "occb", 1);
  } else if(ngrp.has_key("UL_0")) { // finite-temperature NOMSD
    info.NMO   = math::sparse::hdf_csr_shape(ngrp.open_group("UL_0"))[1] / npol;
    info.nup   = math::sparse::hdf_csr_shape(ngrp.open_group("DL_0"))[0];
    info.ndown = 0;
  } else {
    auto const [nup, ncols] = math::sparse::hdf_csr_shape(ngrp.open_group("PsiT_0"));
    info.NMO = ncols / npol;
    info.nup = nup;
    if(info.walker_type == COLLINEAR) {
      info.ndown = math::sparse::hdf_csr_shape(ngrp.open_group("PsiT_1"))[0];
    } else if(info.walker_type == CLOSED) {
      info.ndown = nup;
    } else {
      info.ndown = 0;
    }
  }
  return info;
}

std::tuple<int, int, int> read_info_from_wfn(std::string fileName, std::string type) {
  h5::file file(fileName, 'r');
  auto const info = read_wavefunction_info(open_wavefunction_group(h5::group(file).open_group("Wavefunction"), type));
  return std::make_tuple(info.NMO, info.nup, info.ndown);
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
  return read_spin_type(open_wavefunction_group(h5::group(file).open_group("Wavefunction"), type));
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
                              PHMSDOrbitalType& type)
{
  int npol = (walker_type == NONCOLLINEAR ? 2 : 1);
  utils::check(walker_type != UNDEFINED_WALKER_TYPE, "Undefined walker type.");
  utils::check(walker_type != CLOSED, " walker_type==CLOSED not yet implemented in read_ph_wavefunction_hdf.");
  int NEL    = nup + (walker_type == COLLINEAR ? ndown : 0);

  WALKER_TYPES wtype;
  getCommonInput(grp, ndets, ci_coeff, wtype);
  // make first coefficient positive (or maybe largest???)
  ci_coeff() *= ( std::real(ci_coeff(0)) < 0.0 ? -1.0 : 1.0 );  
  utils::check(wtype != CLOSED, " walker_type==CLOSED not yet implemented for PHMSD Trial wavefunctions.");
  utils::check(wtype != NONCOLLINEAR, " walker_type==NONCOLLINEAR not yet implemented for PHMSD Trial wavefunctions. Contact developers if you need this feature.");

  // limiting to this for now, kind of irrelevant until we find a FCI code that works in
  // a UHF basis
  utils::check(wtype == walker_type, " walker_type ({}) in wavefunction file differs from input file ({}).", walkerTypeToString(wtype), walkerTypeToString(walker_type));

  // the orbital references are numbered PsiT_0, PsiT_1, ... without gaps: none, one, or one per spin
  int nreferences = 0;
  while(grp.has_subgroup("PsiT_" + std::to_string(nreferences))) {
    ++nreferences;
  }
  utils::check(nreferences <= 2, "PHMSD wavefunction has {} orbital references, expected at most 2.", nreferences);
  utils::check(nreferences < 2 || walker_type == COLLINEAR,
               "a PHMSD wavefunction with one orbital reference per spin needs COLLINEAR walkers, got {}",
               walkerTypeToString(walker_type));
  type = (nreferences == 0 ? PHMSDOrbitalType::occ : PHMSDOrbitalType::mixed);

  PsiT.resize(nreferences);
  for(int n = 0; n < nreferences; ++n) {
    h5::group g = grp.open_group("PsiT_" + std::to_string(n));
    PsiT(n) = math::sparse::HDF2CSR<ComplexType,HOST_MEMORY,int,int>(g);
    utils::check(PsiT(n).extent(1) == npol*NMO,
                 "PHMSD orbital reference PsiT_{} has {} columns, expected npol*NMO = {}", n, PsiT(n).extent(1), npol*NMO);
  }
  // the file stores beta occupations unshifted; downstream code expects them offset by NMO
  using nda::range;
  auto all = range::all;
  nda::array<int,2> occa, occb;
  nda::h5_read(grp,"occa",occa);
  nda::h5_read(grp,"occb",occb);
  utils::check(occa.extent(0) >= ndets && occb.extent(0) >= ndets, " occupation arrays too small.");
  utils::check_shape(occa(range(ndets), all), "occa", ndets, nup);
  utils::check_shape(occb(range(ndets), all), "occb", ndets, NEL - nup);
  // with a reference, the occupation numbers index its orbitals (the rows of PsiT_n)
  if(nreferences > 0) {
    for(int s = 0; s < 2; ++s) {
      auto const& occ = (s == 0 ? occa : occb);
      int const ref   = (s == 0 ? 0 : nreferences - 1);
      int const norb  = PsiT(ref).extent(0);
      for(int i = 0; i < ndets; ++i) {
        for(int k = 0; k < occ.extent(1); ++k) {
          utils::check(occ(i, k) < norb, "PHMSD {}({},{}) = {} is past the {} orbitals of the reference PsiT_{}.",
                       s == 0 ? "occa" : "occb", i, k, occ(i, k), norb, ref);
        }
      }
    }
  }
  occs.resize(ndets,NEL);
  occs(all, range(nup)) = occa(range(ndets), all);
  occs(all, range(nup, NEL)) = occb(range(ndets), all) + NMO;
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


int get_number_of_determinants(int ndets_in_file, int requested) {
  if(requested < 1) {
    return ndets_in_file;
  }
  if(requested > ndets_in_file) {
    app_warning("Found less determinants than requested, adjusting request: requested {} > {} in file.", requested, ndets_in_file);
    return ndets_in_file;
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
  auto const info = read_wavefunction_info(grp);
  ndets_to_read = get_number_of_determinants(info.ndets, ndets_to_read);
  app_log(1," - Number of determinants in trial wavefunction: {} ", ndets_to_read);
  ci.resize(ndets_to_read);
  nda::array<ComplexType,1> ci_t(info.ndets);
  utils::h5_read(grp,"ci_coeffs",ci_t);
  walker_type = info.walker_type;
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
