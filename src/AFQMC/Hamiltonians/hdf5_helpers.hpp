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

#pragma once

#include <iostream>

#include "config.h"
#include "AFQMC/config.h"
#include "IO/app_loggers.h"
#include "utilities/check.hpp"
#include "nda/h5.hpp"

namespace sfqmc
{
namespace afqmc
{

inline HamiltonianTypes peekHamType(h5::group grp, std::string format = "std")
{
  if (format  == "coqui") {
    // only format available, add choices as they are implemented
    // this is not enough, it could be THC, KPTHC, etc... Look for cholesky vectors...
    utils::check(grp.has_subgroup("/Interaction"), "Missing Interaction dataset.");
    h5::group igrp = grp.open_group("/Interaction");
    std::vector<int> shape;
    if (igrp.has_key("Vq0"))
      return KPFactorized;
    if (igrp.has_key("factorized_coulomb_matrix"))
    {
      auto l = h5::array_interface::get_dataset_info(igrp,"factorized_coulomb_matrix");
      utils::check(l.lengths[0]>0,"  Error: Found Interaction/factorized_coulomb_matrix with dimension=0 "); 
      return (l.lengths[0]==1?THC:KPTHC);
    }
  } else if(format == "std") {
    h5::group hgrp = grp.open_group("/Hamiltonian");
    if (hgrp.has_subgroup("KPFactorized"))
    {
      return KPFactorized;
    }
    if (hgrp.has_subgroup("DenseFactorized"))
    {
      return RealDenseFactorized;
    }
    if (hgrp.has_subgroup("ModelHamiltonian"))
    {
      return ModelHamiltonian;
    }
  } else {
    utils::check(false, "  Error: Invalid format in peekHamType. ");
    return UNKNOWN;
  }
  utils::check(false,"  Error: Invalid hdf5 file format in peekHamType(). ");
  return UNKNOWN;
}

inline std::string get_hamiltonian_format(h5::group& grp)
{
  if(grp.has_subgroup(std::string("/Hamiltonian")))
    return "std";
  else if(grp.has_subgroup(std::string("/System")) and grp.has_subgroup(std::string("/Interaction")))
    return "coqui"; 
  else
    utils::check(false,"Error in get_hamiltonian_format: Invalid format");
  return "";
}

// Reads the constant energy offset from an integral file: E_nuclear + E_frozen_core,
// plus the Madelung electron self-interaction for periodic (coqui) systems.
// nup/ndn are the trial-WF occupations of each spin block (ndn == 0 for CLOSED and
// NONCOLLINEAR), from which the total electron count is derived for the Madelung term.
// Must be called on the MPI root (the caller broadcasts the result).
inline ComplexType read_energy_offset(h5::group& grp, std::string const& format,
                                      WALKER_TYPES type, long nup, long ndn)
{
  ComplexType E0(0);
  if(format == "std") {
    h5::group hgrp = grp.open_group("Hamiltonian");
    nda::vector<RealType> energy_offsets;  // resized by the read; [nuclear, frozen_core]
    nda::h5_read(hgrp, "Energies", energy_offsets);
    E0 = nda::sum(energy_offsets);
  } else if(format == "coqui") {
    h5::group hgrp = grp.open_group("System");
    RealType nuc(0), fzc(0), madelung(0);
    if(H5Aexists(h5::hid_t(hgrp), "nuclear_energy")) {
      h5::h5_read_attribute(hgrp, "nuclear_energy", nuc);
    }
    if(H5Aexists(h5::hid_t(hgrp), "frozen_core_energy")) {
      h5::h5_read_attribute(hgrp, "frozen_core_energy", fzc);
    }
    if(H5Aexists(h5::hid_t(hgrp), "madelung_constant")) {
      h5::h5_read_attribute(hgrp, "madelung_constant", madelung);
      long nelec = (type == CLOSED) ? 2 * nup : nup + ndn;  // NONCOLLINEAR: ndn == 0
      madelung *= -1.0 * nelec;
    }
    app_log(2, "");
    app_log(2, " - Nuclear coulomb energy: {}", nuc);
    app_log(2, " - Frozen Core energy: {}", fzc);
    app_log(2, " - Electron self-interaction energy: {}", madelung);
    E0 = nuc + fzc + madelung;
  } else {
    utils::check(false, "Error in read_energy_offset: Invalid format: {}", format);
  }
  return E0;
}

inline std::tuple<int, int, int> read_info_from_wfn(std::string fileName, std::string type)
{
  app_log(1, "Reading info from wfn file: {} of type {} ", fileName, type);
  h5::file file(fileName,'r');
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
  h5::group mgrp = wgrp.open_group(type);
  std::vector<int> Idata(5);
  h5::h5_read(mgrp,"dims",Idata);
  return std::make_tuple(Idata[0], Idata[1], Idata[2]);
}

} // namespace afqmc

} // namespace sfqmc

