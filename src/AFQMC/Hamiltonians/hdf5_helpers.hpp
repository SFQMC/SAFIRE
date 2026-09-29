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
#include "AFQMC/Utilities/format_version.hpp"
#include "IO/app_loggers.h"
#include "utilities/check.hpp"
#include "nda/h5.hpp"

namespace sfqmc
{
namespace afqmc
{

inline HamiltonianType peekHamType(h5::group grp, std::string format = "std")
{
  if (format  == "coqui") {
    // only format available, add choices as they are implemented
    // this is not enough, it could be THC, KPTHC, etc... Look for cholesky vectors...
    utils::check(grp.has_subgroup("/Interaction"), "Missing Interaction dataset.");
    h5::group igrp = grp.open_group("/Interaction");
    std::vector<int> shape;
    if (igrp.has_key("Vq0"))
      return HamiltonianType::kp_factorized;
    if (igrp.has_key("factorized_coulomb_matrix"))
    {
      auto l = h5::array_interface::get_dataset_info(igrp,"factorized_coulomb_matrix");
      utils::check(l.lengths[0]>0,"  Error: Found Interaction/factorized_coulomb_matrix with dimension=0 ");
      return (l.lengths[0]==1 ? HamiltonianType::thc : HamiltonianType::kpthc);
    }
  } else if(format == "std") {
    h5::group hgrp = grp.open_group("/Hamiltonian");
    if (hgrp.has_subgroup("KPFactorized"))
    {
      return HamiltonianType::kp_factorized;
    }
    if (hgrp.has_subgroup("DenseFactorized"))
    {
      return HamiltonianType::real_dense_factorized;
    }
    if (hgrp.has_subgroup("ModelHamiltonian"))
    {
      return HamiltonianType::model_hamiltonian;
    }
  } else {
    APP_ABORT("  Error: Invalid format in peekHamType. ");
  }
  APP_ABORT("  Error: Invalid hdf5 file format in peekHamType(). ");
}

inline std::string get_hamiltonian_format(h5::group& grp) {
  if(grp.has_subgroup(std::string("/Hamiltonian"))) {
    check_format_version(grp.open_group("Hamiltonian"), "Hamiltonian");
    return "std";
  } else if(grp.has_subgroup(std::string("/System")) && grp.has_subgroup(std::string("/Interaction"))) {
    return "coqui";
  }
  utils::check(false, "Error in get_hamiltonian_format: Invalid format");
  return "";
}

// Reads the constant energy offset from an integral file: E_nuclear + E_frozen_core, plus the
// Madelung electron self-interaction where the file records a Madelung constant (CoQuí's periodic
// systems). All three are optional attributes, named as CoQuí names them, of the `Hamiltonian`
// group in the std format and of the `System` group in the coqui one.
// nup/ndn are the trial-WF occupations of each spin block (ndn == 0 for CLOSED and
// NONCOLLINEAR), from which the total electron count is derived for the Madelung term.
// Must be called on the MPI root (the caller broadcasts the result).
inline RealType read_energy_offset(h5::group& grp, std::string const& format,
                                   WALKER_TYPES type, long nup, long ndn) {
  utils::check(format == "std" || format == "coqui", "Error in read_energy_offset: Invalid format: {}", format);
  h5::group hgrp = grp.open_group(format == "std" ? "Hamiltonian" : "System");

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
  return nuc + fzc + madelung;
}

} // namespace afqmc

} // namespace sfqmc

