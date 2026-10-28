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

#include <string_view>

#include <hdf5.h>

#include "h5/h5.hpp"
#include "utilities/check.hpp"

namespace sfqmc::afqmc {

// Version of the Hamiltonian and Wavefunction layouts safiretools writes, and of the results.h5
// this executable writes, stored as the `format_version` attribute of the Hamiltonian and
// Wavefunction/NOMSD|PHMSD groups and of the results.h5 root. Bumped with every incompatible change, in step with
// FORMAT_VERSION in utils/safiretools/hdf5.py. CoQuí files carry none.
inline constexpr int FORMAT_VERSION = 1;

inline void write_format_version(h5::group grp) {
  h5::h5_write_attribute(grp, "format_version", FORMAT_VERSION);
}

inline bool has_format_version(h5::group grp) {
  return H5Aexists(h5::hid_t(grp), "format_version") > 0;
}

// Aborts unless `grp` carries the current FORMAT_VERSION; `what` names the group in the message.
inline void check_format_version(h5::group grp, std::string_view what) {
  utils::check(has_format_version(grp),
               "{} has no format_version attribute: it was written before format version {}. "
               "Regenerate it with safiretools.", what, FORMAT_VERSION);
  int version = 0;
  h5::h5_read_attribute(grp, "format_version", version);
  utils::check(version == FORMAT_VERSION,
               "{} has format_version {}, but this executable reads version {}. "
               "Regenerate it with safiretools.", what, version, FORMAT_VERSION);
}

} // namespace sfqmc::afqmc
