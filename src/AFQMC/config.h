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

#include <string>
#include <algorithm>
#include <cstdlib>
#include <ctype.h>
#include <vector>
#include <map>
#include <complex>
#include <tuple>
#include <fstream>

#include "configuration.hpp"
#include "config.0.h"
#include "AFQMC/propagator_types.hpp"
#include "IO/AppAbort.hpp"
#include "utilities/check.hpp"

#include "numerics/sparse/sparse.hpp"

namespace sfqmc
{
namespace afqmc
{

// Global Constant Definitions
const int DEFAULT_MEASURE_INTERVAL_MULTIPLIER = 1; // in units of population control interval
const int DEFAULT_POPULATION_CONTROL_INTERVAL = 10; // in units of steps
const int DEFAULT_WALKER_ORTHO_INTERVAL = 10; // in units of steps
const float DEFAULT_TIME_STEP = 0.01f; // in units of inverse energy (depending on Hamiltonian units)

enum WALKER_TYPES
{
  UNDEFINED_WALKER_TYPE,
  CLOSED,
  COLLINEAR,
  NONCOLLINEAR
};

inline WALKER_TYPES initWALKER_TYPES(int i)
{
  if (i == 0)
    return UNDEFINED_WALKER_TYPE;
  else if (i == 1)
    return CLOSED;
  else if (i == 2)
    return COLLINEAR;
  else if (i == 3)
    return NONCOLLINEAR;
  else if (i == 4)
    utils::check(false, "This wavefunction was generated with the removed FULLYPOLARIZED "
                        "walker type (dims[3]==4). Regenerate it as COLLINEAR (dims[3]==2) "
                        "with ndown=0.");
  return UNDEFINED_WALKER_TYPE;
}

inline auto walkerTypeToDims(WALKER_TYPES type) {
  int nspin = type == COLLINEAR ? 2 : 1;
  int npol = type == NONCOLLINEAR ? 2 : 1;
  return std::make_tuple(nspin, npol);
}

inline WALKER_TYPES walkerTypeFromDims(int nspin, int npol) {
  if(nspin == 1 && npol == 1) {
    return CLOSED;
  }
  if(nspin > 1 && npol == 1) {
    return COLLINEAR;
  }
  if(nspin == 1 && npol > 1) {
    return NONCOLLINEAR;
  }
  utils::check(false, "There is no walker type that has nspin = {}, npol = {}", nspin, npol); 
  return UNDEFINED_WALKER_TYPE;
}

inline bool walkerTypeIsConvertible(WALKER_TYPES from, WALKER_TYPES to) {
  if(from == to) {
    return true;
  }
  if(from < CLOSED || from > NONCOLLINEAR || to < CLOSED || to > NONCOLLINEAR) {
    return false;
  }
  return from <= to;
}

// Like walkerTypeIsConvertible but also makes sure the values of nspin and npol are compatible
// Note that although it is called “walker” it refers to the spin dimensions of any tensor in the code.
inline bool walkerDimsAreConvertible(int nspin_from, int npol_from, int nspin_to, int npol_to) {
  utils::check(std::min({nspin_from, npol_from, nspin_to, npol_to}) > 0 &&
                   // TODO: remove once we do not assume 2 flavors anymore
                   std::max({nspin_from, npol_from, nspin_to, npol_to}) <= 2,
               "for now we assume that 0 < nspin, npol <= 2");

  // CLOSED -> X
  if(nspin_from == 1 && npol_from == 1) {
    return true;
  }

  // X -> X
  if(nspin_from == nspin_to && npol_from == npol_to) {
    return true;
  }

  // COLLINEAR -> NONCOLLINEAR
  return npol_from == 1 && nspin_from == npol_to && nspin_to == 1;
}


inline std::string walkerTypeToString(WALKER_TYPES type)
{
  if (type == UNDEFINED_WALKER_TYPE) return "undefined";
  else if (type == CLOSED) return "closed";
  else if (type == COLLINEAR) return "collinear";
  else if (type == NONCOLLINEAR) return "noncollinear";
  utils::check(false, "unknown walker type: {}", type);
  return "unknown";
}

enum INTEGRAL_TYPES
{
  UNDEFINED_INTEGRAL_TYPE,
  CLOSED_INTEGRAL,
  COLLINEAR_INTEGRAL,
  NONCOLLINEAR_INTEGRAL,
};

enum WAVEFUNCTION_TYPES
{
  NOMSD_WFN,
  PHMSD_WFN
};

inline INTEGRAL_TYPES initINTEGRAL_TYPES(int i)
{
  if (i == 0)
    return UNDEFINED_INTEGRAL_TYPE;
  else if (i == 1)
    return CLOSED_INTEGRAL;
  else if (i == 2)
    return COLLINEAR_INTEGRAL;
  else if (i == 3)
    return NONCOLLINEAR_INTEGRAL;
  return UNDEFINED_INTEGRAL_TYPE;
}

enum SpinTypes
{
  Alpha,
  Beta
};

// Trial wave function types
template<typename T, MEMORY_SPACE MEM>
using PsiT_Matrix_t = math::sparse::csr_matrix<T, MEM, int, int>;
template<MEMORY_SPACE MEM>
using PsiT_Matrix   = PsiT_Matrix_t<ComplexType,MEM>;

enum HamiltonianTypes
{
  THC,
  KPTHC,
  KPFactorized,
  RealDenseFactorized,
  ModelHamiltonian,  
  UNKNOWN
};

} // namespace afqmc
} // namespace sfqmc

