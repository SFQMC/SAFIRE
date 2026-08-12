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
////////////////////////////////////////////////////////////////////////////////

#pragma once

#include <cmath>
#include <complex>
#include <algorithm>
#include "configuration.hpp"
#include "IO/app_loggers.h"
#include "IO/banner.hpp"
#include "utilities/check.hpp"

#include "nda/nda.hpp"
#include "numerics/device_kernels/kernels.h"

namespace math
{

template<nda::MemoryMatrix A_t> 
auto exp_hermitian(A_t const& A, bool printeV = false)
{
  using Array_t = typename std::decay_t<A_t>::regular_type;
  using Type     = nda::get_value_t<A_t>;
  sfqmc::utils::check(A.extent(0) == A.extent(1), "Shape mismatch");
  long N = A.extent(0);

  // do in host for now
  nda::matrix<Type> X(A);

  // A = M*V*dagger(M)
  // careful, M is in Fortran layout
  auto [V,M] = nda::linalg::eigh(X);

  // exp(A) = M*exp(V)*dagger(M)
  if (printeV)
  {
    sfqmc::app_log(1, sfqmc::section("Eigenvalues of exponentiated matrix"));
    sfqmc::app_log(1, " i    eigV[i]    exp(eigV[i]) ");
    for (int j = 0; j < N; j++)
      sfqmc::app_log(1, " {}  {}  {} ", j, V(j), std::exp(V(j)));
    sfqmc::app_log(1, sfqmc::hrule());
    sfqmc::app_log(1, "\n\n");
  } 
  for (int j = 0; j < N; j++)
    M(nda::range::all,j) *= std::sqrt(std::exp(V(j)));
  // M is in Fortran layout, so dagger(M) wraps a C-layout view. BLAS can only fold a lazy
  // conjugate into a 'C' operation when the result has the layout opposite to that view.
  nda::matrix<Type,nda::F_layout> R(N,N);
  nda::blas::gemm(M,nda::dagger(M),R);

  return Array_t(R);
}

} // namespace math
