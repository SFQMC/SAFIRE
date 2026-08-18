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

#include <complex>
#include "configuration.hpp"
#include "nda/nda.hpp"
#include "utilities/check.hpp"
#include "numerics/operations/tensor.hpp"

namespace math
{

/*
 *   A(b,i,i) += alpha
 *
 * The diagonal of a batch of square matrices is a strided rank-2 view, which is what add_scalar
 * takes; beta = 1 makes it an accumulation rather than a fill.
 */
template<typename T, nda::MemoryArrayOfRank<3> A_t>
requires(std::decay_t<A_t>::is_stride_order_C())
void add_diagonal(T alpha, A_t && A)
{
  using V = std::remove_const_t<nda::get_value_t<A_t>>;
  sfqmc::utils::check(A.extent(1) == A.extent(2), "Size mismatch");
  add_scalar(V(alpha), 1.0, memory::diagonal_view(A));
}

}
