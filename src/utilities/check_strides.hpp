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

#include "utilities/check.hpp"
#include "nda/nda.hpp"

namespace sfqmc::utils {

template<nda::Array A>
void check_strides(A const& a)
{
  check(a.indexmap().min_stride() == 1, "Strides mismatch");
}

template<nda::Array A, typename... Args>
void check_strides(A const& a, Args... rest)
{ 
  check(a.indexmap().min_stride() == 1, "Strides mismatch");
  if constexpr (sizeof...(Args))
    check_strides(rest...);
}


} // namespace 
