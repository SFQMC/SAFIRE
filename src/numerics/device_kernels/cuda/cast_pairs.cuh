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

// The (source, destination) value type pairs copy() and accumulate() are instantiated for. Shared by
// copy.cu and accumulate.cu so the two cannot drift: a pair one has and the other does not shows up
// as an undefined symbol at the call site rather than as a compile error here.

#include <complex>

// The pairs are ordered -- the destination is constructed from the source, so double ->
// complex<double> is instantiated and the reverse is not. The first two are the same-type diagonal,
// where that construction is a copy. Cross-type pairs are added on demand; nothing but a caller
// asking for one motivates an entry.
#define _for_each_pair_(M)                                                                         \
  M(double, double)                                                                                \
  M(std::complex<double>, std::complex<double>)                                                    \
  M(double, float)                                                                                 \
  M(float, double)                                                                                 \
  M(std::complex<double>, std::complex<float>)                                                     \
  M(std::complex<float>, std::complex<double>)                                                     \
  M(double, std::complex<double>)
