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

namespace sfqmc {
namespace utils
{

  template <typename T>
  struct remove_complex {typedef T type;};
  template <typename T>
  struct remove_complex<std::complex<T> > {typedef T type;};
    
  template<typename T>
  using remove_complex_t = typename remove_complex<T>::type;

  template <typename T>
  struct add_complex {typedef std::complex<T> type;};
  template <typename T>
  struct add_complex<std::complex<T> > {typedef std::complex<T> type;};
    
  template<typename T>
  using add_complex_t = typename add_complex<T>::type;

} 
}

