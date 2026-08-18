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

#ifndef AFQMC_TYPE_CONVERSION_HPP
#define AFQMC_TYPE_CONVERSION_HPP

#include <type_traits> // decay_t
#include <vector> 
#include <complex> 

namespace sfqmc
{
namespace afqmc
{
template<class MPtr>
using pointedType = std::decay_t<decltype(*std::declval<MPtr&>())>;

// convert to single precision
template<typename T>
struct to_single_precision
{
  using value_type = T;
  using type = T;
};

template<>
struct to_single_precision<double>
{
  using value_type = float;
  using type = float;
};

template<>
struct to_single_precision<std::complex<double>>
{
  using value_type = std::complex<float>;
  using type = std::complex<float>;
};

template<>
struct to_single_precision<long>
{
  using value_type = int;
  using type = int;
};

template<>
struct to_single_precision<unsigned long>
{
  using value_type = unsigned int;
  using type = unsigned int;
};

// convert to double precision
template<typename T>
struct to_double_precision
{
  using value_type = T;
  using type = T;
};

template<>
struct to_double_precision<float>
{
  using value_type = double;
  using type = double;
};

template<>
struct to_double_precision<std::complex<float>>
{
  using value_type = std::complex<double>;
  using type = std::complex<double>;
};

template<>
struct to_double_precision<int>
{
  using value_type = long;
  using type = long;
};

template<>
struct to_double_precision<unsigned int>
{
  using value_type = unsigned long;
  using type = unsigned long;
};


// choose working precision
template <bool SP, typename T>
struct to_working_precision
{
  typedef std::conditional_t<SP, 	
                             typename to_single_precision<T>::value_type, 
			     T> type;
};


// remove complex
template<typename T>
struct remove_complex
{
  using value_type = T;
  using type = T;
};

template<>
struct remove_complex<std::complex<float>>
{
  using value_type = float;
  using type = float;
};

template<>
struct remove_complex<std::complex<double>>
{
  using value_type = double;
  using type = double;
};

// to_complex
template<typename T>
struct to_complex
{
  using type = T;
};

template<>
struct to_complex<float>
{
  using type = std::complex<float>;
};

template<>
struct to_complex<double>
{
  using type = std::complex<double>;
};

// convert a set of types into strings
template<class T>
inline std::string type_to_string()
{
  return std::string("");
}

template<>
inline std::string type_to_string<double>()
{
  return std::string("double");
}

template<>
inline std::string type_to_string<int>()
{
  return std::string("int");
}

template<>
inline std::string type_to_string<std::string>()
{
  return std::string("std::string");
}

// create std::vector<T> from std::vector<Q>
template<class Q, class T>
std::vector<Q> make_vector(std::vector<T> const& other)
{
  std::vector<Q> res;
  res.reserve(other.size());
  for (auto& v : other)
    res.emplace_back(v);
  return res;
}

template<class Q, class T, class Aux>
std::vector<Q> make_vector(std::vector<T> const& other, Aux param)
{
  std::vector<Q> res;
  res.reserve(other.size());
  for (auto& v : other)
    res.emplace_back(v, param);
  return res;
}

//   can use: return std::vector<Q>{std::make_move_iterator(other.begin()),
//                                  std::make_move_iterator(other.end()));
// and can replace move_vector with this outside
template<class Q, class T>
std::vector<Q> move_vector(std::vector<T>&& other)
{
  std::vector<Q> res;
  res.reserve(other.size());
  for (auto& v : other)
    res.emplace_back(std::move(v));
  return res;
}

template<class Q, class T, class Aux>
std::vector<Q> move_vector(std::vector<T>&& other, Aux param)
{
  std::vector<Q> res;
  res.reserve(other.size());
  for (auto& v : other)
    res.emplace_back(std::move(v), param);
  return res;
}
/*
// to bypass the lack of copy constructor of multi::array_refs/basic_arrays
template<class VType, class MType,
         typename = typename std::enable_if_t<MType::dimensionality == 2>
        >
void emplace_back_array_ref(VType& V, MType&& M, bool device=true) {
  // only makes sense for contiguous arrays
  RUNTIME_CHECK(M.stride(0) == M.size(1), "");
  RUNTIME_CHECK(M.stride(1) == 1, "");
  if(device) {  
    V.emplace_back(make_device_ptr(M.origin()),iextensions<2u>{M.size(0),M.size(1)});
  } else {
    V.emplace_back(raw_pointer_cast(M.origin()),iextensions<2u>{M.size(0),M.size(1)});
  }
} 
*/
} // namespace afqmc

} // namespace sfqmc

#endif
