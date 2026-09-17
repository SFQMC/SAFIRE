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

#include "configuration.hpp"
#include <typeinfo>

#include <hdf5.h>
#include <hdf5_hl.h>

#include "h5/h5.hpp"
#include "nda/h5.hpp"
#include "nda/nda.hpp"

#include "utilities/check.hpp"

namespace sfqmc::utils {

namespace detail {

template<typename T, bool is_complex = false, nda::MemoryArray A_t>
void read_cast(h5::group& g, std::string name, A_t && A)
{
  using value_t = nda::get_value_t<A_t>;
  if constexpr (is_complex) {
    nda::array<std::complex<T>,nda::get_rank<A_t>> B(A.shape());
    nda::h5_read(g,name,B);
    if constexpr (std::is_assignable_v<value_t&,std::complex<T>>) 
      A() = B();
    else
      check(false,"Error in utils::h5_read_with_cast: Failed is_assignable_v");
  } else {
    nda::array<T,nda::get_rank<A_t>> B(A.shape());
    nda::h5_read(g,name,B);
    if constexpr (std::is_assignable_v<value_t&,T>) 
      A() = B();
    else
      check(false,"Error in utils::h5_read_with_cast: Failed is_assignable_v");
  }
}

}

/// @brief Whether a dataset stores complex values as the `{r, i}` compound datatype, as
/// written by h5py and Julia's HDF5.jl. Such a dataset carries no `__complex__` attribute
/// and has the same rank as the values it holds.
inline bool dataset_is_compound_complex(h5::array_interface::dataset_info const& l) {
  return h5::hdf5_type_equal(l.ty, h5::hdf5_type<h5::dcplx_t>());
}

/// @brief Whether a dataset holds complex values: either the `__complex__` attribute nda
/// writes, or the `{r, i}` compound datatype.
inline bool dataset_is_complex(h5::array_interface::dataset_info const& l) {
  return l.has_complex_attribute || dataset_is_compound_complex(l);
}

// reads and casts if types don't match
auto h5_read_with_cast(h5::group& g, std::string name, nda::MemoryArray auto && A)
{
  using A_t = std::decay_t<decltype(A)>; 
  using T = nda::get_value_t<A_t>;
  T x = {};
  if constexpr (nda::mem::on_host<A_t>) {
    auto l = h5::array_interface::get_dataset_info(g,name); 
    if (l.has_complex_attribute) {
      using T_real = nda::remove_complex_t<T>; 
      if (H5Tequal(h5::detail::hid_t_of<T_real>(),l.ty)>0) {
        nda::h5_read(g,name,A);
      } else {
        // find a better way. Tedious, so limiting to a few types
        if (H5Tequal(h5::detail::hid_t_of<double>(),l.ty)>0) {
          detail::read_cast<double,true>(g,name,A);
        } else if (H5Tequal(h5::detail::hid_t_of<float>(),l.ty)>0) {
          detail::read_cast<float,true>(g,name,A);
        } else {
          utils::check(false, "Problems with sfqmc::utils::h5_read_with_cast: Missing specialized complex type read:{}",typeid(x).name());
        }
      }  
    } else { 
      if (H5Tequal(h5::detail::hid_t_of<T>(),l.ty)) {
        nda::h5_read(g,name,A);
      } else {
        // find a better way. Tedious, so limiting to a few types
        if (H5Tequal(h5::detail::hid_t_of<int>(),l.ty)>0) {
          detail::read_cast<int>(g,name,A);
        } else if (H5Tequal(h5::detail::hid_t_of<long>(),l.ty)>0) {
          detail::read_cast<long>(g,name,A);
        } else if (H5Tequal(h5::detail::hid_t_of<double>(),l.ty)>0) {
          detail::read_cast<double>(g,name,A);
        } else if (H5Tequal(h5::detail::hid_t_of<float>(),l.ty)>0) {
          detail::read_cast<float>(g,name,A);
        } else {
          utils::check(false, "Problems with sfqmc::utils::h5_read_with_cast: Missing specialized type read:{}",typeid(x).name());
        }  
      } // l.has_complex_attribute
    }
  } else {
    // need to stage copy on host
    auto B = nda::to_host(A);
    read(g,name,B);
    A() = B();
  }
}

/// @brief Read an HDF5 dataset into an nda MemoryArray.
///
/// This wrapper fixes a missing device guard and handling of `{r, i}` compound complex arrays,
/// which are broken upstream.
///
/// @param g    HDF5 group containing the dataset.
/// @param name Name of the dataset within @p g.
/// @param A    Destination array or array-view; must satisfy `nda::MemoryArray`.
auto h5_read(h5::group& g, std::string name, nda::MemoryArray auto && A)
{
  using A_t = std::decay_t<decltype(A)>;
  using T = nda::get_value_t<A_t>;
  if constexpr (nda::mem::on_host<A_t>) {
    if constexpr (nda::is_complex_v<T>) {
      auto l = h5::array_interface::get_dataset_info(g,name);
      if(dataset_is_compound_complex(l)) {
        auto v = nda::detail::prepare_h5_array_view(A);
        h5::array_interface::read(g,name,v);
      } else {
        nda::h5_read(g,name,A);
      }
    } else {
      nda::h5_read(g,name,A);
    }
  } else {
    // need to stage copy on host
    auto B = nda::to_host(A);
    read(g,name,B);
    A() = B();
  }
}

inline h5::group h5_open_or_create(h5::group& group, std::string const& key) {
  if(group.has_key(key)) {
    return group.open_group(key);
  }
  return group.create_group(key);
}

} // namespace sfqmc::utils
