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

/*
 * Implements a vector of sequences of diferent sizes.
 * Designed derived from ucsr_matrix. Essentually similar to ucsr_matrix, but
 * without a column index.
 */
#pragma once

#include <array>
#include <cassert>
#include <iostream>
#include <vector>
#include <numeric>
#include <memory>
#include <type_traits> // enable_if
#include <algorithm>
#include <utility>
#include <tuple>

#include "utilities/check.hpp"

namespace math
{
namespace sparse
{

template<class ValType, MEMORY_SPACE MEM = HOST_MEMORY, class IntType = int>
class array_of_sequences
{
private:
  template<typename T>
  using larray = memory::array<MEM, T, 1>;
  using range  = ::nda::range;

public:
  using value_type = ValType; 
  using int_type   = IntType; 
  static const MEMORY_SPACE mem_type = MEM;

protected:
  using this_t = array_of_sequences<ValType, MEM, IntType>;
  larray<value_type> data_;
  ::nda::array<int_type, 1> row_begin_;
  ::nda::array<int_type, 1> row_end_;

  // Checking here also lets the compiler see that the array extents in the constructors
  // below are non-negative; it warns about a negative allocation otherwise.
  static long checked_size(long sz) {
    sfqmc::utils::check(sz >= 0, "array_of_sequences: negative size: {}", sz);
    return sz;
  }

  // set object to null state
  void reset()
  {
    data_.resize(0); 
    row_begin_.resize(1);
    row_end_.resize(0);
  }

public:
  // erase!!!
  array_of_sequences() {};
  
  template<typename integer_type = long>
  array_of_sequences(long sz, integer_type nnzpr_unique) :
    data_(checked_size(sz)*nnzpr_unique), row_begin_(sz+1,0), row_end_(sz,0)
  {
    if(nnzpr_unique == 0) return;
    for(long i=0; i<sz; ++i) {
      row_begin_(i) = int_type(i*nnzpr_unique);
      row_end_(i) = int_type(i*nnzpr_unique);
    }
    row_begin_(sz) = capacity(); 
  }

  template<typename integer_type = long>
  array_of_sequences(long sz, std::vector<integer_type> const& nnzpr) :
    data_(std::accumulate(nnzpr.begin(),nnzpr.begin()+checked_size(sz),long(0))),
    row_begin_(sz+1,0), row_end_(sz,0)
  {
    // at this point might be too late!!!
    sfqmc::utils::check(nnzpr.size() >= sz, "Size mismatch");
    if(capacity() == 0) return;
    long i0=0;
    for(long i=0; i<sz; ++i) {
      row_begin_(i) = i0;
      row_end_(i)   = i0;
      i0 += long(nnzpr[i]); 
    }
    row_begin_(sz) = i0; 
    sfqmc::utils::check(i0 == capacity(), "Problems assembling array_of_sequences: i0:{}, capacity:{}",i0,capacity());
  }

  void reserve(long nnzpr_unique)
  {
    long sz = size();
    if(sz == 0) return;
    int_type minN = int_type(row_begin_(1) - row_begin_(0));
    for (long i = 0; i < sz; ++i)
      minN = std::min(minN, int_type(row_begin_(i+1) - row_begin_(i)));
    if (int_type(nnzpr_unique) <= minN)
      return;
    larray<value_type> new_(sz*nnzpr_unique);
    for(long i = 0, i0=0; i < sz; ++i, i0+=nnzpr_unique) {
      long n = this->num_elements(i);
      new_(nda::range(i0,i0+n)) = this->sequence(i);
    }
    for(long i = 0; i < sz; ++i) {
      row_begin_(i) = int_type(i*nnzpr_unique);
      row_end_(i) = int_type(i*nnzpr_unique);
    }
    data_ = std::move(new_);
    row_begin_(sz) = capacity(); 
  }

  template<class Vec>
  void reserve(Vec const& nnzpr) {
    long sz = size();
    sfqmc::utils::check(nnzpr.size() >= sz, "Size mismatch");
    if(sz == 0) return;
    bool skip = true;
    for (long i = 0; i < sz; ++i) {
      skip = long(nnzpr(i)) <= long(row_begin_(i+1) - row_begin_(i)); 
      if(not skip) break;
    }
    if (skip) return;

    long cap_ = std::accumulate(nnzpr.begin(),nnzpr.begin()+sz,long(0));
    larray<value_type> new_(cap_);
    for(long i = 0, i0=0; i < sz; ++i) {
      long n = this->num_elements(i);
      new_(nda::range(i0,i0+n)) = this->sequence(i);
      i0 += long(nnzpr[i]);
    }
    for(long i = 0, i0=0; i < sz; ++i) {
      row_begin_(i) = int_type(i0);
      row_end_(i) = int_type(i0);
      i0 += long(nnzpr[i]);
    }
    data_ = std::move(new_);
    row_begin_(sz) = capacity();
  }

  template<typename val_t, MEMORY_SPACE mem_t, typename int_t,
          typename = std::enable_if_t<not (std::is_same_v<value_type,val_t> and
                                           mem_type == mem_t and
                                       std::is_same_v<int_type,int_t>) >>
  array_of_sequences(array_of_sequences<val_t,mem_t,int_t> const& other) :
        data_(other.values()),
        row_begin_(other.sequences_begin()),
        row_end_(other.sequences_end())
  {}

  template<typename integer_type = long, typename Val_t> 
  void emplace_back(integer_type index, Val_t val )
  {
    sfqmc::utils::check(index >= 0 and index < size(), "Out of bounds");
    sfqmc::utils::check(row_end_[index] < row_begin_[index + 1], "row size exceeded the maximum");
    long p = long(row_end_[index]);
    // gpu safe
    data_(nda::range(p,p+1)) = value_type(val); 
    ++row_end_[index];
  }

  auto sequences_begin() const { return row_begin_(); }
  auto sequences_end() const { return row_end_(); }
  auto sequence_begin(long i = 0) const { return row_begin_(i); }
  auto sequence_end(long i = 0) const { return row_end_(i); }
  auto size() const { return row_end_.extent(0); }
  auto capacity(long i) const
  {
    if (size()==0) return long(0);
    return static_cast<long>(row_begin_(i + 1) - row_begin_(i));
  }
  auto capacity() const { return data_.extent(0); }
  auto num_elements() const
  {
    long ret = 0;
    for (long i = 0; i != size(); ++i)
      ret += static_cast<long>(row_end_(i) - row_begin_(i));
    return ret;
  }
  auto num_elements(long i) const
  {
    sfqmc::utils::check(i >= 0 && i < size(), "Invalid index i:{}",i);
    return static_cast<long>(row_end_(i) - row_begin_(i));
  }
  auto values() const { return data_(); }
  auto values() { return data_(); }
  auto sequence(long i) const { 
    sfqmc::utils::check(size() > 0, "Empty structure.");
    sfqmc::utils::check(i >= 0 and i < size(), "Out of bounds");
    return data_(nda::range(row_begin_(i),row_end_(i))); 
  }
  auto sequence(long i) { 
    sfqmc::utils::check(size() > 0, "Empty structure.");
    sfqmc::utils::check(i >= 0 and i < size(), "Out of bounds");
    return data_(nda::range(row_begin_(i),row_end_(i))); 
  }
};

} // namespace sparse
} // namespace math

