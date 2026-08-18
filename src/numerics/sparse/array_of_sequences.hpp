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
  long size1_;
  long capacity_;
  larray<value_type> data_;
  ::nda::array<int_type, 1> row_begin_;
  ::nda::array<int_type, 1> row_end_;

  // set object to null state
  void reset()
  {
    size1_          = 0;
    capacity_       = 0;
    data_.resize(0); 
    row_begin_.resize(1);
    row_end_.resize(0);
  }

public:
  // erase!!!
  array_of_sequences() {};
  
  template<typename integer_type = long>
  array_of_sequences(long sz, integer_type nnzpr_unique) : size1_(sz), capacity_(sz*nnzpr_unique),
    data_(capacity_), row_begin_(size1_+1,0), row_end_(size1_,0)
  {
    if(nnzpr_unique == 0) return;
    for(long i=0; i<size1_; ++i) {
      row_begin_(i) = int_type(i*nnzpr_unique);
      row_end_(i) = int_type(i*nnzpr_unique);
    }
    row_begin_(size1_) = capacity_; 
  }

  template<typename integer_type = long>
  array_of_sequences(long sz, std::vector<integer_type> const& nnzpr) : size1_(sz), 
    capacity_(std::accumulate(nnzpr.begin(),nnzpr.begin()+sz,long(0))), 
    data_(capacity_), row_begin_(size1_+1,0), row_end_(size1_,0)
  {
    // at this point might be too late!!!
    sfqmc::utils::check(nnzpr.size() >= sz, "Size mismatch");
    if(capacity_ == 0) return;
    long i0=0;
    for(long i=0; i<size1_; ++i) {
      row_begin_(i) = i0;
      row_end_(i)   = i0;
      i0 += long(nnzpr[i]); 
    }
    row_begin_(size1_) = i0; 
    sfqmc::utils::check(i0 == capacity_, "Problems assembling array_of_sequences: i0:{}, capacity:{}",i0,capacity_);
  }

  void reserve(long nnzpr_unique)
  {
    if(size1_ == 0) return;
    int_type minN = int_type(row_begin_(1) - row_begin_(0));
    for (long i = 0; i < size1_; ++i)
      minN = std::min(minN, int_type(row_begin_(i+1) - row_begin_(i)));
    if (int_type(nnzpr_unique) <= minN)
      return;
    larray<value_type> new_(size1_*nnzpr_unique);
    for(long i = 0, i0=0; i < size1_; ++i, i0+=nnzpr_unique) {
      long n = this->num_elements(i);
      new_(nda::range(i0,i0+n)) = this->sequence(i);
    }
    capacity_ = size1_*nnzpr_unique;
    for(long i = 0; i < size1_; ++i) {
      row_begin_(i) = int_type(i*nnzpr_unique);
      row_end_(i) = int_type(i*nnzpr_unique);
    }
    row_begin_(size1_) = capacity_; 
    data_ = std::move(new_);
  }

  template<class Vec>
  void reserve(Vec const& nnzpr) {
    sfqmc::utils::check(nnzpr.size() >= size1_, "Size mismatch");
    if(size1_ == 0) return;
    bool skip = true;
    for (long i = 0; i < size1_; ++i) {
      skip = long(nnzpr(i)) <= long(row_begin_(i+1) - row_begin_(i)); 
      if(not skip) break;
    }
    if (skip) return;

    long cap_ = std::accumulate(nnzpr.begin(),nnzpr.begin()+size1_,long(0));
    larray<value_type> new_(cap_);
    for(long i = 0, i0=0; i < size1_; ++i) {
      long n = this->num_elements(i);
      new_(nda::range(i0,i0+n)) = this->sequence(i);
      i0 += long(nnzpr[i]);
    }
    capacity_ = cap_; 
    for(long i = 0, i0=0; i < size1_; ++i) {
      row_begin_(i) = int_type(i0);
      row_end_(i) = int_type(i0);
      i0 += long(nnzpr[i]);
    }
    row_begin_(size1_) = capacity_;
    data_ = std::move(new_);
  }

  template<typename val_t, MEMORY_SPACE mem_t, typename int_t,
          typename = std::enable_if_t<not (std::is_same_v<value_type,val_t> and
                                           mem_type == mem_t and
                                       std::is_same_v<int_type,int_t>) >>
  array_of_sequences(array_of_sequences<val_t,mem_t,int_t> const& other) :
        size1_(other.size()),
        capacity_(other.capacity()),
        data_(other.values()),
        row_begin_(other.sequences_begin()),
        row_end_(other.sequences_end())
  {}

  template<typename integer_type = long, typename Val_t> 
  void emplace_back(integer_type index, Val_t val )
  {
    sfqmc::utils::check(index >= 0 and index < size1_, "Out of bounds");
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
  auto size() const { return size1_; }
  auto capacity(long i) const
  {
    if (size1_==0) return long(0);
    return static_cast<long>(row_begin_(i + 1) - row_begin_(i));
  }
  auto capacity() const
  {
    if (size1_==0) return long(0);
    return capacity_;
  }
  auto num_elements() const
  {
    if (size1_==0) return long(0);
    long ret = 0;
    for (long i = 0; i != size1_; ++i)
      ret += static_cast<long>(row_end_(i) - row_begin_(i));
    return ret;
  }
  auto num_elements(long i) const
  {
    sfqmc::utils::check(i >= 0 && i < size1_, "Invalid index i:{}",i);
    return static_cast<long>(row_end_(i) - row_begin_(i));
  }
  auto values() const { return data_(); }
  auto values() { return data_(); }
  auto sequence(long i) const { 
    sfqmc::utils::check(size1_ > 0, "Empty structure.");
    sfqmc::utils::check(i >= 0 and i < size1_, "Out of bounds");
    return data_(nda::range(row_begin_(i),row_end_(i))); 
  }
  auto sequence(long i) { 
    sfqmc::utils::check(size1_ > 0, "Empty structure.");
    sfqmc::utils::check(i >= 0 and i < size1_, "Out of bounds");
    return data_(nda::range(row_begin_(i),row_end_(i))); 
  }
};

} // namespace sparse
} // namespace math

