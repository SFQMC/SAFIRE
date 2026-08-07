/**
 * ==========================================================================
 * CoQuí: Correlated Quantum ínterface
 *
 * Copyright (c) 2022-2025 Simons Foundation & The CoQuí developer team
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 * 
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * ==========================================================================
 */

#pragma once

#include <array>
#include <cassert>
#include <iostream>
#include <vector>
#include <tuple>
#include <numeric>
#include <memory>
#include <type_traits> 
#include <algorithm>
#include <utility>

#include "configuration.hpp"
#include "utilities/check.hpp"
#include "IO/AppAbort.hpp"
#include "IO/app_loggers.h"
#include "utilities/pair_iterator.hpp"

#include "nda/nda.hpp"

#include "numerics/sparse/ucsr_matrix.hpp"
#include "numerics/sparse/csr_matrix_view.hpp"

namespace math
{
namespace sparse
{

template<typename ValType, MEMORY_SPACE MEM = HOST_MEMORY, typename IndxType = int, typename IntType = long>
class csr_matrix : public ucsr_matrix<ValType, MEM, IndxType, IntType> 
{
  using base        = ucsr_matrix<ValType, MEM, IndxType, IntType>; 
  using base::size1_;
  using base::size2_;
  using base::data_;
  using base::jdata_;
  using base::row_begin_;
  using base::row_end_;

  // orders the column indices within each row, carrying the values along
  template<typename JArray, typename VArray>
  void sort_rows_(JArray& jdata_h, VArray& data_h) const
  {
    using sfqmc::utils::make_paired_iterator;
    for(long p = 0; p < size1_; p++) {
      auto jrow = jdata_h(::nda::range(row_begin_(p), row_end_(p)));
      auto vrow = data_h(::nda::range(row_begin_(p), row_end_(p)));
      std::sort(make_paired_iterator(jrow.begin(), vrow.begin()),
                make_paired_iterator(jrow.end(), vrow.end()),
                [](auto const& a, auto const& b) { return a.first < b.first; });
    }
  }

protected:
  // device copies, only populated if MEM==DEVICE_MEMORY
  memory::array<MEM, IntType, 1> row_begin_dev_ = memory::array<MEM, IntType, 1>(0);
  memory::array<MEM, IntType, 1> row_end_dev_ = memory::array<MEM, IntType, 1>(0);

public:
  using base::reserve;
  using base::shape;
  using base::capacity;
  using base::values;
  using base::columns;
  using base::row_begin;
  using base::row_end;
  using base::nnz;
  using base::compact;
  using base::clear;
  using base::serialize;
  using base::deserialize;
  using base::size_of_serialized_in_bytes;

  using regular_type = csr_matrix<ValType, MEM, IndxType, IntType>;

  // to be able to reuse ops_tags 
  using Array_t = memory::array<MEM, ValType, 2>; 
  static constexpr bool is_stride_order_Fortran() noexcept
    { return false; } 
  static constexpr bool is_stride_order_C() noexcept
    { return true; } 

  using value_type  = base::value_type; 
  using element     = base::element;     
  using element_ptr = base::element_ptr; 
  using index_type  = base::index_type;  
  using int_type    = base::int_type;    

  static constexpr bool sparse        = true;
  static constexpr int dimensionality = 2;
  static constexpr int rank           = 2;
  static constexpr bool sorted        = true;
  static constexpr MEMORY_SPACE mem_type = MEM;

  csr_matrix() {}

  template<typename IType = long, typename integer_type = long>
  requires ( std::is_integral_v<IType> and std::is_integral_v<integer_type> )
  csr_matrix(std::tuple<IType,IType> const& dims, integer_type nnzpr = 0) 
      : base(dims, nnzpr)
  {
    if constexpr (MEM==DEVICE_MEMORY) {
      row_begin_dev_.resize(size1_+1);
      row_end_dev_.resize(size1_);
      row_begin_dev_() = row_begin_(); 
      row_end_dev_() = row_end_(); 
    }
  }
  template<typename IType = long>
  requires ( std::is_integral_v<IType> )
  csr_matrix(std::tuple<IType, IType> const& dims, nda::MemoryVector auto && nnzpr)
      : base(dims, nnzpr)
  {
    if constexpr (MEM==DEVICE_MEMORY) {
      row_begin_dev_.resize(size1_+1);
      row_end_dev_.resize(size1_);
      row_begin_dev_() = row_begin_();
      row_end_dev_() = row_end_();    
    }
  }

  template<typename val_t, MEMORY_SPACE mem_t, typename indx_t, typename int_t,
          typename = std::enable_if_t<not (std::is_same_v<value_type,val_t> and
                                           mem_type == mem_t and
                                           std::is_same_v<index_type,indx_t> and
                                           std::is_same_v<int_type,int_t>) >>
  csr_matrix(csr_matrix<val_t,mem_t,indx_t,int_t> const& other) 
  {
    size1_ = other.shape(0);
    size2_ = other.shape(1);
    data_ = other.values();
    jdata_ = other.columns();
    row_begin_ = other.row_begin();
    row_end_ = other.row_end();
    if constexpr (MEM==DEVICE_MEMORY) {
      row_begin_dev_ = row_begin_();
      row_end_dev_ = row_end_();
    }
  }

  template<typename val_t, MEMORY_SPACE mem_t, typename indx_t, typename int_t>
  csr_matrix(ucsr_matrix<val_t, mem_t, indx_t, int_t> const& other) : base()
  {
    *this = other;
  }
  template<typename val_t, MEMORY_SPACE mem_t, typename indx_t, typename int_t>
  csr_matrix(ucsr_matrix<val_t, mem_t, indx_t, int_t> && other) : base()
  {
    *this = std::move(other);
  }
 

  template<typename val_t, MEMORY_SPACE mem_t, typename indx_t, typename int_t>
  csr_matrix& operator=(ucsr_matrix<val_t, mem_t, indx_t, int_t> const& other)
  {
    auto shape_ = other.shape();
    size1_      = shape_[0];
    size2_      = shape_[1];
    data_       = other.values();
    jdata_      = other.columns();
    row_begin_  = other.row_begin();
    row_end_    = other.row_end();
    if (size1_ == 0 || data_.size() == 0)
      return *this;

    auto data_h = memory::to_memory_space<HOST_MEMORY>(data_());
    auto jdata_h = memory::to_memory_space<HOST_MEMORY>(jdata_());
    sort_rows_(jdata_h, data_h);
    data_() = data_h();
    jdata_() = jdata_h();
    if constexpr (MEM==DEVICE_MEMORY) {
      row_begin_dev_ = row_begin_();
      row_end_dev_ = row_end_();
    }
    return *this;
  }

  template<typename val_t, MEMORY_SPACE mem_t, typename indx_t, typename int_t>
  csr_matrix& operator=(ucsr_matrix<val_t, mem_t, indx_t, int_t> && other)
  { 
    auto shape_ = other.shape();
    size1_      = shape_[0];
    size2_      = shape_[1];
    data_       = std::move(other.values());
    jdata_      = std::move(other.columns());
    row_begin_  = std::move(other.row_begin());
    row_end_    = std::move(other.row_end());
    if (size1_ == 0 || data_.size() == 0)
      return *this;

    auto data_h = memory::to_memory_space<HOST_MEMORY>(data_());
    auto jdata_h = memory::to_memory_space<HOST_MEMORY>(jdata_());
    sort_rows_(jdata_h, data_h);
    data_() = data_h();
    jdata_() = jdata_h();
    if constexpr (MEM==DEVICE_MEMORY) {
      row_begin_dev_ = row_begin_();
      row_end_dev_ = row_end_();
    }
    return *this;
  }

  template<class Pair = std::array<IndxType, 2>>
  value_type& get(Pair&& indices)
  {
    using std::get;
    assert(get<0>(indices) >= 0);
    assert(get<0>(indices) < size1_);
    // need a kernel for this!
    if constexpr (mem_type == DEVICE_MEMORY)
      sfqmc::utils::check(false,"Finish csr_matrix::emplace()");

    auto loc   = std::lower_bound(jdata_.data() + row_begin_(get<0>(indices)), 
                                  jdata_.data() + row_end_(get<0>(indices)), get<1>(indices));
    long disp  = std::distance(jdata_.data() + row_begin_(get<0>(indices)), loc);  
    long disp_ = std::distance(loc, jdata_.data() + row_end_(get<0>(indices)));
    
    if ( not (disp_ > 0 and *loc == get<1>(indices)) )
    { 
      sfqmc::utils::check(row_end_(get<0>(indices)) < row_begin_(get<0>(indices) + 1), 
			"row size exceeded the maximum");
      // new value, shift back and add in correct place
      if (disp_ > 0)
      { 
        // check if row is full
        std::move_backward(data_.data() + row_begin_(get<0>(indices)) + disp,
                           data_.data() + row_end_(get<0>(indices)), 
                           data_.data() + row_end_(get<0>(indices)) + 1);
        std::move_backward(jdata_.data() + row_begin_(get<0>(indices)) + disp,
                           jdata_.data() + row_end_(get<0>(indices)), 
                           jdata_.data() + row_end_(get<0>(indices)) + 1);
      }
      ++row_end_(get<0>(indices));
      jdata_(row_begin_(get<0>(indices)) + disp) = index_type(get<1>(indices)); 
      data_(row_begin_(get<0>(indices)) + disp) = value_type(0);
    }
    return data_(row_begin_(get<0>(indices)) + disp);
  }

  template<class Pair = std::array<IndxType, 2>>
  value_type get_value(Pair&& indices) const
  {
    using std::get;
    assert(get<0>(indices) >= 0);
    assert(get<0>(indices) < size1_);
    // need a kernel for this!
    if constexpr (mem_type == DEVICE_MEMORY)
      sfqmc::utils::check(false,"Finish csr_matrix::emplace()");

    auto loc   = std::lower_bound(jdata_.data() + row_begin_(get<0>(indices)),
                                  jdata_.data() + row_end_(get<0>(indices)), get<1>(indices));
    long disp  = std::distance(jdata_.data() + row_begin_(get<0>(indices)), loc);
    long disp_ = std::distance(loc, jdata_.data() + row_end_(get<0>(indices)));

    if ( not (disp_ > 0 and *loc == get<1>(indices)) )
    {
      sfqmc::utils::check(row_end_(get<0>(indices)) < row_begin_(get<0>(indices) + 1),
                        "row size exceeded the maximum");
      return value_type(0);
    }
    return data_(row_begin_(get<0>(indices)) + disp);
  }

public:
  template<class Pair = std::array<IndxType, 2>, class... Args>
  void emplace(Pair&& indices, Args&&... args)
  {
    get(indices) = value_type(std::forward<Args>(args)...);
  }

  // new column index must be larger than all previous column indexes in the row
  // otherwise throws
  template<class Pair = std::array<IndxType, 2>, class... Args>
  void emplace_back(Pair&& indices, Args&&... args)
  {
    using std::get;
    assert(get<0>(indices) >= 0);
    assert(get<0>(indices) < size1_);
    if constexpr (mem_type == DEVICE_MEMORY)
      sfqmc::utils::check(false,"Finish csr_matrix::emplace()");

    // check that row is not full
    sfqmc::utils::check(row_end_(get<0>(indices)) < row_begin_(get<0>(indices) + 1), 
			"row size exceeded the maximum");
    // check that emplaced column belongs in the end of the row 
    sfqmc::utils::check(row_begin_(get<0>(indices)) == row_end_(get<0>(indices)) or
                 get<1>(indices) > jdata_(row_end_(get<0>(indices)) - 1),
		 "inconsistent column index in emplace_back");
    data_(row_end_(get<0>(indices))) = value_type(std::forward<Args>(args)...);
    jdata_(row_end_(get<0>(indices))) = index_type(get<1>(indices));
    ++row_end_(get<0>(indices));
  }

  template<class Pair = std::array<IndxType, 2>>
  value_type get_value(Pair&& indices)
  { 
    using std::get;
    assert(get<0>(indices) >= 0);
    assert(get<0>(indices) < size1_);
    if constexpr (mem_type == DEVICE_MEMORY)
      sfqmc::utils::check(false,"Finish csr_matrix::emplace()");
    
    auto loc   = std::lower_bound(jdata_.data() + row_begin_(get<0>(indices)), 
                                  jdata_.data() + row_end_(get<0>(indices)), get<1>(indices));
    long disp  = std::distance(jdata_.data() + row_begin_(get<0>(indices)), loc);
    long disp_ = std::distance(loc, jdata_.data() + row_end_(get<0>(indices)));

    if (disp_ > 0 && *loc == get<1>(indices)) 
    { 
      // value exists
      return data_(row_begin_(get<0>(indices)) + disp);
    }
    return value_type(0);
  }

  template<class Pair = std::array<IndxType, 2>, class TT>
  void add(Pair&& indices, TT&& v_)
  { 
    get(indices) += static_cast<value_type>(v_);
  }

  template<class Pair = std::array<IndxType, 2>, class TT>
  void scale(Pair&& indices, TT&& v_)
  { 
    get(indices) *= static_cast<value_type>(v_);
  }

  void remove_empty_spaces()
  {
    if(size1_ == 0) return;
    // MAM: make copies of ranges in GPU and just assign
    if constexpr (mem_type == DEVICE_MEMORY)
      sfqmc::utils::check(false,"Finish csr_matrix::emplace()");
    for (long i = 0; i < size1_ - 1; i++)
    {
      if (row_end_(i) == row_begin_(i + 1))
        continue;
      auto ni = static_cast<long>(row_end_(i + 1) - row_begin_(i + 1));
      std::move(data_.data() + row_begin_(i + 1),
                data_.data() + row_end_(i + 1),
                data_.data() + row_end_(i));
      std::move(jdata_.data() + row_begin_(i + 1),
                jdata_.data() + row_end_(i + 1),
                jdata_.data() + row_end_(i));
      row_begin_(i + 1) = row_end_(i);
      row_end_(i + 1)   = row_begin_(i + 1) + ni;
    }
    row_begin_(base::size1_) = row_end_(base::size1_ - 1);
  }

protected:
  virtual int serialization_code() const { return 22; }
  struct row_reference
  {
    static constexpr int dimensionality = 1;
    static constexpr int rank = 1;
    static constexpr bool sparse        = true;
    static constexpr bool sorted        = true;
    static constexpr MEMORY_SPACE mem_type = csr_matrix::mem_type;
    csr_matrix& self_;
    IndxType i_;
    using value_type   = typename csr_matrix::value_type;  
    using element      = value_type;
    using element_type = value_type;
    using element_ptr  = typename csr_matrix::element_ptr; 
    struct element_reference
    {
      row_reference& self_;
      IndxType j_;
      template<class TT>
      element_reference&& operator=(TT&& tt) &&
      {
        self_.self_.emplace({{self_.i_, j_}}, std::forward<TT>(tt));
        return std::move(*this);
      }
      template<class TT>
      element_reference&& operator+=(TT&& tt) &&
      {
        self_.self_.add({{self_.i_, j_}}, std::forward<TT>(tt));
        return std::move(*this);
      }
      template<class TT>
      element_reference&& operator*=(TT&& tt) &&
      {
        self_.self_.scale({{self_.i_, j_}}, std::forward<TT>(tt));
        return std::move(*this);
      }
      operator value_type() const 
      {
        return self_.self_.get_value({{self_.i_, j_}});
      }  
    };
    using reference = element_reference;
    template<typename integer_type = long>
    reference operator[](integer_type i) &&
    {
      return reference{*this, static_cast<IndxType>(i)};
    }

    auto values() { return self_.data_(::nda::range(self_.row_begin_(i_),self_.row_end_(i_))); }
    auto values() const { return self_.data_(::nda::range(self_.row_begin_(i_),self_.row_end_(i_))); }
    auto columns() const { return self_.jdata_(::nda::range(self_.row_begin_(i_),self_.row_end_(i_))); }
    auto nnz() const { return self_.nnz(i_); } 
    auto capacity() const { return self_.capacity(i_); }
    auto shape() const { return std::array<long, 1>{{self_.size2_}}; }
    template<typename integer_type = long>
    auto size() const
    {
      return long{self_.size2_};
    }
  };

  struct const_row_reference
  {
    static constexpr int dimensionality = 1;
    static constexpr int rank           = 1;
    static constexpr bool sparse        = true;
    static constexpr bool sorted        = true;
    static constexpr MEMORY_SPACE mem_type = csr_matrix::mem_type;
    const csr_matrix& self_;
    const IndxType i_;
    using value_type   = typename csr_matrix::value_type const;
    using element      = value_type;
    using element_type = value_type;
    using element_ptr  = typename csr_matrix::element_ptr const;

    auto values() { return self_.data_(::nda::range(self_.row_begin_(i_),self_.row_end_(i_))); }
    auto values() const { return self_.data_(::nda::range(self_.row_begin_(i_),self_.row_end_(i_))); }
    auto columns() const { return self_.jdata_(::nda::range(self_.row_begin_(i_),self_.row_end_(i_))); }
    auto nnz() const { return self_.nnz(i_); }
    auto capacity() const { return self_.capacity(i_); }
    auto shape() const { return std::array<long, 1>{{self_.size2_}}; }
    template<typename integer_type = long>
    auto size() const
    {
      return long{self_.size2_};
    }
  };

public:
  using reference       = row_reference;
  using const_reference = const_row_reference;
  template<class integer_type>
  reference operator[](integer_type i)
  {
    return reference{*this, static_cast<IndxType>(i)};
  }

  template<class integer_type = long>
  const_reference operator[](integer_type i) const
  {
    return const_reference{*this, static_cast<IndxType>(i)};
  }

  using matrix_view = csr_matrix_view<ValType, MEM, IndxType, IntType>;
  matrix_view sub_matrix(::nda::range r)
  {
    using nda::range;
    assert(r.first() >= 0 && r.last() <= size1_);
    assert(r.first() < r.last());
    // row_begin_ in host memory 
    auto r0 = row_begin_(r.first());
    ::nda::array<int_type, 1> row_b(r.size()+1);
    row_b(range(r.size())) = row_begin_(r) - r0; 
    row_b(r.size()) = row_end_(r.last()-1) - r0; 
    ::nda::array<int_type, 1> row_e = row_end_(r)-r0;
    return matrix_view({r.size(), size2_}, 
              data_(range(row_begin_(r.first()),row_end_(r.last()-1))), 
              jdata_(range(row_begin_(r.first() ),row_end_(r.last()-1))), 
              row_b, row_e);
  }

  auto operator()(::nda::range r)
  {
    sfqmc::utils::check(r.step() == 1, "csr_matrix::operator(): Only contiguous ranges allowed.");
    return sub_matrix(r);
  }

  template<std::integral Int>
  auto& operator()(Int i, Int j) {
    return this->get(std::array<IndxType, 2>{IndxType(i),IndxType(j)});
  }

  template<std::integral Int>
  auto operator()(Int i, Int j) const {
    return this->get_value(std::array<IndxType, 2>{IndxType(i),IndxType(j)});
  }

  auto operator()() { return *this; }
  auto operator()() const { return *this; }

  auto row_begin_device() const {
    if constexpr (MEM==DEVICE_MEMORY) {
      return row_begin_dev_();
    } else {
      return row_begin_();
    } 
  }
  auto row_end_device() const {
    if constexpr (MEM==DEVICE_MEMORY) {
      return row_end_dev_();
    } else {
      return row_end_();
    }
  } 
};

} // namespace sparse
} // namespace math

