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

#pragma once

#include <random>
#include <type_traits>
#include <memory>

#include "config.h"
#include "configuration.hpp"
#include "IO/AppAbort.hpp"

#include "AFQMC/config.h"
#include "AFQMC/Walkers/WalkerConfig.hpp"

namespace sfqmc
{
namespace afqmc
{

/*
 * Basic class to access/interface walker information.
 * It uses wlk_indices and wlk_descriptor to translate a linear, contiguous segment
 * of memory into walker properties.
 * The location of the memory is determined at construction of the object, it is a runtime property.
 */ 
template<MEMORY_SPACE MEM, typename _value_t_>
struct walker
{
public:
  using value_type = _value_t_;
  using decay_value_type = typename std::decay<value_type>::type;
  using SMType  = memory::array_view<MEM,value_type,2,nda::C_layout>;
  using SVType  = memory::array_view<MEM,value_type,1,nda::C_layout>;

  walker() {
    utils::check(false, "Error: Empty walker not allowed");
  }

  walker(value_type* p, long sz, const wlk_indices& i_, const wlk_descriptor& d_)
      : _data(p), _size(sz), indx(i_), desc(d_)
  { }

  walker(nda::MemoryArrayOfRank<1> auto&& a, const wlk_indices& i_, const wlk_descriptor& d_)
      : _data(a.data()), _size(a.size()), 
        indx(i_), desc(d_)
  {
    utils::check(a.strides()[0] == 1, "Stride mismatch."); 
  }

  ~walker() {}

  // no copy/move assignment
  walker(walker&& other)      = default;
  walker(walker const& other) = default;
  walker& operator=(walker&& other) = delete;
  walker& operator=(walker const& other) = delete;

  auto base() { return _data; }
  auto size() const { return _size; }
  auto SlaterMatrix(SpinTypes s)
  {
    return (s == Alpha) ? (SMType({desc[0], desc[1]}, getw_(SM)))
                        : (SMType({desc[0], desc[2]}, getw_(SM) + desc[0] * desc[1]));
  }
  auto UMatrix(SpinTypes s)
  {
    return (s == Alpha) ? (SMType({desc[0], desc[1]}, getw_(UR)))
                        : (SMType({desc[0], desc[2]}, getw_(UR) + desc[0] * desc[1]));
  }
  auto DMatrix(SpinTypes s)
  {
    return (s == Alpha) ? (SVType({desc[0]}, getw_(DR)))
                        : (SVType({desc[0]}, getw_(DR) + desc[0]));
  }
  auto VMatrix(SpinTypes s)
  {
    return (s == Alpha) ? (SMType({desc[0], desc[1]}, getw_(VR)))
                        : (SMType({desc[0], desc[2]}, getw_(VR) + desc[0] * desc[1]));
  }
  auto SlaterMatrixN(SpinTypes s)
  {
    utils::check(indx[SMN] >= 0, "error: access to uninitialized BP sector. ");
    return (s == Alpha) ? (SMType({desc[0], desc[1]}, getw_(SMN)))
                        : (SMType({desc[0], desc[2]}, getw_(SMN) + desc[0] * desc[1]));
  }
  // accessor functions. Only defined from host, no device calls allowed. 
  decay_value_type get_property(walker_data P) const { return get_value(P); }
  template<typename V>
  void set_property(walker_data P, V val) { set_value(P,static_cast<decay_value_type>(val)); }
  decay_value_type energy() const { return get_value(E1_) + get_value(EXX_) + get_value(EJ_); }
  // replaces Slater Matrix at timestep M+N to timestep N for back propagation.
  void setSlaterMatrixN()
  {
      SlaterMatrixN(Alpha) = SlaterMatrix(Alpha);
      if (desc[2] > 0) SlaterMatrixN(Beta) = SlaterMatrix(Beta);
  }

private:
  value_type* _data = nullptr;
  long _size = 0;
  const wlk_indices& indx;
  const wlk_descriptor& desc;

  auto getw_(int P) { return _data + indx[P]; }
  auto getw_(int P) const { return _data + indx[P]; }

  void check_allowed_property(walker_data P) const {
    utils::check(P==WEIGHT or P==PHASE or P==PHASE1 or P==PHASE2 or P==THETA or 
        P==PSEUDO_ELOC_ or P==E1_ or P==EXX_ or P==EJ_ or P==OVLP or P==LOGSCL_UP
        or P==LOGSCL_DN or P==IS_UNITARY, "Invalid property.");
  }

  decay_value_type get_value(walker_data P) const {
    check_allowed_property(P);
    decay_value_type res;
#if defined(ENABLE_DEVICE)
    if constexpr (MEM == DEVICE_MEMORY) {
      nda::mem::memcpy<nda::mem::Host,nda::mem::Device>(std::addressof(res), getw_(P), sizeof(value_type));   
      return res;
    } else 
#endif
    {
      res = *getw_(P);
      return res;
    }
  }

  void set_value(walker_data P, decay_value_type val) {
    check_allowed_property(P);
#if defined(ENABLE_DEVICE)
    if constexpr (MEM == DEVICE_MEMORY) {
      nda::mem::memcpy<nda::mem::Device,nda::mem::Host>(getw_(P), std::addressof(val), sizeof(decay_value_type));
    } else
#endif
    {
      *getw_(P) = val;
    }
  }
};

// MAM: How to make sure that ptr is allocated consistently with MEM? Would need to work with array_views...
template<MEMORY_SPACE MEM, typename _value_t_>
struct walker_iterator
    : public boost::
          iterator_facade<walker_iterator<MEM,_value_t_>, void, std::random_access_iterator_tag, walker<MEM,_value_t_>, std::ptrdiff_t>
{
public:
  walker_iterator(int k, nda::MemoryArrayOfRank<2> auto&& w_, const wlk_indices& i_, const wlk_descriptor& d_)
      : pos(k), ptr(w_.data()), stride(w_.strides()[0]), size(w_.extent(1)), 
        indx(&i_), desc(&d_)
  {
    utils::check(w_.strides()[1] == 1, "Stride mismatch");
  }

  walker_iterator(int k, _value_t_* _p, long st, long sz, const wlk_indices& i_, const wlk_descriptor& d_)
      : pos(k), ptr(_p), stride(st), size(sz), indx(&i_), desc(&d_) 
  { }

  using element         = _value_t_; 
  using pointer         = element*;
  using difference_type = std::ptrdiff_t;
  using reference       = walker<MEM,element>;

private:
  int pos;
  mutable element* ptr = nullptr;
  const long stride = 0;
  const long size = 0;
  wlk_indices const* indx;
  wlk_descriptor const* desc;

  friend class boost::iterator_core_access;

  void increment() { ++pos; }
  void decrement() { --pos; }
  bool equal(walker_iterator const& other) const { return pos == other.pos; }
  void advance(difference_type n) { pos += n; }
  difference_type distance_to(walker_iterator other) const { return other.pos - pos; }
  reference dereference() const { return reference(ptr+pos*stride,size,*indx,*desc); }
};

} // namespace afqmc

} // namespace sfqmc

