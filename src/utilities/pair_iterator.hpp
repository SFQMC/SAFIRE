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

#include <compare>
#include <iterator>
#include <utility>

// A random access iterator over two parallel sequences, so that std::sort can order one
// of them while carrying the other along.
//
// Once we move to C++23 this whole header should be replaced by
//   std::ranges::sort(std::views::zip(keys, values), {}, [](auto const& t) { return std::get<0>(t); });
// Note that zipping nda views also needs nda's rank-1 array_iterator::operator[] to be
// const-qualified; without that it fails std::random_access_iterator and zip_view rejects it.

namespace sfqmc::utils
{

// Proxy reference into two parallel sequences. A program-defined type, so ADL finds
// swap() without adding overloads to namespace std.
template<typename T1, typename T2>
struct paired_ref {
  T1& first;
  T2& second;

  paired_ref(T1& a, T2& b) : first(a), second(b) {}
  operator std::pair<T1, T2>() const { return {first, second}; }

  paired_ref const& operator=(paired_ref const& o) const { first = o.first; second = o.second; return *this; }
  paired_ref const& operator=(std::pair<T1, T2> const& v) const { first = v.first; second = v.second; return *this; }

  friend void swap(paired_ref a, paired_ref b) {
    using std::swap;
    swap(a.first, b.first);
    swap(a.second, b.second);
  }
};

template<typename It1, typename It2>
class paired_iterator {
  It1 first_{};
  It2 second_{};

  using v1 = typename std::iterator_traits<It1>::value_type;
  using v2 = typename std::iterator_traits<It2>::value_type;

public:
  using value_type        = std::pair<v1, v2>;
  using reference         = paired_ref<v1, v2>;
  using pointer           = void;
  using difference_type   = typename std::iterator_traits<It1>::difference_type;
  using iterator_category = std::random_access_iterator_tag;

  paired_iterator() = default;
  paired_iterator(It1 a, It2 b) : first_(a), second_(b) {}

  reference operator*() const { return {*first_, *second_}; }
  reference operator[](difference_type n) const { return *(*this + n); }

  paired_iterator& operator++() { ++first_; ++second_; return *this; }
  paired_iterator& operator--() { --first_; --second_; return *this; }
  paired_iterator operator++(int) { auto c = *this; ++*this; return c; }
  paired_iterator operator--(int) { auto c = *this; --*this; return c; }
  paired_iterator& operator+=(difference_type n) { first_ += n; second_ += n; return *this; }
  paired_iterator& operator-=(difference_type n) { first_ -= n; second_ -= n; return *this; }

  friend paired_iterator operator+(paired_iterator it, difference_type n) { return it += n; }
  friend paired_iterator operator+(difference_type n, paired_iterator it) { return it += n; }
  friend paired_iterator operator-(paired_iterator it, difference_type n) { return it -= n; }
  friend difference_type operator-(paired_iterator const& a, paired_iterator const& b) { return a.first_ - b.first_; }

  friend bool operator==(paired_iterator const& a, paired_iterator const& b) { return a.first_ == b.first_; }
  // `= default` would be deleted: nda's rank-1 array_iterator has no operator<=>
  friend std::strong_ordering operator<=>(paired_iterator const& a, paired_iterator const& b) { return (a.first_ - b.first_) <=> 0; }
};

template<typename It1, typename It2>
paired_iterator<It1, It2> make_paired_iterator(It1 a, It2 b) { return {a, b}; }

} // namespace sfqmc::utils
