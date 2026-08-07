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

#include <compare>
#include <cstddef>
#include <map>
#include <type_traits>
#include "AFQMC/config.h"
#include "numerics/sparse/array_of_sequences.hpp"

namespace sfqmc
{
namespace afqmc
{
/*
// CHEAT!!!
template<class TP, class integer>
long get_index(TP const& tp_, integer loc)
{
  if (loc == 0)
    return long(std::get<0>(tp_));
  else if (loc == 1)
    return long(std::get<1>(tp_));
  else
    throw std::runtime_error(" Error in sfqmc::afqmc::get_index<TP,integer>(). ");
}
*/
template<class Vector>
void push_excitation(Vector const& abij, Vector& v)
{
  if (abij.size() == 0)
    return;
  utils::check(v.size() % abij.size() == 0, "Size mismatch");
  long n = abij.size();
  for (typename Vector::iterator it = v.begin(); it < v.end(); it += n)
    if (std::equal(abij.begin(), abij.end(), it))
      return;
  v.insert(v.end(), abij.begin(), abij.end());
}

template<class Vector>
long find_excitation(Vector const& abij, Vector& v)
{
  if (abij.size() == 0)
    return 0; // this assumes that the reference is 0
  utils::check(v.size() % abij.size() == 0, "Size mismatch");
  long n   = abij.size();
  long loc = 0;
  for (typename Vector::iterator it = v.begin(); it < v.end(); it += n, loc++)
    if (std::equal(abij.begin(), abij.end(), it))
      return loc;
  utils::check(false,"Error: Sequence not found in find_excitation.");
  return 0;
}

template<class excitations>
//std::map<int, int> find_active_space(bool single_list, WALKER_TYPES walker_type, excitations const& abij, int NMO, int nup, int ndown)
std::vector<long> find_active_space(WALKER_TYPES walker_type, excitations const& abij, int NMO, int nup, int ndown)
{
  if( walker_type == NONCOLLINEAR ) utils::check(ndown == 0, "ndown>0 with non-collinear");
//  std::map<int, int> mo2active;
//  for (int i = 0; i < 2 * NMO; i++)
//    mo2active[i] = -1;
  std::vector<long> count(2 * NMO, 0);
  // reference first
  auto refc = abij.reference_configuration();
  for (int i = 0; i < nup + ndown; i++, ++refc)
    ++count[*refc];
  auto nex = abij.maximum_excitation_number();
  for (int n = 1; n < nex[0]; ++n)
  {
    auto it    = abij.alpha_begin(n);
    auto itend = abij.alpha_end(n);
    for (; it < itend; ++it)
    {
      auto exct = *it + n; // skip locations
      for (int ak = 0; ak < n; ++ak, ++exct)
        ++count[*exct];
    }
  }
  if( walker_type == COLLINEAR) {
    for (int n = 1; n < nex[1]; ++n)
    {
      auto it    = abij.beta_begin(n);
      auto itend = abij.beta_end(n);
      for (; it < itend; ++it)
      {
        auto exct = *it + n; // skip locations
        for (int ak = 0; ak < n; ++ak, ++exct)
          ++count[*exct];
      }
    }
  }
  return count;
/*
  if (not single_list)
  {
// check!!!!!
    utils::check(walker_type == COLLINEAR, "single_list=false requires collinear. walker_type: {}", walkerTypeToString(walker_type));
    int ik = 0;
    for (int i = 0; i < NMO; ++i)
      if (count[i] > 0 || count[i + NMO] > 0)
      {
        if (count[i] > 0)
          mo2active[i] = ik;
        if (count[i + NMO] > 0)
          mo2active[i + NMO] = ik;
        ++ik;
      }
  }
  else
  {
    int ik = 0;
    for (int i = 0; i < npol*NMO; ++i)
      if (count[i] > 0)
        mo2active[i] = ik++;
    if(walker_type == COLLINEAR) {
      ik = 0;
      for (int i = NMO; i < 2 * NMO; ++i)
        if (count[i] > 0)
          mo2active[i] = ik++;
    }
  }
  return mo2active;
*/
}

/*
 * - exct stores, for each electron excitation, the location of the orbital in the reference 
 *    being excited and the index of the excited orbital.
 */
template<class Vector, class T>
int get_excitation_number(bool getIndx, nda::MemoryVector auto&& refc, nda::MemoryVector auto&& confg, Vector& exct, T& ci, Vector& Iwork)
{
  int NE = refc.size();
  exct.clear();
  int cnt = 0;
  if (getIndx)
    std::copy(refc.begin(), refc.end(), Iwork.begin());
  auto it = refc.data();
  utils::check(Iwork.size() >= refc.size(), "Size mismatch");
  for (int i = 0; i < NE; i++, it++)
    if (!std::binary_search(confg.begin(), confg.end(), *it))
    {
      if (getIndx)
      {
        // store the location, NOT the index!!!
        //exct.emplace_back(*it);
        exct.emplace_back(i);
      }
      cnt++;
    }
  if (!getIndx)
    return cnt;
  it       = confg.data();
  int cnt2 = 0;
  for (int i = 0; i < NE; i++, it++)
    if (!std::binary_search(refc.begin(), refc.end(), *it))
    {
      exct.emplace_back(*it);
      Iwork[exct[cnt2]] = *it;
      cnt2++;
    }
  utils::check(cnt == cnt2, "Logic error!");
  // sort Iwork and count number of exchanges to determine permutation sign
  // sooo slow but sooo simple too
  for (int i = 0; i < NE; i++)
    for (int j = i + 1; j < NE; j++)
    {
      if (Iwork[j] < Iwork[i])
      {
        ci *= T(-1.0);
        std::swap(Iwork[i], Iwork[j]);
      }
    }
  return cnt;
}

template<typename intT, class csr>
nda::array<long,1> get_nnz(csr const& PsiT_MO, intT* refc, long N, long shift)
{
  nda::array<long,1> res(N);
  for (long i = 0; i < N; i++)
    res(i) = PsiT_MO.nnz(refc[i] - shift);
  return res;
}

// Holds information necessary to reconstruct particle-hole multi-determinant expansions
template<class I       = int,
         class VType   = std::complex<double>,
         MEMORY_SPACE MEM = HOST_MEMORY> 
struct ph_excitations
{
public:
  using integer_type       = I;
  using configuration_type = std::tuple<int, int, VType>;

private:
  using confg_aos  = math::sparse::array_of_sequences<configuration_type,HOST_MEMORY,int>;
  using index_aos  = math::sparse::array_of_sequences<integer_type,HOST_MEMORY,int>; 
  using dev_index_aos  = math::sparse::array_of_sequences<integer_type,MEM,int>; 
  // Iterates over a packed list of particle-hole excitations. Each element is a block of 2*D
  // indices (D holes followed by D particles); dereferencing yields a pointer to the block.
  // Integer may be const-qualified, which yields Excitation_const_Iterator.
  template<typename Integer>
  class Iterator
  {
  public:
    using difference_type = std::ptrdiff_t;
    using value_type      = Integer*;
    using reference       = Integer*;

    Iterator() = default;
    Iterator(Integer* index, long d_) : p_index(index), D(d_) {}

    // non-const -> const conversion
    template<typename U>
      requires(std::is_same_v<Integer, U const>)
    Iterator(Iterator<U> const& other) : p_index(other.p_index), D(other.D) {}

    reference operator*() const { return p_index; }

    Iterator& operator++() {
      p_index += 2 * D;
      return *this;
    }
    Iterator operator++(int) {
      auto tmp = *this;
      ++(*this);
      return tmp;
    }

    friend bool operator==(Iterator const& a, Iterator const& b) { return a.p_index == b.p_index; }
    friend auto operator<=>(Iterator const& a, Iterator const& b) { return a.p_index <=> b.p_index; }

  private:
    template<typename>
    friend class Iterator;

    Integer* p_index = nullptr;
    long D           = 1;
  };

  template<typename It, typename AOS>
  static It excitation_begin(AOS& u, int n) {
    utils::check(n > 0, "ph_excitations: excitation number must be > 0, got {}", n);
    if(n < u.size()) {
      return It(u.sequence(n).data(), n);
    }
    return excitation_end<It>(u, n);
  }

  template<typename It, typename AOS>
  static It excitation_end(AOS& u, int n) {
    utils::check(n > 0, "ph_excitations: excitation number must be > 0, got {}", n);
    if(n < u.size()) {
      return It(u.values().data() + u.sequence_end(n), n);
    }
    return It(u.values().data() + u.sequence_end(u.size() - 1), 1);
  }

public:
  using Excitation_Iterator       = Iterator<integer_type>;
  using Excitation_const_Iterator = Iterator<integer_type const>;

  ph_excitations() = default;

  // Note: terms_per_excitation[0] has special meaning, the number of electrons in the calculation.
  // coefficients[0] will store the reference configuration itself.
  ph_excitations(long number_of_configurations,
                 int na_,
                 int nb_,
                 std::vector<long>& unique_alpha_counts,
                 std::vector<long>& unique_beta_counts)
      : 
        nup(na_),
        ndown(nb_),
        configurations(1, number_of_configurations),
        reference(1, nup + ndown),
        unique_alpha(unique_alpha_counts.size(), unique_alpha_counts),
        unique_beta(unique_beta_counts.size(), unique_beta_counts)
#if defined(ENABLE_DEVICE)
        ,reference_dev(1, nup + ndown), 
        unique_alpha_dev(unique_alpha_counts.size(), unique_alpha_counts), 
        unique_beta_dev(unique_beta_counts.size(), unique_beta_counts) 
#endif
  {
    long emax = std::max(unique_alpha_counts.size(), unique_beta_counts.size());
    sum_of_exct.resize(emax + 1);
    sum_of_exct[0] = {0, 0};
    sum_of_exct[1] = {1, 1};
    for (long n = 1; n < unique_alpha.size(); ++n)
      sum_of_exct[n + 1][0] = sum_of_exct[n][0] + unique_alpha_counts[n] / long(2) / n;
    for (long n = unique_alpha.size() + 1; n <= emax; n++)
      sum_of_exct[n][0] = sum_of_exct[n - 1][0];
    for (long n = 1; n < unique_beta.size(); ++n)
      sum_of_exct[n + 1][1] = sum_of_exct[n][1] + unique_beta_counts[n] / long(2) / n;
    for (long n = unique_beta.size() + 1; n <= emax; n++)
      sum_of_exct[n][1] = sum_of_exct[n - 1][1];
  }

  std::array<long, 2> maximum_excitation_number() const { return {unique_alpha.size(), unique_beta.size()}; }
  long number_of_unique_alpha_excitations(int n) const
  {
    if (n == 0)
      return 1;
    return unique_alpha.num_elements(n) / 2 / n;
  }
  long number_of_unique_beta_excitations(int n) const
  {
    if (n == 0)
      return 1;
    return unique_beta.num_elements(n) / 2 / n;
  }
  std::array<long, 2> number_of_unique_excitations(int n) const
  {
    if (n == 0)
      return {1, 1};
    std::array<long, 2> res{0, 0};
    if (n < unique_alpha.size())
      res[0] = unique_alpha.num_elements(n) / 2 / n;
    if (n < unique_beta.size())
      res[1] = unique_beta.num_elements(n) / 2 / n;
    return res;
  }
  std::array<long, 2> number_of_unique_excitations() const { return sum_of_exct.back(); }

  long number_of_configurations() const { return configurations.num_elements(0); }

  // returns the number of unique excitations with particle number less than n
  std::array<long, 2> number_of_unique_smaller_than(int n) const { return sum_of_exct[n]; }

  template<class intIt>
  void add_alpha(long n, intIt indx)
  {
    utils::check(n > 0 and n < unique_alpha.size(), "out of bounds");
    for (int i = 0; i < 2 * n; i++, ++indx)
      unique_alpha.emplace_back(n, static_cast<integer_type>(*indx));
  }

  template<class intIt>
  void add_beta(long n, intIt indx)
  {
    utils::check(n > 0 and n < unique_beta.size(), "out of bounds");
    for (int i = 0; i < 2 * n; i++, ++indx)
      unique_beta.emplace_back(n, static_cast<integer_type>(*indx));
  }

  template<class Vector>
  void add_reference(Vector& refa, Vector& refb)
  {
    for (auto k : refa)
      reference.emplace_back(0, static_cast<integer_type>(k));
    for (auto k : refb)
      reference.emplace_back(0, static_cast<integer_type>(k));
  }

  // index=0 is reserved for the reference!!!
  template<typename integer, typename value>
  void add_configuration(integer alpha_indx, integer beta_index, value ci)
  {
    configurations.emplace_back(0, configuration_type{alpha_indx, beta_index, ci});
  }

  integer_type const* reference_configuration(int spin = 0) const
  {
    return reference.values().data() + (spin == 0 ? 0 : nup);
  }

  integer_type* reference_configuration(int spin = 0)
  {
    return reference.values().data() + (spin == 0 ? 0 : nup);
  }

  auto configurations_begin() const { return configurations.values().begin(); }

  auto configurations_end() const
  {
    return configurations.values().begin() + configurations.sequence_end(0);
  }

  auto configuration(int i) const { return configurations.values().begin() + i; }

  Excitation_Iterator alpha_begin(int n) { return excitation_begin<Excitation_Iterator>(unique_alpha, n); }

  Excitation_Iterator alpha_end(int n) { return excitation_end<Excitation_Iterator>(unique_alpha, n); }

  Excitation_const_Iterator alpha_begin(int n) const {
    return excitation_begin<Excitation_const_Iterator>(unique_alpha, n);
  }

  Excitation_const_Iterator alpha_end(int n) const {
    return excitation_end<Excitation_const_Iterator>(unique_alpha, n);
  }

  Excitation_Iterator beta_begin(int n) { return excitation_begin<Excitation_Iterator>(unique_beta, n); }

  Excitation_Iterator beta_end(int n) { return excitation_end<Excitation_Iterator>(unique_beta, n); }

  Excitation_const_Iterator beta_begin(int n) const {
    return excitation_begin<Excitation_const_Iterator>(unique_beta, n);
  }

  Excitation_const_Iterator beta_end(int n) const {
    return excitation_end<Excitation_const_Iterator>(unique_beta, n);
  }

  // for generic access
  std::array<Excitation_Iterator, 2> unique_begin(int n)
  {
    return std::array<Excitation_Iterator, 2>{alpha_begin(n), beta_begin(n)};
  }
  std::array<Excitation_Iterator, 2> unique_end(int n)
  {
    return std::array<Excitation_Iterator, 2>{alpha_end(n), beta_end(n)};
  }
  std::array<Excitation_const_Iterator, 2> unique_begin(int n) const
  {
    return std::array<Excitation_const_Iterator, 2>{alpha_begin(n), beta_begin(n)};
  }
  std::array<Excitation_const_Iterator, 2> unique_end(int n) const
  {
    return std::array<Excitation_const_Iterator, 2>{alpha_end(n), beta_end(n)};
  }

  template<typename Vec>
  void get_alpha_configuration(long index, Vec&& confg) const
  {
    utils::check(confg.size() >= nup, "out of bounds");
    std::copy_n(reference.values().data(), nup, confg.data());
    if (index == 0)
      return;
    // could use lower bound
    for (int i = 1; i < unique_alpha.size(); i++)
    {
      if (index >= sum_of_exct[i][0] && index < sum_of_exct[i + 1][0])
      {
        long dn = index - sum_of_exct[i][0];
        auto exct = unique_alpha.sequence(i).data() + 2 * i * dn;
        for (int n = 0; n < i; n++)
          confg[exct[n]] = exct[n + i];
        return;
      }
    }
    utils::check(false," Error in ph_excitations::get_alpha_configuration() ");
  }

  template<typename Vec>
  void get_beta_configuration(long index, Vec&& confg) const
  {
    utils::check(confg.size() >= ndown, "out of bounds");
    std::copy_n(reference.values().data() + nup, ndown, confg.data());
    if (index == 0)
      return;
    // could use lower bound
    for (int i = 1; i < unique_beta.size(); i++)
    {
      if (index >= sum_of_exct[i][1] && index < sum_of_exct[i + 1][1])
      {
        long dn = index - sum_of_exct[i][1];
        auto exct = unique_beta.sequence(i).data() + 2 * i * dn;
        for (int n = 0; n < i; n++)
          confg[exct[n]] = exct[n + i];
        return;
      }
    }
    utils::check(false," Error in ph_excitations::get_beta_configuration() ");
  }

  template<typename Vec>
  void get_configuration(int spin, long index, Vec&& confg) const
  {
    if (spin == 0)
      get_alpha_configuration(index, confg);
    else
      get_beta_configuration(index, confg);
  }

  auto get_excitation_list_device(int ispin, int iex) 
  {
    if constexpr (MEM==HOST_MEMORY) {
      if(ispin == 0) {
        utils::check(iex >= 0 and iex < unique_alpha.size(), "out of bounds");
        return unique_alpha.sequence(iex);
      } else {
        utils::check(iex >= 0 and iex < unique_beta.size(), "out of bounds");
        return unique_beta.sequence(iex);
      }
    } else {
      if(not dev_init) {
        dev_init = true;
        reference_dev = reference;
        unique_alpha_dev = unique_alpha;
        unique_beta_dev = unique_beta;  
      }
      if(ispin == 0) {
        utils::check(iex >= 0 and iex < unique_alpha_dev.size(), "out of bounds");
        return unique_alpha_dev.sequence(iex);
      } else {
        utils::check(iex >= 0 and iex < unique_beta_dev.size(), "out of bounds");
        return unique_beta_dev.sequence(iex);
      }
    }
  }

  auto get_reference_configuration_device(int ispin) 
  {
    if constexpr (MEM==HOST_MEMORY) {
      return reference.sequence(0)(nda::range(ispin*nup,nup+ispin*ndown)); 
    } else {
      if(not dev_init) {
        dev_init = true;
        reference_dev = reference;
        unique_alpha_dev = unique_alpha;   
       unique_beta_dev = unique_beta;    
      } 
      return reference_dev.sequence(0)(nda::range(ispin*nup,nup+ispin*ndown)); 
    }
  }

private:

  int nup, ndown;
  confg_aos configurations;
  index_aos reference;
  index_aos unique_alpha;
  index_aos unique_beta;
  std::vector<std::array<long, 2>> sum_of_exct;
  bool dev_init = false;
  dev_index_aos reference_dev;
  dev_index_aos unique_alpha_dev;
  dev_index_aos unique_beta_dev;

};

} // namespace afqmc

} // namespace sfqmc

