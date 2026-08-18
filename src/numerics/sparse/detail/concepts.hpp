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

#include <concepts>
#include "nda/nda.hpp"

namespace math::sparse
{

/*
 * Sparse vector concept
 */
template <typename A>
concept CSRVector = requires(A const& a) {
  { a.nnz() };
  { a.values() };
  { a.columns() };
  { a.size() }; 
  { a.shape() } -> ::nda::StdArrayOfLong;
  { std::decay_t<A>::sparse == true };
  { std::decay_t<A>::rank == 1 };
  { std::decay_t<A>::sorted == true };
};

/*
 * Sparse matrix concept
 */
template <typename A>
concept CSRMatrix = requires(A const& a) {
  { a.nnz() };
  { a.values() };
  { a.columns() };
  { a.row_begin() };
  { a.row_end() };
  { a.nnz(int()) };
  { a.values(int()) };
  { a.columns(int()) };
  { a.row_begin(int()) };
  { a.row_end(int()) };
  { a.shape() } -> ::nda::StdArrayOfLong;
  { std::decay_t<A>::sparse == true };
  { std::decay_t<A>::rank == 2 };
  { std::decay_t<A>::sorted == true };
}; 

}

namespace memory
{

template<math::sparse::CSRMatrix A>
constexpr MEMORY_SPACE get_memory_space()
{
  return std::decay_t<A>::mem_type;
}

}

namespace nda
{

namespace mem {

  template <math::sparse::CSRMatrix A>
  static constexpr AddressSpace get_addr_space<A> = to_nda_address_space(A::mem_type);

  template <math::sparse::CSRVector A>
  static constexpr AddressSpace get_addr_space<A> = to_nda_address_space(A::mem_type);

} // mem

} // nda

