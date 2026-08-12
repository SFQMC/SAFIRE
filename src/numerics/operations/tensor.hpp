#pragma once

#include "configuration.hpp"
#include "utilities/check.hpp"
#include "nda/nda.hpp"

namespace math
{


// hermitization of A along the last two dimensions
void hermitize(nda::Array auto&& A) {
  constexpr int rank = nda::get_rank<decltype(A)>;
  static_assert(rank > 2, "A needs at least rank 2");

  int n = A.size() / (A.extent(rank-2) * A.extent(rank-1));
  auto A3 = reshape(A, n, A.extent(rank-2), A.extent(rank-1));

  for(int k = 0; k < A3.extent(0); k++) {
    for(int i = 0; i < A3.extent(1); i++) {
      for(int j = i; j < A3.extent(2); j++) {
        if(i == j) {
          A3(k, i, i) = nda::real(A3(k, i, i));
        } else {
          A3(k, i, j) = 0.5 * A3(k, i, j) + 0.5* nda::conj(A3(k, j, i));
          A3(k, j, i) = nda::conj(A3(k, i, j));
        }
      }
    }
  }
}

// B(...) = T(A(...)), where T = nda::get_value_t<decltype(B)>
template<nda::MemoryArray A_t, nda::MemoryArray B_t>
requires( nda::get_rank<A_t> == nda::get_rank<B_t> ) 
void copy(A_t const& A, B_t && B) {
  sfqmc::utils::check(A.shape() == B.shape(), "Shape mismatch");
  if constexpr (std::is_same_v<nda::get_value_t<A_t>,nda::get_value_t<B_t>>) {
#if defined(ENABLE_DEVICE)
    if constexpr (nda::mem::have_device_compatible_addr_space<A_t,B_t>) {
      kernels::device::copy(A,B);
    } else
#endif
      B() = A();
  } else {
#if defined(ENABLE_DEVICE)
    if constexpr (nda::mem::have_device_compatible_addr_space<A_t,B_t>) {
      kernels::device::copy_cast(A,B);
    } else 
#endif
      B() = A();
  }
}

// B(...) += T(alpha*A(...)), where T = nda::get_value_t<decltype(B)>
template<nda::MemoryArray A_t, nda::MemoryArray B_t>
requires( nda::get_rank<A_t> == nda::get_rank<B_t> )
void accumulate(nda::get_value_t<A_t> alpha, A_t const& A, B_t && B) {
  sfqmc::utils::check(A.shape() == B.shape(), "Shape mismatch");
  if constexpr (std::is_same_v<nda::get_value_t<A_t>,nda::get_value_t<B_t>>) {
#if defined(ENABLE_DEVICE)
    if constexpr (nda::mem::have_device_compatible_addr_space<A_t,B_t> and nda::get_rank<A_t> < 5) {
      kernels::device::accumulate(alpha,A,B);
    } else
#endif
      nda::tensor::add(alpha,A,nda::get_value_t<B_t>{1.0},B);
  } else {
#if defined(ENABLE_DEVICE)
    if constexpr (nda::mem::have_device_compatible_addr_space<A_t,B_t>) {
      kernels::device::accumulate_cast(alpha,A,B);
    } else 
#endif
      B() += alpha*A();
  }
}

template<nda::MemoryArray Arr>
void zero_imag(Arr && A)
{
  using value_type = nda::get_value_t<Arr>;
  if constexpr (nda::is_complex_v<value_type>) {
    if constexpr(nda::mem::on_host<Arr>) {
      for( auto& v : A ) v = value_type(v.real(),0.0);
    } else {
#if defined(ENABLE_DEVICE)
      kernels::device::zero_imag(A);
#else
      sfqmc::utils::check(false,"Error: Found device array without ENABLE_DEVICE.");
#endif
    }
  }
}

template<nda::MemoryArray Arr>
void set_identity(Arr && A)
requires( nda::get_rank<Arr> > 1 and nda::get_rank<Arr> < 6 and std::decay_t<Arr>::is_stride_order_C())
{
  using value_type = nda::get_value_t<Arr>;
  auto A_diag = memory::diagonal_view(A);
  A() = value_type(0.0);
  A_diag() = value_type(1.0);
}

}
