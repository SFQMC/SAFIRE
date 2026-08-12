
#pragma once

#include <complex>
#include "numerics/device_kernels/cuda/nda_aux.hpp"

namespace kernels::device
{

namespace detail
{

template<typename A, typename B> 
void copy_cast_impl(A const& a, B & b);

template<typename A, typename B> 
void accumulate_cast_impl(nda::get_value_t<A> alpha, A const& a, B & b);

}

/*
 *   B(...) = T(A(...)) 
 */
template<nda::MemoryArray A_t, nda::MemoryArray B_t> 
void copy_cast(A_t const& A, B_t && B)
requires( nda::get_rank<A_t> == nda::get_rank<B_t> and
       ((std::decay_t<A_t>::is_stride_order_C() and std::decay_t<B_t>::is_stride_order_C()) or
        (std::decay_t<A_t>::is_stride_order_Fortran() and std::decay_t<B_t>::is_stride_order_Fortran())))
{
  sfqmc::utils::check(A.shape() == B.shape(), "Shape mismatch");
  if(A.is_contiguous() and B.is_contiguous()) {
    kernels::device::copy_cast(nda::flatten(A),nda::flatten(B));
    return; 
  }
  // careful with ranks here 
  auto A_b = to_basic_layout(A()); 
  auto B_b = to_basic_layout(B()); 
  if constexpr (nda::get_rank<A_t> < 3)
    detail::copy_cast_impl(A_b,B_b);  
  else
    sfqmc::utils::check(false, "Calling copy_cast with rank > 2 array. Finish");
}

/*
 *   B(...) += T(alpha * A(...)) 
 */
template<nda::MemoryArray A_t, nda::MemoryArray B_t>
void accumulate_cast(nda::get_value_t<A_t> alpha, A_t const& A, B_t && B)
requires( nda::get_rank<A_t> == nda::get_rank<B_t> and
       ((std::decay_t<A_t>::is_stride_order_C() and std::decay_t<B_t>::is_stride_order_C()) or
        (std::decay_t<A_t>::is_stride_order_Fortran() and std::decay_t<B_t>::is_stride_order_Fortran())))
{
  sfqmc::utils::check(A.shape() == B.shape(), "Shape mismatch");
  if(A.is_contiguous() and B.is_contiguous()) {
    kernels::device::accumulate_cast(alpha,nda::flatten(A),nda::flatten(B));
    return;
  }
  // careful with ranks here 
  auto A_b = to_basic_layout(A());
  auto B_b = to_basic_layout(B());
  if constexpr (nda::get_rank<A_t> < 3)
    detail::accumulate_cast_impl(alpha,A_b,B_b);
  else
    sfqmc::utils::check(false, "Calling accumulate_cast with rank > 2 array. Finish");
} 

} // namespace kernels::device

