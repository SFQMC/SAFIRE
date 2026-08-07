

//////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////

#include "stdio.h"
#include <complex>
#include <algorithm>

#include "configuration.hpp"
#include "utilities/check.hpp"
#include "utilities/type_traits.hpp"
#include "numerics/device_kernels/cuda/cuda_settings.h"
#include "numerics/device_kernels/cuda/cuda_aux.hpp"
#include "nda/nda.hpp"
#include <cuda/std/mdspan>
#include "cub/device/device_for.cuh"

namespace kernels::device::detail
{

// This can be generalized with a random access iterator that works on the GPU.
template<typename A, typename B> 
void copy_cast_impl(A const& a, B & b)
{
  static_assert(nda::get_rank<A> == nda::get_rank<B> and nda::get_rank<A> < 3, "Rank mismatch");
  using T = nda::get_value_t<B>;
  constexpr int rank = nda::get_rank<A>;
  sfqmc::utils::check(a.shape() == b.shape(), "Shape mismatch");
  auto a_d = to_cuda_std_mdspan(a);
  auto b_d = to_cuda_std_mdspan(b);
  long sz = a.size();
  if constexpr (rank==1) {
    auto f = [=] __device__(long n) {
      b_d(n) = T(a_d(n));
    };
    cub::DeviceFor::Bulk(sz,f);
  } else if constexpr (rank==2) {
    if constexpr (std::decay_t<A>::is_stride_order_C()) {
      int N = a.extent(1); 
      auto f = [=] __device__(long n) {
        long i = n/N;
        long j = n - i*N;
        b_d(i,j) = T(a_d(i,j));
      };
      cub::DeviceFor::Bulk(sz,f);
    } else {
      int N = a.extent(0); 
      auto f = [=] __device__(long n) {
        long i = n/N;
        long j = n - i*N;
        b_d(j,i) = T(a_d(j,i));
      };
      cub::DeviceFor::Bulk(sz,f);
    }
  } 
}

template<typename A, typename B>
void accumulate_cast_impl(nda::get_value_t<A> alpha, A const& a, B & b)
{
  static_assert(nda::get_rank<A> == nda::get_rank<B> and nda::get_rank<A> < 3, "Rank mismatch");
  using T = nda::get_value_t<B>;
  constexpr int rank = nda::get_rank<A>;
  sfqmc::utils::check(a.shape() == b.shape(), "Shape mismatch");
  auto a_d = to_cuda_std_mdspan(a);
  auto b_d = to_cuda_std_mdspan(b);
  long sz = a.size();
  if constexpr (rank==1) {
    auto f = [=] __device__(long n) {
      b_d(n) += T(alpha*a_d(n));
    };
    cub::DeviceFor::Bulk(sz,f);
  } else if constexpr (rank==2) {
    if constexpr (std::decay_t<A>::is_stride_order_C()) {
      int N = a.extent(1);
      auto f = [=] __device__(long n) {
        long i = n/N;
        long j = n - i*N;
        b_d(i,j) += T(alpha*a_d(i,j));
      };
      cub::DeviceFor::Bulk(sz,f);
    } else {
      int N = a.extent(0);
      auto f = [=] __device__(long n) {
        long i = n/N;
        long j = n - i*N;
        b_d(j,i) += T(alpha*a_d(j,i));
      };
      cub::DeviceFor::Bulk(sz,f);
    }
  }
}

using memory::device_array_view;
using std::complex;

template<int Rank>
using basic_layout_t = typename nda::basic_layout<0, nda::C_stride_order<Rank>, nda::layout_prop_e::none>;

#define _inst_(V1,T1,V2,T2) \
template void copy_cast_impl(V1<const T1,1,basic_layout_t<1>> const&, V2<T2,1,basic_layout_t<1>> &);  \
template void copy_cast_impl(V1<const T1,2,basic_layout_t<2>> const&, V2<T2,2,basic_layout_t<2>> &);  

_inst_(device_array_view,double,device_array_view,float)
_inst_(device_array_view,float,device_array_view,double)
_inst_(device_array_view,std::complex<double>,device_array_view,std::complex<float>)
_inst_(device_array_view,std::complex<float>,device_array_view,std::complex<double>)
_inst_(device_array_view,double,device_array_view,std::complex<double>)
// what else???


} // namespace kernels::device::detail

