#include <complex>
#include <algorithm>
#include <cuda/std/cmath>

#include "configuration.hpp"
#include "utilities/check.hpp"
#include "utilities/type_traits.hpp"
#include "numerics/device_kernels/cuda/cuda_aux.hpp"
#include "numerics/device_kernels/cuda/cuda_settings.h"
#include "nda/nda.hpp"
#include "nda/tensor.hpp"
#include <cuda/std/mdspan>
#include <cub/device/device_for.cuh>
#include "numerics/operations/add_diagonal_impl.hpp"

namespace kernels::device::detail
{

template<typename V> 
void apply_impl(nda::get_value_t<V> alpha, V& A, nda::tensor::unary_op oper)
{
  constexpr int rank = nda::get_rank<V>;
  if(A.size()==0) return;
  // these seem fine in cutensor_permute
  if(oper==nda::tensor::unary_op::IDENTITY or oper==nda::tensor::unary_op::CONJ) // or oper==nda::tensor::unary_op::RCP)
  {
    nda::tensor::scale(alpha,A,oper);
    return;
  } else if(oper==nda::tensor::unary_op::NEG) {
    nda::tensor::scale(-alpha,A);
    return;
  }
  auto alpha_d = cuda_std_value_cast(alpha);
  auto A_d = to_cuda_std_mdspan(A);
  long N = A.size();
  using value_t = std::decay_t<decltype(alpha_d)>;

  // some operations appear to not be allowed in cutensor permute
  switch (oper) {
    case nda::tensor::unary_op::SQRT:
    {
      //a = alpha * sqrt(a); break;
      if constexpr (rank==1) {
        auto F = [=] __device__(long n) {
          A_d(n) = alpha_d * cuda::std::sqrt(A_d(n));
        };
        cub::DeviceFor::Bulk(N,F);
        break;
      } else if constexpr (rank==2) {
        long nc = A.extent(1);
        auto F = [=] __device__(long n) {
          long i = n/nc;
          long j = n - i*nc;
          A_d(i,j) = alpha_d * cuda::std::sqrt(A_d(i,j));
        };
        cub::DeviceFor::Bulk(N,F);
        break;
      } else {
        sfqmc::utils::check(false,"Invalid rank in kernel::apply.");
      }
    }
    case nda::tensor::unary_op::ABS:
    {
      //a = alpha * abs(a); break;
      if constexpr (rank==1) {
        auto F = [=] __device__(long n) {
          A_d(n) = alpha_d * cuda::std::abs(A_d(n));
        };
        cub::DeviceFor::Bulk(N,F);
        break;
      } else if constexpr (rank==2) {
        long nc = A.extent(1);
        auto F = [=] __device__(long n) {
          long i = n/nc;
          long j = n - i*nc;
          A_d(i,j) = alpha_d * cuda::std::abs(A_d(i,j));
        };
        cub::DeviceFor::Bulk(N,F);
        break;
      } else {
        sfqmc::utils::check(false,"Invalid rank in kernel::apply.");
      }
    }
    case nda::tensor::unary_op::RCP:
    {
      //a = alpha * (1/a); break;
      if constexpr (rank==1) {
        auto F = [=] __device__(long n) {
          A_d(n) = alpha_d/A_d(n);
        };
        cub::DeviceFor::Bulk(N,F);
        break;
      } else if constexpr (rank==2) {
        long nc = A.extent(1);
        auto F = [=] __device__(long n) {
          long i = n/nc;
          long j = n - i*nc;
          A_d(i,j) = alpha_d/A_d(i,j);
        };
        cub::DeviceFor::Bulk(N,F);
        break;
      } else {
        sfqmc::utils::check(false,"Invalid rank in kernel::apply.");
      }
    }
    case nda::tensor::unary_op::EXP:
    {
      //a = alpha * nda::exp(a); break;
      if constexpr (rank==1) {
        auto F = [=] __device__(long n) {
          A_d(n) = alpha_d * cuda::std::exp(A_d(n));
        };
        cub::DeviceFor::Bulk(N,F);
        break;
      } else if constexpr (rank==2) {
        long nc = A.extent(1);
        auto F = [=] __device__(long n) {
          long i = n/nc;
          long j = n - i*nc;
          A_d(i,j) = alpha_d * cuda::std::exp(A_d(i,j));
        };
        cub::DeviceFor::Bulk(N,F);
        break;
      } else {
        sfqmc::utils::check(false,"Invalid rank in kernel::apply.");
      }
    }
    case nda::tensor::unary_op::LOG:
    {
      // a = alpha * nda::log(a); break;
      if constexpr (rank==1) {
        auto F = [=] __device__(long n) {
          A_d(n) = alpha_d * cuda::std::log(A_d(n));
        }; 
        cub::DeviceFor::Bulk(N,F);
        break;
      } else if constexpr (rank==2) {
        long nc = A.extent(1);
        auto F = [=] __device__(long n) {
          long i = n/nc;
          long j = n - i*nc;
          A_d(i,j) = alpha_d * cuda::std::log(A_d(i,j));
        }; 
        cub::DeviceFor::Bulk(N,F);
        break;
      } else {
        sfqmc::utils::check(false,"Invalid rank in kernel::apply.");
      } 
    }
    default: {
      sfqmc::utils::check(false,"Invalid operation in kernel::apply.");
    } 
  };
}
  
using memory::device_array_view;
using std::complex;

template<int Rank>
using basic_layout_t = typename nda::basic_layout<0, nda::C_stride_order<Rank>, nda::layout_prop_e::none>;

#define _inst_(T,V) \
template void apply_impl(T,V<T,1,basic_layout_t<1>>&,nda::tensor::unary_op);  \
template void apply_impl(T,V<T,2,basic_layout_t<2>>&,nda::tensor::unary_op);  \
template void apply_impl(T,V<T,3,basic_layout_t<3>>&,nda::tensor::unary_op);

_inst_(double,device_array_view)
_inst_(std::complex<double>,device_array_view)

} //kernels::device::detail
