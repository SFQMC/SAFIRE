#include "numerics/device_kernels/cuda/launch.cuh"
#include "numerics/device_kernels/device_api.hpp"
#include "numerics/operations/split_singular_vals_impl.hpp"

namespace kernels::device
{

void splitDmatrix(view<std::complex<double> const, 2> A, view<std::complex<double>, 2> B,
                  view<std::complex<double>, 2> C, view<std::complex<double>, 1> res,
                  view<std::complex<double> const, 1> scl)
{
  math::detail::splitDmatrix_impl<decltype(A), decltype(B), decltype(C), decltype(res),
                                  decltype(scl)>
      f{A, B, C, res, scl};
  bulk(A.extent(0), f);
}

} // namespace kernels::device
