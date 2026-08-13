#include "AFQMC/Propagators/construct_X_kernel.hpp"
#include "numerics/device_kernels/cuda/launch.cuh"
#include "numerics/device_kernels/device_api.hpp"

namespace kernels::device
{

void construct_X(bool zero, bool fp, double sqrtdt, double vbias_bound,
                 view<int const, 1> FieldTypes, view<std::complex<double> const, 1> vMF,
                 view<std::complex<double>, 1> mf_factor,
                 view<std::complex<double>, 1> hybrid_weight, view<double const, 2> RN,
                 view<std::complex<double>, 2> X)
{
  sfqmc::afqmc::detail::construct_X_impl f{zero, fp,  sqrtdt,        vbias_bound, FieldTypes,
                                           vMF,  mf_factor, hybrid_weight, RN,          X};
  // layout_left goes against the natural stride order of X, but reduces contention on the atomic add in the kernel
  for_each<::cuda::std::layout_left>(X, f);
}

} // namespace kernels::device
