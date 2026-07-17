
#pragma once


#if defined(ENABLE_CUDA)

#include "numerics/device_kernels/cuda/argmax_min.cuh"
#include "numerics/device_kernels/cuda/copy_select.cuh"
#include "numerics/device_kernels/cuda/copy_cast.cuh"
#include "numerics/device_kernels/cuda/complex_tools.cuh"
#include "numerics/device_kernels/cuda/complex_tools.cuh"
#include "numerics/device_kernels/cuda/construct_fields.cuh"
#include "numerics/device_kernels/cuda/determinants.cuh"
#include "numerics/device_kernels/cuda/accumulate.cuh"
#include "numerics/device_kernels/cuda/split_singular_vals.cuh"
#include "numerics/device_kernels/cuda/add_diagonal.cuh"
#include "numerics/device_kernels/cuda/add_scalar.cuh"
#include "numerics/device_kernels/cuda/phmsd_routines.cuh"
#include "numerics/device_kernels/cuda/apply.cuh"
#include "numerics/device_kernels/cuda/stochastic_reductions.cuh"

#endif

