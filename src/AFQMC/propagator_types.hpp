#pragma once

// Split out of AFQMC/config.h so a device kernel body can name PropagatorTypes without compiling
// the whole AFQMC configuration under nvcc.

namespace sfqmc
{
namespace afqmc
{

/* Remember to propagate any changes to this enum to the device Kernel
   routines for construct_X_Model */
enum PropagatorTypes
{
  ContinuousChargePropagator,
  ContinuousSpinPropagator,
  DiscreteChargePropagator,
  DiscreteSpinPropagator,
  UndefinedPropagator
};

} // namespace afqmc
} // namespace sfqmc
