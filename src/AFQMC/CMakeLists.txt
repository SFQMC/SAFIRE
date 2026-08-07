target_sources(safire_lib
  PRIVATE
    Utilities/AFQMCTimer.cpp 
    Utilities/readWfn.cpp
    AFQMCFactory.cpp
    parameters.cpp
    parameter_defaults.cpp
    Drivers/DriverFactory.cpp
    Drivers/AFQMCDriver.cpp
    Drivers/FTAFQMCDriver.cpp
    Propagators/AFQMCBasePropagator.cpp
    Propagators/PropagatorFactory.cpp
    Wavefunctions/WavefunctionFactory.cpp
    Wavefunctions/StochasticWfn.cpp
    HamiltonianOperations/HamiltonianOperations.cpp
    Hamiltonians/KPFactorizedHamiltonian.cpp
    Hamiltonians/RealDenseHamiltonian.cpp
    Hamiltonians/ModelHamOpsGenerator.cpp
    Hamiltonians/HamiltonianFactory.cpp
    Hamiltonians/THCHamiltonian.cpp
    Hamiltonians/KPTHCHamiltonian.cpp
)


