target_sources(safire_lib
  PRIVATE
    Utilities/AFQMCTimer.cpp 
    Utilities/readWfn.cpp
    parameters.cpp
    parameter_defaults.cpp
    execute.cpp
    Drivers/run_afqmc.cpp
    Drivers/run_ftafqmc.cpp
    Propagators/AFQMCBasePropagator.cpp
    Walkers/WalkerSetBase.cpp
    Wavefunctions/Wavefunction.cpp
    Wavefunctions/NOMSD.cpp
    Wavefunctions/PHMSD.cpp
    Wavefunctions/NOMSD_FT.cpp
    HamiltonianOperations/HamiltonianOperations.cpp
    Hamiltonians/Hamiltonian.cpp
    Hamiltonians/KPFactorizedHamiltonian.cpp
    Hamiltonians/RealDenseHamiltonian.cpp
    Hamiltonians/ModelHamOpsGenerator.cpp
    Hamiltonians/THCHamiltonian.cpp
    Hamiltonians/KPTHCHamiltonian.cpp
)


