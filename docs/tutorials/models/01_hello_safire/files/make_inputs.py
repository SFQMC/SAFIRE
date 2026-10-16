# This file is distributed under the Apache License, Version 2.0 License.
# See LICENSE file in top directory for details.
#
# Copyright (c) 2021-2025 The Simons Foundation, Inc.
#
# You may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0

from safiretools import HamiltonianBuilder, Lattice, Wavefunction


def main():
    """
    hamiltonian.h5 and wavefunction.h5: the 4×4 periodic Hubbard model at
    U/t = 4 with 5+5 electrons, and its free-electron trial wavefunction.
    """
    lattice = Lattice.from_dict(
        params=dict(L1=4, L2=4, boundary1="PBC", boundary2="PBC")
    )

    hamiltonian = HamiltonianBuilder.from_input(
        source=dict(hamiltonian=dict(t=1.0, U=4.0, spin_symm="collinear")),
        lattice=lattice
    ).get_hamiltonian()
    hamiltonian.to_hdf5("hamiltonian.h5")

    Wavefunction.from_free_electron(
        source=hamiltonian,
        nelec=(5, 5)
    ).to_hdf5("wavefunction.h5")


if __name__ == '__main__':
    main()
