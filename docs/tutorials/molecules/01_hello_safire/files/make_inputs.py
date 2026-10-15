# This file is distributed under the Apache License, Version 2.0 License.
# See LICENSE file in top directory for details.
#
# Copyright (c) 2021-2025 The Simons Foundation, Inc.
#
# You may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0

from pyscf import gto, scf

from safiretools import MolecularHamiltonian, Wavefunction


def main():
    """
    hamiltonian.h5 and wavefunction.h5: H2 in STO-3G at a bond length of 1.4 Bohr, in the
    canonical RHF orbital basis, with the RHF determinant as the trial
    wavefunction.
    """
    mol = gto.M(atom="H 0 0 0; H 0 0 1.4", unit="Bohr", basis="sto-3g")
    mf = scf.RHF(mol).run()

    MolecularHamiltonian.from_pyscf(mf).to_hdf5("hamiltonian.h5")
    Wavefunction.from_pyscf(mf).to_hdf5("wavefunction.h5")


if __name__ == '__main__':
    main()
