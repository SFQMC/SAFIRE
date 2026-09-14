# This file is distributed under the Apache License, Version 2.0 License.
# See LICENSE file in top directory for details.
#
# Copyright (c) 2021-2025 The Simons Foundation, Inc.
#
# You may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0

from safiretools import MolecularHamiltonian, Wavefunction


def main():

    # inputs
    basis_chk = '../scf/rohf.chk'
    chol_tol = 1e-5
    cas_afqmc = (3,32)
    
    # output
    fout = 'afqmc.h5'

    MolecularHamiltonian.from_pyscf(
        basis_chk,
        chol_cut=chol_tol,
        real_chol=True,
        verbose=True,
        cas=cas_afqmc  # provide the CAS info here
    ).to_hdf5(fout)

    # Freezing the 10 doubly occupied core orbitals leaves 3 alpha and 0 beta
    #   electrons in the active space, so the trial is collinear with ndown == 0
    Wavefunction.from_pyscf(
        basis_chk,
        cas = cas_afqmc  # must match the CAS info used for the Hamiltonian!
    ).to_hdf5(fout)


if __name__ == '__main__':
    main()
