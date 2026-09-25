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
    AFQMC inputs for the oxygen atom: a Hamiltonian in the ROHF orbital basis
    and a UHF trial wavefunction.
    """

    #####################################
    #                                   #
    #       Run the SCF calculations    #
    #                                   #
    #####################################

    mol = gto.M(
        atom='O 0. 0. 0.',
        basis='ccpvdz',
        spin=2,
        verbose=4
    )

    # the orbital basis
    rohf = scf.ROHF(mol).newton()
    rohf.kernel()

    # the trial wavefunction
    uhf = scf.UHF(mol).newton()
    uhf.kernel()

    chol_tol = 1e-5

    # output
    fout = 'afqmc.h5'

    #####################################
    #                                   #
    #  Write Hamiltonian in ROHF basis  #
    #                                   #
    #####################################

    MolecularHamiltonian.from_pyscf(
        rohf,
        chol_cut=chol_tol,
        verbose=True
    ).to_hdf5(fout)

    #####################################
    #                                   #
    #      Write Trial Wavefunction     #
    #                                   #
    #####################################

    Wavefunction.from_pyscf(
        uhf,
        basis=rohf
    ).to_hdf5(fout)


if __name__ == '__main__':
    main()
