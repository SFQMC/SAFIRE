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
    orbital_basis_chk = '../scf/rohf.chk'
    wavefunction_chk = '../scf/uhf.chk'

    chol_tol = 1e-5

    # output
    fout = 'afqmc.h5'

    #####################################
    #                                   #
    #  Write Hamiltonian in ROHF basis  #
    #                                   #
    #####################################

    MolecularHamiltonian.from_pyscf(
        orbital_basis_chk,
        chol_cut=chol_tol,
        real_chol=True,
        verbose=True
    ).to_hdf5(fout)
    
    #####################################
    #                                   #
    #      Write Trial Wavefunction     #
    #                                   #
    #####################################

    Wavefunction.from_pyscf(
        wavefunction_chk,
        basis=orbital_basis_chk
    ).to_hdf5(fout)


if __name__ == '__main__':
    main()

