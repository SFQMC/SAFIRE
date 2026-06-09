# This file is distributed under the Apache License, Version 2.0 License.
# See LICENSE file in top directory for details.
#
# Copyright (c) 2021-2025 The Simons Foundation, Inc.
#
# You may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0

from afqmctools.utils.pyscf_utils import load_from_pyscf_chk_mol
from afqmctools.hamiltonian.mol import write_hamil_mol
from afqmctools.wavefunction.mol import write_wfn_mol


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

    scf_data = load_from_pyscf_chk_mol(
        orbital_basis_chk,
        'scf'
    )

    write_hamil_mol(
        scf_data=scf_data,
        hamil_file=fout, 
        chol_cut=chol_tol, 
        dense=True,
        real_chol=True, 
        verbose=True
    )
    
    #####################################
    #                                   #
    #      Write Trial Wavefunction     #
    #                                   #
    #####################################

    write_wfn_mol(
        scf_data=load_from_pyscf_chk_mol(
            wavefunction_chk,
            'scf'
        ),
        basis_scf_data=scf_data,
        filename=fout
    )


if __name__ == '__main__':
    main()

