# This file is distributed under the Apache License, Version 2.0 License.
# See LICENSE file in top directory for details.
#
# Copyright (c) 2021-2025 The Simons Foundation, Inc.
#
# You may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0

import numpy as np

from afqmctools.utils.pyscf_utils import load_from_pyscf_chk_mol
from afqmctools.hamiltonian.mol import write_hamil_mol
from afqmctools.wavefunction.mol import write_wfn_mol


def main():

    # inputs
    basis_chk = '../scf/rohf.chk'
    wfn_chk = '../scf/uhf.chk'
    chol_tol = 1e-5
    cas_afqmc = (3,32)
    
    # output
    fout = 'afqmc.h5'


    basis_scf_data = load_from_pyscf_chk_mol(basis_chk, 'scf')

    mol = basis_scf_data['mol']
    nelec = mol.nelec

    ncore = nelec[0] - cas_afqmc[0]
    assert ncore >= 0

    print("="*20 + " scf_info " + "="*20)
    for key,value in basis_scf_data.items():
        print(f"{key}  :  {value}")

    write_hamil_mol(
        basis_scf_data,
        fout, 
        chol_tol, 
        dense=True,
        real_chol=True, 
        verbose=True,
        walker_type='fullypolarized',
        cas=cas_afqmc  # provide the CAS info here
    )
    

if __name__ == '__main__':
    main()
