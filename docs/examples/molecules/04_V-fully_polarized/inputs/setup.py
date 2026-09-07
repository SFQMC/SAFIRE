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
from safiretools import Hamiltonian

# NOTE: deliberately still on afqmctools' write_wfn_mol. This example's afqmc.json
# asks for "walker_type": "FULLYPOLARIZED", a spin symmetry safiretools no longer
# has: a reference with no beta electrons is COLLINEAR with ndown == 0 there, and
# the SAFIRE walker setup does not yet accept an empty beta sector. Port this to
# Wavefunction.from_pyscf once that C++ change lands.
from afqmctools.wavefunction.mol import write_wfn_mol


def main():

    # inputs
    basis_chk = '../scf/rohf.chk'
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

    Hamiltonian.from_pyscf(
        basis_scf_data,
        chol_cut=chol_tol,
        real_chol=True,
        verbose=True,
        cas=cas_afqmc  # provide the CAS info here
    ).to_hdf5(fout)
    
    write_wfn_mol(
        scf_data = basis_scf_data,
        filename = fout,
        cas = cas_afqmc  # must match the CAS info used for the Hamiltonian!
    )


if __name__ == '__main__':
    main()
