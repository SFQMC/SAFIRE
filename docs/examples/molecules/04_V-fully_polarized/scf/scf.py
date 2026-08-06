#!/usr/bin/env python3

# This file is distributed under the Apache License, Version 2.0 License.
# See LICENSE file in top directory for details.
#
# Copyright (c) 2021-2025 The Simons Foundation, Inc.
#
# You may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0

from pyscf import gto, scf, mcscf


def main():
    """
    Computing the energy of ferro-magnetically coupled
       Vandium atoms.
    """

    # 0. use an isolated vanadium atom to build a guess for the ferro-magnetic case
    single_mol = gto.M(
        atom='V 0. 0. 0.',
        basis='ccpvdz',
        spin=3,
        verbose=5
    )

    # this will be a basis
    mf = scf.ROHF(single_mol).newton()
    mf.chkfile = 'rohf.chk'
    mf.kernel()

    # Report ROHF energy decomposition and total energy.
    e_elec, e_coul = mf.energy_elec()
    e_nuc = mf.energy_nuc()
    e_one = e_elec - e_coul
    e_tot = mf.e_tot

    print("ROHF one-electron energy: ", e_one)
    print("ROHF Coulomb/exchange energy: ", e_coul)
    print("ROHF electronic energy: ", e_elec)
    print("ROHF nuclear repulsion energy: ", e_nuc)
    print("ROHF total energy: ", e_tot)

    
    
    # Getting a reference energy:
    mycas = mcscf.CASCI(mf,32,3)
    E_casci =  mycas.kernel()
    print("CASCI(32,3) energy: ", E_casci[0])


if __name__ == '__main__':
    main()
