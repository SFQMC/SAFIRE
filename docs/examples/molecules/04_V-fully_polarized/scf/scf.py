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

import numpy as np
import matplotlib.pyplot as plt
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

    mf.analyze()

    # this will be a trial wavefunction
    uhf = scf.UHF(single_mol).newton()
    uhf.chkfile = 'uhf.chk'
    uhf.kernel()
    
    # Getting a reference energy:
    mycas = mcscf.CASCI(mf,32,3)
    E_casci =  mycas.kernel()
    print("CASCI(32,3) energy: ", E_casci[0])


if __name__ == '__main__':
    main()
