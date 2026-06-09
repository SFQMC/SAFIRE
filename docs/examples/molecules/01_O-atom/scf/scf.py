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


from pyscf import gto, scf

def main():
    """
    Computing the energy of ferro-magnetically coupled
       Vandium atoms.       
    """
    mol = gto.M(
        atom='O 0. 0. 0.',
        basis='ccpvdz',
        spin=2,
        verbose=4
    )

    mf = scf.ROHF(mol).newton()
    mf.chkfile = 'rohf.chk'
    mf.kernel()

    mf = scf.UHF(mol).newton()
    mf.chkfile = 'uhf.chk'
    mf.kernel()

if __name__ == '__main__':
    main()
