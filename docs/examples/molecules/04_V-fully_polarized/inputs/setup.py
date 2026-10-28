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

from safiretools import MolecularHamiltonian, Wavefunction

# The internally stable ROHF solution for the S=3/2 vanadium atom in cc-pVDZ.
#   An isolated V atom has other ROHF solutions, and the
#   CASCI and AFQMC reference energies quoted in this example all assume this one.
ROHF_ENERGY = -942.884909528
ROHF_ENERGY_TOL = 1e-4


def solve_stable_rohf(mol, max_rotations=5):
    """
    Solve ROHF, re-solving from the lowest instability direction until the
    solution is internally stable.

    Internal stability asks whether the solution is a genuine ROHF minimum
    rather than a saddle point; it does not consider the symmetry-breaking
    (ROHF -> UHF) rotations that external stability would.
    """
    mf = scf.ROHF(mol).newton()
    mf.kernel()

    for _ in range(max_rotations):
        mo, _, stable, _ = mf.stability(return_status=True)
        if stable:
            break
        # rotate the orbitals along the instability and re-solve from there
        mf.kernel(mo, mf.mo_occ)
    else:
        raise RuntimeError(
            f"ROHF is still internally unstable after {max_rotations} rotations"
        )

    if not mf.converged:
        raise RuntimeError("ROHF reached a stable solution but did not converge")

    return mf


def main():
    """
    AFQMC inputs for the fully polarized vanadium atom, with a CASCI(32o,3e)
    reference energy.
    """

    # 1. ROHF on the isolated vanadium atom, whose orbitals are the basis
    mol = gto.M(
        atom='V 0. 0. 0.',
        basis='ccpvdz',
        spin=3,
        verbose=5
    )

    rohf = solve_stable_rohf(mol)

    # Report ROHF energy decomposition and total energy.
    e_elec, e_coul = rohf.energy_elec()
    e_nuc = rohf.energy_nuc()
    e_one = e_elec - e_coul
    e_tot = rohf.e_tot

    print("ROHF one-electron energy: ", e_one)
    print("ROHF Coulomb/exchange energy: ", e_coul)
    print("ROHF electronic energy: ", e_elec)
    print("ROHF nuclear repulsion energy: ", e_nuc)
    print("ROHF total energy: ", e_tot)

    if abs(e_tot - ROHF_ENERGY) > ROHF_ENERGY_TOL:
        raise RuntimeError(
            f"ROHF total energy {e_tot:.9f} Ha differs from the expected stable "
            f"solution {ROHF_ENERGY:.9f} Ha by more than {ROHF_ENERGY_TOL:g} Ha. "
            "The CASCI and AFQMC reference energies in this example assume that "
            "solution, so they will not be reproduced."
        )

    # Getting a reference energy:
    mycas = mcscf.CASCI(rohf,32,3)
    E_casci =  mycas.kernel()
    print("CASCI(32,3) energy: ", E_casci[0])

    # 2. the AFQMC inputs
    chol_tol = 1e-5
    cas_afqmc = (3,32)

    # output
    fout = 'afqmc.h5'

    MolecularHamiltonian.from_pyscf(
        rohf,
        chol_cut=chol_tol,
        verbose=True,
        active_space=cas_afqmc  # provide the active space here
    ).to_hdf5(fout)

    # Freezing the 10 doubly occupied core orbitals leaves 3 alpha and 0 beta
    #   electrons in the active space, so the trial is collinear with ndown == 0
    Wavefunction.from_pyscf(
        rohf,
        active_space = cas_afqmc  # must match the Hamiltonian's active space!
    ).to_hdf5(fout)


if __name__ == '__main__':
    main()
