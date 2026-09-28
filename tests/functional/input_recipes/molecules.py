# This file is distributed under the Apache License, Version 2.0 License.
# See LICENSE file in top directory for details.
#
# Copyright (c) 2021-2025 The Simons Foundation, Inc.
#
# You may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0

"""
Recipes for the molecular systems: BH, N2, Li and Pb.

All four run pyscf and then hand the SCF objects to safiretools; only the
declared HDF5 inputs are written into the inputs tree.

A note on spin symmetry. ``MolecularHamiltonian.from_pyscf`` reads it off the
source object's hcore: a spatial one gives a closed hamiltonian, and a GHF
source (``to_ghf()``, or ``with_soc = True`` for the spin-orbit ECP) a
noncollinear one. The collinear hamiltonian carries no more information than
the closed one, so it is built with the raw constructor from the closed one's
arrays.

A note on reproducibility. Every system here has degenerate orbitals - the pi
shells of BH and N2, the p/d/f shells of the Pb atom - and without symmetry an
SCF is free to return any rotation within a degenerate shell. The truncated CI
expansions (BH's CASCI, N2's CASSCF) then keep a run-dependent set of
determinants: for N2 the 50-determinant trial energy moved by up to 20 mHa
between runs. BH, N2 and Pb are therefore built with ``symmetry=True``, which
puts each member of a degenerate shell in its own irrep and pins the basis.
Repeated runs then give identical hamiltonians and identical trial energies.
What remains free is the sign of individual orbitals and determinants, the
order of equal-weight determinants, and the global spin orientation of a GHF
solution; none of it changes any energy.

Li is the exception: its ROHF quartet breaks the point-group symmetry, so it is
built without it and its orbital basis is only reproducible up to rotation.
The inputs committed before this change were built without symmetry, so
regenerating them changes the trials, and the reference results have to be
regenerated alongside the inputs.
"""

import warnings
from pathlib import Path
from typing import List

import numpy as np

from . import BuildContext, Recipe


# ============================================================================
# Shared helpers
# ============================================================================

def _pyscf_verbosity(ctx: BuildContext) -> int:
    return 5 if ctx.verbose else 3


def _write_hamiltonian(mf, filename: Path, chol_cut: float, *, basis=None,
                       verbose: bool = False):
    """Write a dense hamiltonian with `mf`'s one-body term, in `basis`'s orbitals.

    The spin symmetry follows from ``mf.get_hcore()``: closed for a spatial
    hcore, noncollinear for the spinor one a GHF source gives.
    """
    from safiretools import MolecularHamiltonian

    hamiltonian = MolecularHamiltonian.from_pyscf(
        mf,
        basis=basis,
        chol_cut=chol_cut,
        verbose=verbose,
    )
    hamiltonian.to_hdf5(filename)
    return hamiltonian


def _write_nomsd(source, filename: Path, *, basis, psi0=None) -> None:
    """Write a single-determinant trial from `source`, in `basis`'s orbitals.

    The determinant is rotated by ``C^dag S`` onto the basis the hamiltonian was
    written in, which is what the AFQMC code expects a trial to be expressed in.
    `Wavefunction.from_pyscf` does that, and picks the spin symmetry from the
    SCF solution - including the spinor promotion a GHF reference needs.
    """
    from safiretools import Wavefunction

    wavefunction = Wavefunction.from_pyscf(source, basis=basis)
    if psi0 is not None:
        wavefunction.psi0 = psi0
    wavefunction.to_hdf5(filename)


# ============================================================================
# BH
# ============================================================================

def build_bh(ctx: BuildContext) -> None:
    """BH at a stretched bond: one RHF-basis hamiltonian in three symmetries,
    plus twelve trial wavefunctions covering NOMSD and ph-MSD in each symmetry.

    The CASCI expansion is taken in the *RHF* orbital basis on purpose, so every
    wavefunction here can be paired with every hamiltonian here.
    """
    from pyscf import gto, mcscf, scf

    from safiretools import (
        MolecularHamiltonian,
        NOMSDWavefunction,
        PHMSDWavefunction,
        SpinSymm,
    )

    out = ctx.out_dir
    ci_tol = 0.02
    chol_tol = 5.0e-4
    delta = 1.5 * 1.2344  # Angstrom; 1.5x the equilibrium bond length

    mol = gto.M(
        atom=f"B {delta / 2} 0.0 0.0\nH {-delta / 2} 0.0 0.0",
        basis="ccpvdz",
        spin=0,
        symmetry=True,
        verbose=_pyscf_verbosity(ctx),
    )
    nelec = mol.nelec
    na, nb = nelec
    nmo = mol.nao_nr()

    # --- RHF: the orbital basis everything else is expressed in ------------
    rhf = scf.RHF(mol)
    rhf.kernel()

    closed = _write_hamiltonian(rhf, out / "afqmc_H_rhf_closed.h5", chol_tol,
                                verbose=ctx.verbose)
    MolecularHamiltonian(
        hcore=closed.hcore[0, 0, :, 0, :],
        chol=closed.chol,
        enuc=closed.enuc,
        spin_symm=SpinSymm.COLLINEAR,
        ortho=closed.ortho,
    ).to_hdf5(out / "afqmc_H_rhf_collinear.h5")
    _write_hamiltonian(rhf.to_ghf(), out / "afqmc_H_rhf_noncollinear.h5",
                       chol_tol, basis=rhf, verbose=ctx.verbose)

    _write_nomsd(rhf, out / "afqmc_rhf_nomsd.h5", basis=rhf)

    # --- CASCI in the RHF basis -------------------------------------------
    # CASCI rather than CASSCF so the orbitals stay exactly the RHF ones.
    mc = mcscf.CASCI(rhf, 8, 4)
    mc.run()

    # the expansion's determinants, reassembled below into every variant
    expansion = PHMSDWavefunction.from_pyscf_cas(mc, tol=ci_tol)
    ci, occa, occb = expansion.coeffs, expansion.occa, expansion.occb
    print(f"    number of determinants: {len(ci)}", flush=True)

    def write_phmsd(filename, coeffs, alpha, beta, spin_symm, **kwargs):
        PHMSDWavefunction(coeffs=coeffs, occa=alpha, occb=beta, nmo=nmo,
                          nelec=nelec, spin_symm=spin_symm, **kwargs
                          ).to_hdf5(out / filename)

    # ph-MSD, collinear: the honest form of the expansion.
    write_phmsd("afqmc_casci_uhf_phmsd.h5", ci, occa, occb, SpinSymm.COLLINEAR)
    write_phmsd("afqmc_casci_uhf_1phmsd.h5", ci[:1], occa[:1], occb[:1],
                SpinSymm.COLLINEAR)

    # ph-MSD, noncollinear: beta occupations folded into the spinor index, so
    # both polarizations live in the alpha channel and the beta one is empty.
    occ_noco = np.array([np.append(oa, ob + nmo) for oa, ob in zip(occa, occb)])
    empty = np.empty((len(occ_noco), 0), dtype=int)
    write_phmsd("afqmc_casci_ghf_phmsd.h5", ci, occ_noco, empty,
                SpinSymm.NONCOLLINEAR)
    write_phmsd("afqmc_casci_ghf_1phmsd.h5", ci[:1], occ_noco[:1], empty[:1],
                SpinSymm.NONCOLLINEAR)

    # The same expansion over an explicit RHF reference, which is the only case
    # that exercises the "mixed" type != 0 path in readWfn.cpp.
    rhf_reference = np.eye(nmo)[:, :na]
    write_phmsd("afqmc_casci_rhf_phmsd.h5", ci, occa, occb, SpinSymm.COLLINEAR,
                orbitals=rhf_reference)
    write_phmsd("afqmc_casci_rhf_1phmsd.h5", ci[:1], occa[:1], occb[:1],
                SpinSymm.COLLINEAR, orbitals=rhf_reference)

    # The same expansion as NOMSD. In the RHF basis every determinant is a
    # column selection from the identity, so the orbital matrices are exact.
    # The layout of `dets` sets the spin symmetry (see NOMSDWavefunction).
    identity = np.eye(nmo)
    alpha = identity[:, occa].transpose(1, 0, 2)  # (ndets, nmo, na)
    beta = identity[:, occb].transpose(1, 0, 2)   # (ndets, nmo, nb)

    def write_nomsd(filename, dets):
        NOMSDWavefunction(coeffs=ci, dets=dets).to_hdf5(out / filename)

    write_nomsd("afqmc_casci_uhf_nomsd.h5", (alpha, beta))

    # spinor orbitals (ndets, 2, nmo, na + nb): alpha electrons in the spin-up
    # component, beta electrons in the spin-down one
    spinors = np.zeros((len(ci), 2, nmo, na + nb), dtype=np.complex128)
    spinors[:, 0, :, :na] = alpha
    spinors[:, 1, :, na:] = beta
    write_nomsd("afqmc_casci_ghf_nomsd.h5", spinors)

    write_nomsd("afqmc_casci_rhf_nomsd.h5", alpha)

    # --- UHF and GHF trials, expressed in the RHF basis --------------------
    uhf = scf.UHF(mol=mol).newton()
    uhf.kernel()

    _write_nomsd(uhf, out / "afqmc_uhf_nomsd.h5", basis=rhf)

    ghf = uhf.to_ghf()
    dm0 = ghf.make_rdm1()
    ghf = ghf.newton()
    ghf.kernel(dm0=dm0)

    _write_nomsd(ghf, out / "afqmc_ghf_nomsd.h5", basis=rhf)


# ============================================================================
# N2
# ============================================================================

def build_n2(ctx: BuildContext) -> None:
    """Stretched N2 (3.0 Bohr) with a CASSCF(12o,6e) reference.

    Both the orbital basis and the ph-MSD trial come from the same CASSCF run,
    so this is the case that exercises a genuine multi-determinant trial.
    """
    from pyscf import gto, mcscf, scf

    from safiretools import Wavefunction

    out = ctx.out_dir
    delta = 3.0  # Bohr

    mol = gto.M(
        atom=f"N 0. 0. {delta / 2}\nN 0. 0. -{delta / 2}",
        basis="ccpvdz",
        unit="Bohr",
        symmetry=True,
        verbose=_pyscf_verbosity(ctx),
    )

    rhf = scf.RHF(mol)
    rhf.run()

    mc = mcscf.CASSCF(rhf, 12, 6).run()

    Wavefunction.from_pyscf_cas(
        mc, tol=1.0e-4, max_det=50).to_hdf5(out / "cas_wfn.h5")

    # The hamiltonian is written in the CASSCF natural orbital basis.
    _write_hamiltonian(rhf, out / "cas_basis_hamil.h5", 1e-4, basis=mc,
                       verbose=ctx.verbose)


# ============================================================================
# Li
# ============================================================================

def build_li(ctx: BuildContext) -> None:
    """Li atom in its fully polarised quartet state (nelec = (3, 0)).

    A small open-shell case where the alpha and beta sectors have different
    sizes, which is what makes it worth testing.
    """
    from pyscf import gto, scf

    out = ctx.out_dir

    # No point-group symmetry here: the lowest ROHF quartet breaks it, and the
    # symmetry-constrained solve lands 2.1 mHa higher.
    mol = gto.M(atom="Li 0. 0. 0.", basis="ccpvdz", spin=3,
                verbose=_pyscf_verbosity(ctx))

    rohf = scf.ROHF(mol).newton()
    rohf.kernel()

    _write_hamiltonian(rohf, out / "hamil_closed.h5", 1e-5, verbose=ctx.verbose)
    _write_nomsd(rohf, out / "rohf_nomsd_polarized.h5", basis=rohf)


# ============================================================================
# Pb
# ============================================================================

# Spin-orbit ECP for Pb: 60 core electrons, with the two-column
# (scalar, spin-orbit) form that pyscf's ECP-SOC integrals need.
PB_ECP_SOC = {
    "Pb": """
    Pb nelec 60
    Pb ul
    2       1.0000000              0.0000000
    Pb S
    2      12.2963030            281.2854990
    2       8.6326340             62.5202170
    Pb P
    2      10.2417900             72.2768970      -144.553795
    2       8.9241760            144.5910830       144.591083
    2       6.5813420              4.7586930        -9.517385
    2       6.2554030              9.9406210         9.940621
    Pb D
    2       7.7543360             35.8485070       -35.848507
    2       7.7202810             53.7243420        35.816228
    2       4.9702640             10.1152560       -10.115256
    2       4.5637890             14.8337310         9.889154
    Pb F
    2       3.8875120             12.2098920        -8.139928
    2       3.8119630             16.1902910         8.095145
    Pb G
    2       5.6915770             -9.0966650         4.548332
    2       5.7155670            -11.5319960        -4.612798
    """
}


def build_pb(ctx: BuildContext) -> None:
    """Pb anion with a spin-orbit ECP: the spin-orbit coupling case.

    Two hamiltonians in the ROHF orbital basis - one spin-free, one with the
    ECP spin-orbit term folded into hcore - and three trials (UHF, spin-free
    GHF, spin-orbit GHF).
    """
    from pyscf import gto, scf

    out = ctx.out_dir
    chol_tol = 5e-4

    mol = gto.M(
        atom="Pb 0. 0. 0.",
        basis="ccpvdzpp",
        ecp=PB_ECP_SOC,
        charge=-1,
        spin=3,
        symmetry=True,
        verbose=_pyscf_verbosity(ctx),
    )
    # ROHF supplies the orbital basis for everything below.
    rohf = scf.ROHF(mol)
    rohf.kernel()

    uhf = scf.UHF(mol)
    uhf.kernel()

    ghf = scf.GHF(mol)
    ghf.kernel()

    # Spin-orbit coupling mixes the spatial irreps, so this solve drops the
    # point-group symmetry the others use.
    ghf_soc = scf.GHF(mol.copy().build(symmetry=False))
    ghf_soc.with_soc = True
    ghf_soc.kernel()

    _write_nomsd(uhf, out / "afqmc_uhf_nomsd.h5", basis=rohf)

    # Spin-free: a GHF without with_soc has the scalar hcore on both spins,
    # which is deliberate here, so its warning is silenced.
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message=".*spin-orbit ECP")
        _write_hamiltonian(ghf, out / "afqmc_H_rhf_basis_noncollinear_sf.h5",
                           chol_tol, basis=rohf, verbose=ctx.verbose)

    # Spin-orbit: with_soc folds the ECP spin-orbit term into hcore.
    _write_hamiltonian(ghf_soc, out / "afqmc_H_rhf_basis_noncollinear_soc.h5",
                       chol_tol, basis=rohf, verbose=ctx.verbose)

    _write_nomsd(ghf, out / "afqmc_ghf_sf_nomsd.h5", basis=rohf)
    _write_nomsd(ghf_soc, out / "afqmc_ghf_soc_nomsd.h5", basis=rohf)


# ============================================================================
# Registry
# ============================================================================

def recipes() -> List[Recipe]:
    return [
        Recipe(key="BH", data_dir="BH", build=build_bh),
        Recipe(key="N2", data_dir="N2", build=build_n2),
        Recipe(key="Li", data_dir="Li", build=build_li),
        Recipe(key="Pb", data_dir="Pb", build=build_pb),
    ]
