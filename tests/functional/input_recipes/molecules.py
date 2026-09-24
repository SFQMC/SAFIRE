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

All four run pyscf and then hand the checkpoint to afqmctools. The pyscf
checkpoints land in ``ctx.scratch`` so a rerun is self-contained; only the
declared HDF5 inputs are written into the inputs tree.

A note on spin symmetry. ``write_hamil_mol`` no longer takes a ``walker_type``
argument: it infers the symmetry from the shape of ``scf_data['hcore']`` and
writes ``nelec`` straight from ``mol.nelec``. The collinear and noncollinear
hamiltonians here therefore go through ``generate_hamiltonian`` +
``write_dense`` directly, which is the only way to stack the one-body term and
to write the ``(nup + ndn, 0)`` electron count the noncollinear convention
wants.

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

from pathlib import Path
from typing import List

import h5py as h5
import numpy as np

from . import BuildContext, Recipe


# ============================================================================
# Shared helpers
# ============================================================================

def _pyscf_verbosity(ctx: BuildContext) -> int:
    return 5 if ctx.verbose else 3


def _write_hamiltonian(scf_data, filename: Path, chol_cut: float, *,
                       spin_symm: str, verbose: bool = False) -> None:
    """Write a dense generic hamiltonian in the requested spin symmetry.

    ``spin_symm`` is one of ``closed`` / ``collinear`` / ``noncollinear``. All
    three share the same ``X^dag h X`` one-body block; what differs is how it is
    blocked out, which `MolecularHamiltonian` does from `spin_symm` alone - one
    spin sector closed, two identical ones collinear, and a single sector over a
    spinor basis noncollinear.
    """
    from safiretools import MolecularHamiltonian

    if spin_symm == "noncollinear" and scf_data["hcore"].shape[-1] != 2 * scf_data["norb"]:
        # A scalar hcore promoted into the spinor basis: h -> I_2 (x) h.
        scf_data = dict(scf_data)
        scf_data["hcore"] = np.kron(np.eye(2), scf_data["hcore"])

    MolecularHamiltonian.from_pyscf(
        scf_data,
        chol_cut=chol_cut,
        spin_symm=spin_symm,
        verbose=verbose,
    ).to_hdf5(filename)


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

    from safiretools import NOMSDWavefunction, PHMSDWavefunction, SpinSymm
    from safiretools.convert.pyscf import load_pyscf_chk_mol
    from safiretools.wavefunction.pyscf import ci_expansion, read_cas_meta

    out, scratch = ctx.out_dir, ctx.scratch
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
    rhf_chk = scratch / "rhf.chk"
    mf = scf.RHF(mol)
    mf.chkfile = str(rhf_chk)
    mf.kernel()

    rhf_data = load_pyscf_chk_mol(rhf_chk)
    _write_hamiltonian(rhf_data, out / "afqmc_H_rhf_closed.h5", chol_tol,
                       spin_symm="closed", verbose=ctx.verbose)
    _write_hamiltonian(rhf_data, out / "afqmc_H_rhf_collinear.h5", chol_tol,
                       spin_symm="collinear", verbose=ctx.verbose)
    _write_hamiltonian(rhf_data, out / "afqmc_H_rhf_noncollinear.h5", chol_tol,
                       spin_symm="noncollinear", verbose=ctx.verbose)

    _write_nomsd(rhf_data, out / "afqmc_rhf_nomsd.h5", basis=rhf_data)

    # --- CASCI in the RHF basis -------------------------------------------
    # CASCI rather than CASSCF so the orbitals stay exactly the RHF ones.
    mc = mcscf.CASCI(mf, 8, 4)
    mc.chkfile = str(rhf_chk)
    mc.run()
    mcscf.chkfile.dump_mcscf(mc, str(rhf_chk))
    with h5.File(rhf_chk, "a") as fh5:
        if "mcscf/ci" in fh5:
            del fh5["mcscf/ci"]
        fh5["mcscf/ci"] = mc.ci

    cas_meta = read_cas_meta(rhf_chk)
    ncas, ncore = int(cas_meta["ncas"]), int(cas_meta["ncore"])
    ci, occa, occb = ci_expansion(
        cas_meta["ci"],
        norb=ncas,
        nelec=[n - ncore for n in nelec],
        ncore=ncore,
        tol=ci_tol,
    )
    print(f"    number of determinants: {len(ci)}", flush=True)
    ci = np.array(ci, dtype=np.complex128)

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
    rhf_reference = [np.eye(nmo)[:, :na]]
    write_phmsd("afqmc_casci_rhf_phmsd.h5", ci, occa, occb, SpinSymm.COLLINEAR,
                orbitals=rhf_reference)
    write_phmsd("afqmc_casci_rhf_1phmsd.h5", ci[:1], occa[:1], occb[:1],
                SpinSymm.COLLINEAR, orbitals=rhf_reference)

    # The same expansion as NOMSD. In the RHF basis every determinant is a
    # column selection from the identity, so the orbital matrices are exact.
    identity = np.eye(nmo)

    def write_nomsd(filename, dets, spin_symm):
        NOMSDWavefunction(coeffs=ci, dets=np.array(dets), nelec=nelec,
                          spin_symm=spin_symm, nmo=nmo).to_hdf5(out / filename)

    nomsd_collinear = []
    for oa, ob in zip(occa, occb, strict=True):
        phi = np.zeros((nmo, sum(nelec)), dtype=np.complex128)
        phi[:, :na] = identity[:, oa]
        phi[:, na:] = identity[:, ob]
        nomsd_collinear.append(phi)
    write_nomsd("afqmc_casci_uhf_nomsd.h5", nomsd_collinear, SpinSymm.COLLINEAR)

    nomsd_noncollinear = []
    for oa, ob in zip(occa, occb, strict=True):
        phi = np.zeros((2 * nmo, sum(nelec)), dtype=np.complex128)
        phi[:nmo, :na] = identity[:, oa]
        phi[nmo:, na:] = identity[:, ob]
        nomsd_noncollinear.append(phi)
    write_nomsd("afqmc_casci_ghf_nomsd.h5", nomsd_noncollinear,
                SpinSymm.NONCOLLINEAR)

    nomsd_closed = []
    for oa in occa:
        phi = np.zeros((nmo, na), dtype=np.complex128)
        phi[:, :na] = identity[:, oa]
        nomsd_closed.append(phi)
    write_nomsd("afqmc_casci_rhf_nomsd.h5", nomsd_closed, SpinSymm.CLOSED)

    # --- UHF and GHF trials, expressed in the RHF basis --------------------
    uhf_chk = scratch / "uhf.chk"
    mf = scf.UHF(mol=mol).newton()
    mf.chkfile = str(uhf_chk)
    mf.kernel()

    _write_nomsd(load_pyscf_chk_mol(uhf_chk), out / "afqmc_uhf_nomsd.h5",
                 basis=rhf_data)

    # Same UHF trial, but started from the RHF determinant: exercises the
    # separate initial-walker path.
    rhf_initial = np.zeros((nmo, na), dtype=np.complex128)
    rhf_initial[:na, :na] = np.eye(na)
    _write_nomsd(load_pyscf_chk_mol(uhf_chk),
                 out / "afqmc_uhf_nomsd_init_rhf.h5",
                 basis=rhf_data, psi0=[rhf_initial, rhf_initial])

    ghf_chk = scratch / "ghf.chk"
    mf = mf.to_ghf()
    dm0 = mf.make_rdm1()
    mf = mf.newton()
    mf.chkfile = str(ghf_chk)
    mf.kernel(dm0=dm0)

    _write_nomsd(load_pyscf_chk_mol(ghf_chk), out / "afqmc_ghf_nomsd.h5",
                 basis=rhf_data)


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
    from safiretools.convert.pyscf import load_pyscf_chk_mol

    out, scratch = ctx.out_dir, ctx.scratch
    delta = 3.0  # Bohr

    mol = gto.M(
        atom=f"N 0. 0. {delta / 2}\nN 0. 0. -{delta / 2}",
        basis="ccpvdz",
        unit="Bohr",
        symmetry=True,
        verbose=_pyscf_verbosity(ctx),
    )

    casscf_chk = scratch / "rhf_casscf_chkfile.h5"
    rhf = scf.RHF(mol)
    rhf.chkfile = str(casscf_chk)
    rhf.run()

    mc = mcscf.CASSCF(rhf, 12, 6).run()

    with h5.File(casscf_chk, "a") as fh5:
        for name, value in (("ci", mc.ci), ("ncore", mc.ncore), ("ncas", mc.ncas)):
            if f"mcscf/{name}" in fh5:
                del fh5[f"mcscf/{name}"]
            fh5[f"mcscf/{name}"] = np.asarray(value)

    Wavefunction.from_pyscf_cas(
        mol, casscf_chk, tol=1.0e-4, max_det=50).to_hdf5(out / "cas_wfn.h5")

    # The hamiltonian is written in the CASSCF natural orbital basis.
    basis_scf_data = load_pyscf_chk_mol(chkfile=casscf_chk, base="mcscf")
    _write_hamiltonian(basis_scf_data, out / "cas_basis_hamil.h5", 1e-4,
                       spin_symm="closed", verbose=ctx.verbose)


# ============================================================================
# Li
# ============================================================================

def build_li(ctx: BuildContext) -> None:
    """Li atom in its fully polarised quartet state (nelec = (3, 0)).

    A small open-shell case where the alpha and beta sectors have different
    sizes, which is what makes it worth testing.
    """
    from pyscf import gto, scf

    from safiretools.convert.pyscf import load_pyscf_chk_mol

    out, scratch = ctx.out_dir, ctx.scratch

    # No point-group symmetry here: the lowest ROHF quartet breaks it, and the
    # symmetry-constrained solve lands 2.1 mHa higher.
    mol = gto.M(atom="Li 0. 0. 0.", basis="ccpvdz", spin=3,
                verbose=_pyscf_verbosity(ctx))

    rohf_chk = scratch / "rohf.chk"
    mf = scf.ROHF(mol).newton()
    mf.chkfile = str(rohf_chk)
    mf.kernel()

    scf_data = load_pyscf_chk_mol(rohf_chk, "scf")
    _write_hamiltonian(scf_data, out / "hamil_closed.h5", 1e-5,
                       spin_symm="closed", verbose=ctx.verbose)
    _write_nomsd(scf_data, out / "rohf_nomsd_polarized.h5", basis=scf_data)


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

    from safiretools.convert.pyscf import load_pyscf_chk_mol

    out, scratch = ctx.out_dir, ctx.scratch
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
    nelec = mol.nelec

    rohf_chk = scratch / "rohf.chk"
    uhf_chk = scratch / "uhf.chk"
    ghf_chk = scratch / "ghf.chk"
    ghf_soc_chk = scratch / "ghf_soc.chk"

    # ROHF supplies the orbital basis for everything below.
    mf = scf.ROHF(mol)
    mf.chkfile = str(rohf_chk)
    mf.kernel()

    mf = scf.UHF(mol)
    mf.chkfile = str(uhf_chk)
    mf.kernel()

    mf = scf.GHF(mol)
    mf.chkfile = str(ghf_chk)
    mf.kernel()

    # Spin-orbit coupling mixes the spatial irreps, so this solve drops the
    # point-group symmetry the others use.
    mf = scf.GHF(mol.copy().build(symmetry=False))
    mf.chkfile = str(ghf_soc_chk)
    mf.with_soc = True
    mf.kernel()

    basis_data = load_pyscf_chk_mol(rohf_chk, "scf")

    _write_nomsd(load_pyscf_chk_mol(uhf_chk), out / "afqmc_uhf_nomsd.h5",
                 basis=basis_data)

    # Spin-free: promote the scalar hcore into the spinor basis.
    _write_hamiltonian(basis_data,
                       out / "afqmc_H_rhf_basis_noncollinear_sf.h5", chol_tol,
                       spin_symm="noncollinear", verbose=ctx.verbose)

    # Spin-orbit: hcore already comes back as a complex (2 norb, 2 norb) block.
    soc_data = load_pyscf_chk_mol(rohf_chk, "scf", soc_type="ecp")
    _write_hamiltonian(soc_data,
                       out / "afqmc_H_rhf_basis_noncollinear_soc.h5", chol_tol,
                       spin_symm="noncollinear", verbose=ctx.verbose)

    _write_nomsd(load_pyscf_chk_mol(ghf_chk), out / "afqmc_ghf_sf_nomsd.h5",
                 basis=basis_data)
    _write_nomsd(load_pyscf_chk_mol(ghf_soc_chk),
                 out / "afqmc_ghf_soc_nomsd.h5", basis=basis_data)


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
