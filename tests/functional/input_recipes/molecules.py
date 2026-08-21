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

A note on reproducibility. These recipes reproduce the *physics* of the
committed files but not their bytes. Every system here has degenerate orbitals
- the pi shells of BH and N2, the p/d/f shells of the Pb atom - and an SCF is
free to return any rotation within a degenerate shell, with any sign. So the
orbital basis ``Hamiltonian/X``, and everything expressed in it, comes out
rotated relative to what is committed. Basis-independent quantities do agree:
the eigenvalues of ``hcore`` match to ~1e-14, the Cholesky rank is identical,
and the Pb SCF energies reproduce the original 2024 run to twelve digits.

The consequence worth knowing is that the CI expansions (BH's CASCI, N2's
CASSCF) redistribute weight among degenerate configurations, so a truncated
expansion keeps a slightly different set of determinants. That changes the
trial wavefunction quality by a small amount, which means the reference results
have to be regenerated alongside the inputs.
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
                       spin_symm: str, nelec=None, verbose: bool = False) -> None:
    """Write a dense generic hamiltonian in the requested spin symmetry.

    ``spin_symm`` is one of ``closed`` / ``collinear`` / ``noncollinear``:

    - closed: the plain ``X^dag h X`` block, ``nelec = (nup, ndn)``.
    - collinear: the same block stacked twice, giving hcore of shape
      ``(2 norb, norb)`` - identical alpha and beta sectors, but written in the
      layout the collinear code path reads.
    - noncollinear: the spin-blocked hcore of shape ``(2 norb, 2 norb)``, with
      all electrons reported in the alpha slot.
    """
    from afqmctools.hamiltonian.io import write_dense
    from afqmctools.hamiltonian.mol import generate_hamiltonian

    if spin_symm == "noncollinear" and scf_data["hcore"].shape[-1] != 2 * scf_data["norb"]:
        # A scalar hcore promoted into the spinor basis: h -> I_2 (x) h.
        scf_data = dict(scf_data)
        scf_data["hcore"] = np.kron(np.eye(2), scf_data["hcore"])

    walker_type = {"closed": "rhf", "collinear": "uhf", "noncollinear": "ghf"}[spin_symm]

    hcore, chol_vecs, mol_nelec, enuc, X = generate_hamiltonian(
        scf_data,
        chol_cut=chol_cut,
        verbose=verbose,
        walker_type=walker_type,
    )

    if spin_symm == "collinear":
        hcore = np.append(hcore, hcore, axis=0)

    if nelec is None:
        nelec = (sum(mol_nelec), 0) if spin_symm == "noncollinear" else mol_nelec

    write_dense(
        hcore,
        chol_vecs.T,  # want L_{(ik),n}
        nelec,
        nmo=X.shape[-1],
        enuc=enuc,
        real_chol=not np.iscomplexobj(chol_vecs),
        filename=filename,
        ortho=X,
    )


def _to_orbital_basis(scf_data):
    """Projector from the AO basis onto the reference orbital basis.

    ``C^dag S`` maps a set of AO-basis orbitals onto the (orthonormal) basis
    that the hamiltonian was written in, which is what the AFQMC code expects a
    trial wavefunction to be expressed in.
    """
    overlap = scf_data["mol"].intor("int1e_ovlp")
    basis = scf_data["mo_coeff"]
    transform = basis.conj().T @ overlap
    return lambda orbitals: transform @ orbitals


def _write_ghf_nomsd(chkfile: Path, filename: Path, basis_scf_data, nelec) -> None:
    """Write a single-determinant GHF trial, rotated into the reference basis."""
    from afqmctools.wavefunction.mol import write_wfn

    nmo = basis_scf_data["mo_coeff"].shape[-1]
    project = _to_orbital_basis(basis_scf_data)

    with h5.File(chkfile, "r") as fh5:
        mo_coeff = fh5["/scf/mo_coeff"][...]

    nocc = sum(nelec)
    phi = np.zeros((2 * nmo, nocc), dtype=np.complex128)
    phi[:nmo, :nocc] = project(mo_coeff[:nmo, :nocc])
    phi[nmo:, :nocc] = project(mo_coeff[nmo:, :nocc])

    write_wfn(
        filename=filename,
        wfn=(np.array([1.0], dtype=np.complex128), np.array([phi])),
        walker_type="ghf",
        nelec=(nocc, 0),
        norb=nmo,
    )


# ============================================================================
# BH
# ============================================================================

def _unsafe_write_phmsd(filename, wfn, walker_type, nelec, norb,
                        init=None, orbmat=None) -> None:
    """Write a ph-MSD wavefunction bypassing the writer's consistency checks.

    The BH cases deliberately feed the code determinant expansions that the
    user-facing writer would reject - an RHF-shaped ph-MSD whose beta sector
    repeats the alpha one, and a GHF-shaped expansion built from collinear
    occupations. They exist to exercise code paths, not to describe a physical
    trial, so they are written by hand.
    """
    from afqmctools.utils.io import add_group, to_complex
    from afqmctools.utils.slater_types import (_SlaterType, _slater2dims,
                                               _slater_enum_map)
    from afqmctools.wavefunction.common import write_phmsd

    walker_type = _slater_enum_map(walker_type)
    coeffs, occa, occb = wfn
    nalpha, nbeta = nelec
    npol = 2 if walker_type == _SlaterType.NONCOLLINEAR else 1

    with h5.File(filename, "a") as fh5:
        wfn_group = add_group(fh5, "Wavefunction/PHMSD")
        if walker_type == _SlaterType.CLOSED:
            # (nalpha, 0) so that no beta sector is written
            write_phmsd(wfn_group, occa, occb, (nalpha, 0), norb,
                        init=init, orbmat=orbmat)
        else:
            write_phmsd(wfn_group, occa, occb, nelec, norb * npol,
                        init=init, orbmat=orbmat)

        if coeffs.dtype == float:
            coeffs = np.array(coeffs, dtype=np.complex128)
        wfn_group["ci_coeffs"] = to_complex(coeffs)

        if walker_type == _SlaterType.NONCOLLINEAR:
            dims = [norb, nalpha + nbeta, 0, _slater2dims(walker_type), len(coeffs)]
        else:
            dims = [norb, nalpha, nbeta, _slater2dims(walker_type), len(coeffs)]
        wfn_group["dims"] = np.array(dims, dtype=np.int32)


def build_bh(ctx: BuildContext) -> None:
    """BH at a stretched bond: one RHF-basis hamiltonian in three symmetries,
    plus twelve trial wavefunctions covering NOMSD and ph-MSD in each symmetry.

    The CASCI expansion is taken in the *RHF* orbital basis on purpose, so every
    wavefunction here can be paired with every hamiltonian here.
    """
    from pyscf import gto, mcscf, scf

    from afqmctools.utils.pyscf_utils import (ci2chk, ci_wavefunction,
                                              load_from_pyscf_chk_mol,
                                              read_cas_meta)
    from afqmctools.wavefunction.mol import write_wfn, write_wfn_mol

    out, scratch = ctx.out_dir, ctx.scratch
    ci_tol = 0.02
    chol_tol = 5.0e-4
    delta = 1.5 * 1.2344  # Angstrom; 1.5x the equilibrium bond length

    mol = gto.M(
        atom=f"B {delta / 2} 0.0 0.0\nH {-delta / 2} 0.0 0.0",
        basis="ccpvdz",
        spin=0,
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

    rhf_data = load_from_pyscf_chk_mol(rhf_chk)
    _write_hamiltonian(rhf_data, out / "afqmc_H_rhf_closed.h5", chol_tol,
                       spin_symm="closed", verbose=ctx.verbose)
    _write_hamiltonian(rhf_data, out / "afqmc_H_rhf_collinear.h5", chol_tol,
                       spin_symm="collinear", verbose=ctx.verbose)
    _write_hamiltonian(rhf_data, out / "afqmc_H_rhf_noncollinear.h5", chol_tol,
                       spin_symm="noncollinear", verbose=ctx.verbose)

    write_wfn_mol(scf_data=rhf_data, filename=out / "afqmc_rhf_nomsd.h5")

    # --- CASCI in the RHF basis -------------------------------------------
    # CASCI rather than CASSCF so the orbitals stay exactly the RHF ones.
    mc = mcscf.CASCI(mf, 8, 4)
    mc.chkfile = str(rhf_chk)
    mc.run()
    mcscf.chkfile.dump_mcscf(mc, str(rhf_chk))
    ci2chk(rhf_chk, mc.ci)

    cas_meta = read_cas_meta(rhf_chk)
    ncas, ncore = cas_meta["ncas"], cas_meta["ncore"]
    ci, occa, occb = ci_wavefunction(
        ciab=cas_meta["ci"],
        norb=ncas,
        nelec=[n - ncore for n in nelec],
        ncore=ncore,
        tol=ci_tol,
    )
    print(f"    number of determinants: {len(ci)}", flush=True)
    ci = np.array(ci, dtype=np.complex128)

    # ph-MSD, collinear: the honest form of the expansion.
    write_wfn(out / "afqmc_casci_uhf_phmsd.h5", (ci, occa, occb), "uhf", nelec, nmo)
    write_wfn(out / "afqmc_casci_uhf_1phmsd.h5",
              (ci[:1], occa[:1], occb[:1]), "uhf", nelec, nmo)

    # ph-MSD, noncollinear: beta occupations folded into the spinor index.
    occ_noco = [np.append(oa, ob + nmo) for oa, ob in zip(occa, occb)]
    empty = [np.empty_like(occ_noco[0]) for _ in occ_noco]
    _unsafe_write_phmsd(out / "afqmc_casci_ghf_phmsd.h5",
                        (ci, occ_noco, empty),
                        walker_type="ghf", nelec=(sum(nelec), 0), norb=nmo)
    _unsafe_write_phmsd(out / "afqmc_casci_ghf_1phmsd.h5",
                        (ci[:1], occ_noco[:1], empty[:1]),
                        walker_type="ghf", nelec=(sum(nelec), 0), norb=nmo)

    # ph-MSD, closed: an RHF-shaped expansion for the closed walker path.
    _unsafe_write_phmsd(out / "afqmc_casci_rhf_phmsd.h5",
                        (ci, occa, occb.copy()),
                        walker_type="rhf", nelec=nelec, norb=nmo)
    _unsafe_write_phmsd(out / "afqmc_casci_rhf_1phmsd.h5",
                        (ci[:1], occa[:1], occb.copy()[:1]),
                        walker_type="rhf", nelec=nelec, norb=nmo)

    # The same expansion as NOMSD. In the RHF basis every determinant is a
    # column selection from the identity, so the orbital matrices are exact.
    identity = np.eye(nmo)

    nomsd_collinear = []
    for oa, ob in zip(occa, occb, strict=True):
        phi = np.zeros((nmo, sum(nelec)), dtype=np.complex128)
        phi[:, :na] = identity[:, oa]
        phi[:, na:] = identity[:, ob]
        nomsd_collinear.append(phi)
    write_wfn(filename=out / "afqmc_casci_uhf_nomsd.h5",
              wfn=(ci, np.array(nomsd_collinear)),
              walker_type="uhf", nelec=nelec, norb=nmo)

    nomsd_noncollinear = []
    for oa, ob in zip(occa, occb, strict=True):
        phi = np.zeros((2 * nmo, sum(nelec)), dtype=np.complex128)
        phi[:nmo, :na] = identity[:, oa]
        phi[nmo:, na:] = identity[:, ob]
        nomsd_noncollinear.append(phi)
    write_wfn(filename=out / "afqmc_casci_ghf_nomsd.h5",
              wfn=(ci, np.array(nomsd_noncollinear)),
              walker_type="ghf", nelec=nelec, norb=nmo)

    nomsd_closed = []
    for oa in occa:
        phi = np.zeros((nmo, na), dtype=np.complex128)
        phi[:, :na] = identity[:, oa]
        nomsd_closed.append(phi)
    write_wfn(filename=out / "afqmc_casci_rhf_nomsd.h5",
              wfn=(ci, np.array(nomsd_closed)),
              walker_type="rhf", nelec=nelec, norb=nmo)

    # --- UHF and GHF trials, expressed in the RHF basis --------------------
    uhf_chk = scratch / "uhf.chk"
    mf = scf.UHF(mol=mol).newton()
    mf.chkfile = str(uhf_chk)
    mf.kernel()

    write_wfn_mol(scf_data=load_from_pyscf_chk_mol(uhf_chk),
                  filename=out / "afqmc_uhf_nomsd.h5",
                  basis_scf_data=rhf_data)

    # Same UHF trial, but started from the RHF determinant: exercises the
    # separate initial-walker path.
    rhf_initial = np.zeros((nmo, na), dtype=np.complex128)
    rhf_initial[:na, :na] = np.eye(na)
    write_wfn_mol(scf_data=load_from_pyscf_chk_mol(uhf_chk),
                  filename=out / "afqmc_uhf_nomsd_init_rhf.h5",
                  basis_scf_data=rhf_data,
                  init=[rhf_initial, rhf_initial])

    ghf_chk = scratch / "ghf.chk"
    mf = mf.to_ghf()
    dm0 = mf.make_rdm1()
    mf = mf.newton()
    mf.chkfile = str(ghf_chk)
    mf.kernel(dm0=dm0)

    write_wfn_mol(scf_data=load_from_pyscf_chk_mol(ghf_chk),
                  filename=out / "afqmc_ghf_nomsd.h5",
                  basis_scf_data=rhf_data)


# ============================================================================
# N2
# ============================================================================

def build_n2(ctx: BuildContext) -> None:
    """Stretched N2 (3.0 Bohr) with a CASSCF(12o,6e) reference.

    Both the orbital basis and the ph-MSD trial come from the same CASSCF run,
    so this is the case that exercises a genuine multi-determinant trial.
    """
    from pyscf import gto, mcscf, scf

    from afqmctools.hamiltonian.io import write_to_hdf5
    from afqmctools.utils.pyscf_utils import load_from_pyscf_chk_mol
    from afqmctools.wavefunction.mol import write_cas_wfn

    out, scratch = ctx.out_dir, ctx.scratch
    delta = 3.0  # Bohr

    mol = gto.M(
        atom=f"N 0. 0. {delta / 2}\nN 0. 0. -{delta / 2}",
        basis="ccpvdz",
        unit="Bohr",
        verbose=_pyscf_verbosity(ctx),
    )

    casscf_chk = scratch / "rhf_casscf_chkfile.h5"
    rhf = scf.RHF(mol)
    rhf.chkfile = str(casscf_chk)
    rhf.run()

    mc = mcscf.CASSCF(rhf, 12, 6).run()

    # write_to_hdf5 wants something with a dtype, and mc.ncore / mc.ncas are
    # plain Python ints.
    with h5.File(casscf_chk, "a") as fh5:
        write_to_hdf5(fh5, "mcscf/ci", data=np.asarray(mc.ci))
        write_to_hdf5(fh5, "mcscf/ncore", data=np.asarray(mc.ncore))
        write_to_hdf5(fh5, "mcscf/ncas", data=np.asarray(mc.ncas))

    write_cas_wfn(
        mol=mol,
        cas_chkfile=casscf_chk,
        tol_trunc=1.0e-4,
        outname=out / "cas_wfn.h5",
        max_det=50,
    )

    # The hamiltonian is written in the CASSCF natural orbital basis.
    basis_scf_data = load_from_pyscf_chk_mol(chkfile=casscf_chk, base="mcscf")
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

    from afqmctools.utils.pyscf_utils import load_from_pyscf_chk_mol
    from afqmctools.wavefunction.mol import write_wfn_mol

    out, scratch = ctx.out_dir, ctx.scratch

    mol = gto.M(atom="Li 0. 0. 0.", basis="ccpvdz", spin=3,
                verbose=_pyscf_verbosity(ctx))

    rohf_chk = scratch / "rohf.chk"
    mf = scf.ROHF(mol).newton()
    mf.chkfile = str(rohf_chk)
    mf.kernel()

    scf_data = load_from_pyscf_chk_mol(rohf_chk, "scf")
    _write_hamiltonian(scf_data, out / "hamil_closed.h5", 1e-5,
                       spin_symm="closed", verbose=ctx.verbose)
    write_wfn_mol(scf_data=scf_data, filename=out / "rohf_nomsd_polarized.h5")


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

    from afqmctools.utils.pyscf_utils import load_from_pyscf_chk_mol
    from afqmctools.wavefunction.mol import write_wfn_mol

    out, scratch = ctx.out_dir, ctx.scratch
    chol_tol = 5e-4

    mol = gto.M(
        atom="Pb 0. 0. 0.",
        basis="ccpvdzpp",
        ecp=PB_ECP_SOC,
        charge=-1,
        spin=3,
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

    mf = scf.GHF(mol)
    mf.chkfile = str(ghf_soc_chk)
    mf.with_soc = True
    mf.kernel()

    basis_data = load_from_pyscf_chk_mol(rohf_chk, "scf")

    write_wfn_mol(scf_data=load_from_pyscf_chk_mol(uhf_chk),
                  filename=out / "afqmc_uhf_nomsd.h5",
                  basis_scf_data=basis_data)

    # Spin-free: promote the scalar hcore into the spinor basis.
    _write_hamiltonian(basis_data,
                       out / "afqmc_H_rhf_basis_noncollinear_sf.h5", chol_tol,
                       spin_symm="noncollinear", verbose=ctx.verbose)

    # Spin-orbit: hcore already comes back as a complex (2 norb, 2 norb) block.
    soc_data = load_from_pyscf_chk_mol(rohf_chk, "scf", soc_type="ecp")
    _write_hamiltonian(soc_data,
                       out / "afqmc_H_rhf_basis_noncollinear_soc.h5", chol_tol,
                       spin_symm="noncollinear", verbose=ctx.verbose)

    _write_ghf_nomsd(ghf_chk, out / "afqmc_ghf_sf_nomsd.h5", basis_data, nelec)
    _write_ghf_nomsd(ghf_soc_chk, out / "afqmc_ghf_soc_nomsd.h5", basis_data, nelec)


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
