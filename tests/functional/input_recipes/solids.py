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
Recipes for the diamond solid, downfolded from a plane-wave DFT calculation.

Unlike everything else in this tool these two recipes drive external codes:

    pw.x  ->  pw2coqui.x  ->  coqui  ->  afqmctools

Quantum ESPRESSO produces the PBE mean field, ``pw2coqui.x`` exports it, CoQui
downfolds it into a Cholesky- or THC-factorised hamiltonian together with the
mean-field wavefunction, and two small afqmctools steps derive the closed and
noncollinear trials from that wavefunction.

Neither external code is a Python dependency, so both recipes are skipped
unless the executables resolve. :func:`resolve_external_tools` below documents
the environment variables; ``make_inputs.py --list`` reports what is missing.

Both codes are run as ordinary serial commands. These are small cases, and if
you want them launched any other way, or need a particular toolchain on PATH,
set that up in your shell before invoking the tool.

The QE inputs live in ``assets/C_diamond`` - a template ``scf.in`` (the two
recipes differ only in the k-grid), the ``pw2coqui.x`` namelist and the ONCV
PBE carbon pseudopotential. The CoQui TOML is written at build time because it
has to carry absolute paths into the scratch directory.
"""

import os
import shlex
import shutil
import subprocess as sp
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import numpy as np

from . import BuildContext, Recipe
from ._common import ASSETS, require_tools

DIAMOND_ASSETS = ASSETS / "C_diamond"


# ============================================================================
# Finding and running the external codes
# ============================================================================

def resolve_external_tools(env=None) -> Dict[str, Optional[Path]]:
    """Locate Quantum ESPRESSO and CoQui from the environment.

    These are the only recipes in the tool that shell out, so this lives here
    rather than in the shared plumbing. The paths come from the environment so
    that the recipes stay free of site-specific ones:

    ``QE_BIN_DIR``
        directory holding ``pw.x`` and ``pw2coqui.x``. Individual overrides
        ``PW_X`` and ``PW2COQUI_X`` win over it.
    ``COQUI_EXEC``
        the CoQui (formerly ``aimbes``) executable, or ``COQUI_BIN_DIR``.

    Anything already on PATH is found without being named. Entries are ``None``
    when they could not be found, which is what ``make_inputs.py`` skips a
    recipe on.
    """
    env = os.environ if env is None else env

    def _find(explicit: str, name: str, bindir: str) -> Optional[Path]:
        if env.get(explicit):
            return Path(env[explicit])
        if env.get(bindir):
            candidate = Path(env[bindir]) / name
            if candidate.exists():
                return candidate
        found = shutil.which(name)
        return Path(found) if found else None

    return {
        "pw_x": _find("PW_X", "pw.x", "QE_BIN_DIR"),
        "pw2coqui_x": _find("PW2COQUI_X", "pw2coqui.x", "QE_BIN_DIR"),
        "coqui": _find("COQUI_EXEC", "coqui", "COQUI_BIN_DIR"),
    }


def _run(exe: Path, args: Sequence[str], cwd: Path, log: Path) -> None:
    """Run ``exe`` in ``cwd``, sending its output to ``log``.

    Raises ``sp.CalledProcessError`` on a non-zero exit, so a broken external
    step stops the recipe instead of leaving a half-written HDF5 file behind.
    """
    argv = [str(exe), *(str(a) for a in args)]
    print(f"    $ {shlex.join(argv)}  (in {cwd})", flush=True)
    with open(log, "w") as out:
        sp.run(argv, cwd=cwd, stdout=out, stderr=sp.STDOUT, check=True)


def _run_quantum_espresso(ctx: BuildContext, kgrid: str) -> Path:
    """Run the PBE scf and export it for CoQui. Returns the QE outdir."""
    require_tools(ctx.tools, "pw_x", "pw2coqui_x")

    qe_dir = ctx.scratch / "qe"
    qe_dir.mkdir(parents=True, exist_ok=True)

    shutil.copy(DIAMOND_ASSETS / "C_ONCV_PBE-1.2.upf", qe_dir)
    shutil.copy(DIAMOND_ASSETS / "pw2coqui.in", qe_dir)

    template = (DIAMOND_ASSETS / "scf.in.template").read_text()
    (qe_dir / "scf.in").write_text(template.format(kgrid=kgrid))

    _run(ctx.tools["pw_x"], ["-inp", "scf.in"], cwd=qe_dir, log=qe_dir / "scf.out")
    _run(ctx.tools["pw2coqui_x"], ["-in", "pw2coqui.in"], cwd=qe_dir,
         log=qe_dir / "pw2coqui.out")

    return qe_dir / "OUT"


CHOLESKY_TOML = """\
[mean_field.qe]
name     = "mf"
prefix   = "pwscf"
outdir   = "{outdir}"
nbnd     = 8

[interaction.cholesky]
name        = "eri"
mean_field  = "mf"
output      = "{hamiltonian}"
write_type  = "single"
tol      = 1e-5
ecut     = 50

[hamiltonian]
mean_field  = "mf"
interaction = "eri"
output      = "{hamiltonian}"

[wavefunction.mf]
mean_field  = "mf"
output      = "{wavefunction}"
"""

THC_TOML = """\
[mean_field.qe]
name     = "mf"
prefix   = "pwscf"
outdir   = "{outdir}"
nbnd     = 8

[interaction.thc]
name        = "eri"
mean_field  = "mf"
save        = "thc.h5"
thresh      = 1e-6
ecut        = 50.0

[hamiltonian]
mean_field  = "mf"
interaction = "eri"
output      = "{hamiltonian}"

[wavefunction.mf]
mean_field  = "mf"
output      = "{wavefunction}"
"""


def _run_coqui(ctx: BuildContext, name: str, toml: str, outdir: Path,
               hamiltonian: str, wavefunction: str) -> Path:
    """Run one CoQui downfolding. Returns the working directory."""
    require_tools(ctx.tools, "coqui")

    work = ctx.scratch / name
    work.mkdir(parents=True, exist_ok=True)
    (work / "hamil.toml").write_text(
        toml.format(outdir=outdir, hamiltonian=hamiltonian,
                    wavefunction=wavefunction)
    )

    log = work / "hamil.out"
    try:
        _run(ctx.tools["coqui"], ["--verbosity=2", "--filenames", "hamil.toml"],
             cwd=work, log=log)
    except sp.CalledProcessError as exc:
        raise RuntimeError(f"CoQui exited {exc.returncode}; see {log}.") from exc
    return work


# ============================================================================
# afqmctools post-processing
# ============================================================================

def _write_closed_trial(filename: Path, nelec=(4, 4), norb: int = 8) -> None:
    """The trivial closed-shell trial: the lowest ``nup`` bands, occupied.

    In the downfolded band basis the mean-field determinant *is* a column
    selection from the identity, so this needs no input from CoQui.
    """
    from afqmctools.wavefunction.mol import write_wfn

    orbitals = np.eye(norb)
    write_wfn(
        filename=filename,
        wfn=(np.array([1.0]), np.array([orbitals[:, :nelec[0]]])),
        walker_type="rhf",
        nelec=nelec,
        norb=norb,
    )


def _collinear_to_noncollinear(source: Path, destination: Path) -> None:
    """Re-express a collinear trial in the spinor basis.

    The alpha and beta blocks go on the diagonal of a 2x2 spin structure, which
    gives a noncollinear determinant describing the same state - the point being
    to feed the noncollinear code path a wavefunction with a known answer.
    """
    from afqmctools.utils.types import get_spin_symm_enum
    from afqmctools.wavefunction.converter import read_wavefunction
    from afqmctools.wavefunction.mol import write_wfn

    (coeffs, phi), _psi0, nelec, _spin_symm = read_wavefunction(source)

    nmo = phi.shape[1]
    phi_up = phi[0, :, :nelec[0]]
    phi_down = phi[0, :, nelec[0]:nelec[0] + nelec[1]]

    phi_noncollinear = np.block([
        [phi_up, np.zeros_like(phi_up)],
        [np.zeros_like(phi_down), phi_down],
    ])[np.newaxis, :, :]

    write_wfn(
        destination,
        [coeffs, phi_noncollinear],
        walker_type=get_spin_symm_enum("noncollinear"),
        nelec=nelec,
        norb=nmo,
    )


# ============================================================================
# Recipes
# ============================================================================

def build_diamond(ctx: BuildContext) -> None:
    """Carbon diamond at the gamma point: Cholesky and THC hamiltonians.

    The THC run reuses the same mean field, so both factorisations describe the
    same system and can be compared against each other.
    """
    out = ctx.out_dir
    outdir = _run_quantum_espresso(ctx, kgrid="1 1 1")

    chol = _run_coqui(ctx, "coqui_cholesky", CHOLESKY_TOML, outdir,
                      hamiltonian="ham_chol_1e-5.h5",
                      wavefunction="wfn_mf_pbe.h5")
    thc = _run_coqui(ctx, "coqui_thc", THC_TOML, outdir,
                     hamiltonian="ham_thc_1e-6.h5",
                     wavefunction="wfn_mf_pbe.h5")

    shutil.copy(chol / "ham_chol_1e-5.h5", out / "ham_chol_1e-5.h5")
    shutil.copy(thc / "ham_thc_1e-6.h5", out / "ham_thc_1e-6.h5")
    shutil.copy(chol / "wfn_mf_pbe.h5", out / "wfn_mf_pbe.h5")

    _write_closed_trial(out / "wfn_mf_pbe_closed.h5")
    _collinear_to_noncollinear(out / "wfn_mf_pbe.h5",
                               out / "wfn_mf_pbe_noncollinear.h5")


def build_diamond_2x2x2(ctx: BuildContext) -> None:
    """Carbon diamond on a 2x2x2 k-grid: the k-point Cholesky case."""
    out = ctx.out_dir
    outdir = _run_quantum_espresso(ctx, kgrid="2 2 2")

    chol = _run_coqui(ctx, "coqui_cholesky", CHOLESKY_TOML, outdir,
                      hamiltonian="ham_chol_1e-5.h5",
                      wavefunction="wfn_mf_pbe.h5")

    shutil.copy(chol / "ham_chol_1e-5.h5", out / "ham_2x2x2_chol_1e-5.h5")
    shutil.copy(chol / "wfn_mf_pbe.h5", out / "wfn_mf_2x2x2_pbe.h5")


def recipes() -> List[Recipe]:
    external = ("pw_x", "pw2coqui_x", "coqui")
    return [
        Recipe(key="diamond", data_dir="C_diamond_coqui",
               build=build_diamond, external=external),
        Recipe(key="diamond_2x2x2", data_dir="C_diamond_coqui",
               build=build_diamond_2x2x2, external=external),
    ]
