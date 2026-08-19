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
Helpers shared by the input recipes.

Nothing in here knows about any particular system; it is the plumbing that
:mod:`input_recipes` and ``make_inputs.py`` build on - external-executable
resolution, subprocess launching, and HDF5 comparison for ``--check``.
"""

import os
import shlex
import shutil
import subprocess as sp
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import h5py as h5
import numpy as np

ASSETS = Path(__file__).resolve().parent / "assets"


# ============================================================================
# External executables (Quantum ESPRESSO, CoQui)
# ============================================================================

class MissingExternalTool(RuntimeError):
    """Raised when a recipe needs an external code that is not configured."""


@dataclass
class ExternalTools:
    """Resolved paths to the non-Python codes some recipes drive.

    Everything is taken from the environment so that the recipes stay free of
    site-specific paths:

    ``MPIEXEC``
        launcher prefix, e.g. ``"mpirun -n 8"`` or ``"srun -n 8"``. Empty means
        run the executable directly. ``QE_MPIEXEC`` and ``COQUI_MPIEXEC``
        override it for the respective code; the two codes do not want the same
        rank count.
    ``QE_BIN_DIR``
        directory holding ``pw.x`` and ``pw2coqui.x``. Individual overrides
        ``PW_X`` and ``PW2COQUI_X`` win over it.
    ``COQUI_EXEC``
        the CoQui (formerly ``aimbes``) executable, or ``COQUI_BIN_DIR``.
    ``QE_ENV_SCRIPT`` / ``COQUI_ENV_SCRIPT``
        optional shell scripts sourced immediately before the corresponding
        executable runs. Quantum ESPRESSO and CoQui are routinely built against
        different toolchains - at Flatiron they sit in different module trees,
        which cannot both be loaded in one shell - so each gets its own
        environment rather than inheriting this process's.
    """

    mpiexec: List[str] = field(default_factory=list)
    qe_mpiexec: List[str] = field(default_factory=list)
    coqui_mpiexec: List[str] = field(default_factory=list)
    pw_x: Optional[Path] = None
    pw2coqui_x: Optional[Path] = None
    coqui: Optional[Path] = None
    qe_env: Optional[Path] = None
    coqui_env: Optional[Path] = None

    @classmethod
    def from_env(cls, env=None) -> "ExternalTools":
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

        def _script(name: str) -> Optional[Path]:
            return Path(env[name]) if env.get(name) else None

        default_launcher = env.get("MPIEXEC", "")
        return cls(
            mpiexec=shlex.split(default_launcher),
            qe_mpiexec=shlex.split(env.get("QE_MPIEXEC", default_launcher)),
            coqui_mpiexec=shlex.split(env.get("COQUI_MPIEXEC", default_launcher)),
            pw_x=_find("PW_X", "pw.x", "QE_BIN_DIR"),
            pw2coqui_x=_find("PW2COQUI_X", "pw2coqui.x", "QE_BIN_DIR"),
            coqui=_find("COQUI_EXEC", "coqui", "COQUI_BIN_DIR"),
            qe_env=_script("QE_ENV_SCRIPT"),
            coqui_env=_script("COQUI_ENV_SCRIPT"),
        )

    def missing(self, required: Sequence[str]) -> List[str]:
        """Names of the required tools that could not be resolved."""
        return [name for name in required if getattr(self, name, None) is None]

    def require(self, *names: str) -> None:
        missing = self.missing(names)
        if missing:
            raise MissingExternalTool(
                "could not resolve: " + ", ".join(missing)
                + ". Set QE_BIN_DIR / COQUI_EXEC (see --help) or put them on PATH."
            )


def run_external(exe: Path, args: Sequence[str], cwd: Path, log: Path,
                 launcher: Sequence[str] = (), env_script: Optional[Path] = None,
                 stdin: Optional[Path] = None,
                 timeout: Optional[float] = None) -> None:
    """Run ``exe`` under ``launcher``, teeing output to ``log``.

    ``launcher`` is the MPI prefix, already split; pass an empty sequence to run
    serially. With ``env_script`` the command goes through a login shell that
    sources it first, so the executable gets the toolchain it was built against.

    Raises ``sp.CalledProcessError`` on a non-zero exit so a broken external
    step stops the recipe instead of leaving a half-written HDF5 file behind.
    """
    argv = [*launcher, str(exe), *[str(a) for a in args]]
    printable = shlex.join(argv)

    if env_script is not None:
        # `module` is a shell function, so this needs a login shell.
        cmd = ["bash", "-lc", f"set -e; source {shlex.quote(str(env_script))}; "
                              f"exec {printable}"]
        printable = f"source {env_script} && {printable}"
    else:
        cmd = argv

    print(f"    $ {printable}  (in {cwd})", flush=True)
    with open(log, "w") as out:
        infile = open(stdin, "r") if stdin is not None else None
        try:
            sp.run(cmd, cwd=cwd, stdout=out, stderr=sp.STDOUT, stdin=infile,
                   check=True, timeout=timeout)
        finally:
            if infile is not None:
                infile.close()


# ============================================================================
# HDF5 utilities
# ============================================================================

def dataset_paths(fh5: h5.File) -> List[str]:
    """Every dataset path in the file, sorted."""
    found: List[str] = []
    fh5.visititems(lambda name, obj: found.append(name)
                   if isinstance(obj, h5.Dataset) else None)
    return sorted(found)


def copy_groups(src: Path, dest: Path, groups: Sequence[str]) -> List[str]:
    """Copy ``groups`` from ``src`` into ``dest``; returns the ones copied.

    Used to carry hand-made data (the C++ unit tests' ``TEST_RESULTS``) across a
    regeneration of the file that holds it. Missing groups are skipped quietly -
    the caller decides whether that matters.
    """
    copied: List[str] = []
    with h5.File(src, "r") as fin, h5.File(dest, "a") as fout:
        for group in groups:
            if group not in fin:
                continue
            if group in fout:
                del fout[group]
            fin.copy(group, fout, name=group)
            copied.append(group)
    return copied


def compare_h5(reference: Path, candidate: Path,
               rtol: float = 1e-8, atol: float = 1e-10) -> List[str]:
    """Describe how ``candidate`` differs from ``reference``.

    Returns an empty list when they agree. The comparison is numeric rather than
    byte-wise on purpose: HDF5 layout, chunk order and library version all move
    the bytes around without changing the physics, and the committed inputs were
    themselves written by several different afqmctools versions.
    """
    diffs: List[str] = []
    with h5.File(reference, "r") as fref, h5.File(candidate, "r") as fcan:
        ref_keys = dataset_paths(fref)
        can_keys = dataset_paths(fcan)

        for key in sorted(set(ref_keys) - set(can_keys)):
            diffs.append(f"missing dataset {key}")
        for key in sorted(set(can_keys) - set(ref_keys)):
            diffs.append(f"unexpected dataset {key}")

        for key in sorted(set(ref_keys) & set(can_keys)):
            a, b = fref[key][()], fcan[key][()]
            a, b = np.asarray(a), np.asarray(b)
            if a.shape != b.shape:
                diffs.append(f"{key}: shape {a.shape} != {b.shape}")
                continue
            if a.dtype.kind in "SUO" or b.dtype.kind in "SUO":
                if not np.array_equal(a, b):
                    diffs.append(f"{key}: {a!r} != {b!r}")
                continue
            if not np.allclose(a, b, rtol=rtol, atol=atol, equal_nan=True):
                worst = np.max(np.abs(a - b)) if a.size else float("nan")
                diffs.append(f"{key}: max abs difference {worst:.3e}")
    return diffs


def describe_tree(root: Path) -> Dict[str, int]:
    """``{relative path: size in bytes}`` for every file under ``root``."""
    return {str(p.relative_to(root)): p.stat().st_size
            for p in sorted(root.rglob("*")) if p.is_file()}
