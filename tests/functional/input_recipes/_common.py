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
:mod:`input_recipes` and ``make_inputs.py`` build on - subprocess launching and
HDF5 comparison. Which external executables exist, and how they are found, is
the business of the family module that needs them (:mod:`solids`).
"""

import shlex
import subprocess as sp
from pathlib import Path
from typing import List, Optional, Sequence

import h5py as h5
import numpy as np

ASSETS = Path(__file__).resolve().parent / "assets"


# ============================================================================
# External executables
# ============================================================================

class MissingExternalTool(RuntimeError):
    """Raised when a recipe needs an external code that is not configured."""


def require_tools(tools: dict, *names: str) -> None:
    """Raise :class:`MissingExternalTool` unless every name resolved."""
    missing = missing_tools(tools, names)
    if missing:
        raise MissingExternalTool(
            "could not resolve: " + ", ".join(missing)
            + ". Set QE_BIN_DIR / COQUI_EXEC (see the README) or put them on PATH."
        )


def missing_tools(tools: dict, required: Sequence[str]) -> List[str]:
    """Names of the required tools that could not be resolved."""
    return [name for name in required if tools.get(name) is None]


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

    Used to graft data that has no generator - the C++ unit tests'
    ``TEST_RESULTS``, checked in under ``assets/`` - into a freshly written
    file. Missing groups are skipped quietly; the caller decides whether that
    matters.
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
