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

Nothing in here knows about any particular system. Which external executables
exist, and how they are found, is the business of the family module that needs
them (:mod:`solids`).
"""

from pathlib import Path
from typing import List, Sequence

import h5py as h5

ASSETS = Path(__file__).resolve().parent / "assets"


class MissingExternalTool(RuntimeError):
    """Raised when a recipe needs an external code that is not configured."""


def missing_tools(tools: dict, required: Sequence[str]) -> List[str]:
    """Names of the required tools that could not be resolved."""
    return [name for name in required if tools.get(name) is None]


def require_tools(tools: dict, *names: str) -> None:
    """Raise :class:`MissingExternalTool` unless every name resolved."""
    missing = missing_tools(tools, names)
    if missing:
        raise MissingExternalTool(
            "could not resolve: " + ", ".join(missing)
            + ". Set QE_BIN_DIR / COQUI_EXEC (see the README) or put them on PATH."
        )


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
