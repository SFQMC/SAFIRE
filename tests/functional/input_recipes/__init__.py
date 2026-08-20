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
Registry of recipes that regenerate ``tests/functional/afqmc_inputs``.

Each :class:`Recipe` owns one system directory and declares the files it writes
there. ``make_inputs.py`` imports :func:`build_recipes` and does the scheduling
and checking; the recipe itself only has to produce the files, into an empty
directory it is handed.

The inputs tree is shared: ``run_functional.py`` reads it through
``functional_cases.py``, and the C++ unit tests read the same directory through
``unit_test_base()`` in ``tests/test_common.hpp``. Several files therefore exist
only for the C++ side and appear in no functional case - they are still covered
here, because the goal is that the whole directory can be rebuilt.

Adding a recipe
---------------
Write a ``build(ctx)`` function taking a :class:`BuildContext`, writing its files
into ``ctx.out_dir``, then register it in :func:`build_recipes`. Keep any
intermediate files (pyscf checkpoints, plane-wave runs) under ``ctx.scratch`` so
that the inputs tree only ever gains the declared outputs.
"""

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, List, Sequence, Tuple

# jax reads this when it is imported, and afqmctools pulls it in indirectly, so
# it has to be set before anything else here. Without it jax claims a cuda
# backend on a machine that has the plugin installed but no visible device and
# then dies. Everything these recipes ask of jax is a seconds-long solve on a
# handful of sites; set JAX_PLATFORMS yourself if you want otherwise.
os.environ.setdefault("JAX_PLATFORMS", "cpu")

from ._common import MissingExternalTool  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
INPUTS_ROOT = ROOT / "afqmc_inputs"


@dataclass
class BuildContext:
    """Everything a recipe is handed when it runs."""

    out_dir: Path          # the system directory the recipe writes into, empty
    scratch: Path          # private scratch space, created before the call
    tools: dict            # resolved external executables, see solids.py
    verbose: bool = False


@dataclass
class Recipe:
    """One system directory's worth of AFQMC inputs.

    Attributes
    ----------
    data_dir
        Directory under ``afqmc_inputs/`` that the recipe fills.
    produces
        Filenames written into ``data_dir``. The runner checks after the fact
        that every one of them appeared.
    build
        ``build(ctx: BuildContext) -> None``.
    external
        Names of the external executables the recipe needs, as keyed by
        ``solids.resolve_external_tools``. A recipe with unresolved tools is
        skipped rather than failed.
    """

    key: str
    data_dir: str
    produces: List[str]
    build: Callable[[BuildContext], None]
    description: str = ""
    external: Tuple[str, ...] = ()
    notes: str = ""

    def out_dir(self, inputs_root: Path) -> Path:
        return inputs_root / self.data_dir

    def targets(self, inputs_root: Path) -> List[Path]:
        out = self.out_dir(inputs_root)
        return [out / name for name in self.produces]


def build_recipes() -> Dict[str, Recipe]:
    """Every recipe, keyed by the name accepted on the command line."""
    # Imported here so that a broken optional dependency in one family (pyscf,
    # say) does not stop the others from being listed or run.
    from . import models, molecules, solids

    recipes: List[Recipe] = []
    recipes += molecules.recipes()
    recipes += models.recipes()
    recipes += solids.recipes()

    by_key: Dict[str, Recipe] = {}
    for recipe in recipes:
        if recipe.key in by_key:
            raise ValueError(f"duplicate recipe key {recipe.key!r}")
        by_key[recipe.key] = recipe
    return by_key


def unclaimed_files(recipes: Sequence[Recipe], inputs_root: Path) -> List[str]:
    """Files present in the inputs tree that no recipe claims to produce."""
    claimed = {str(p.relative_to(inputs_root))
               for r in recipes for p in r.targets(inputs_root)}
    present = {str(p.relative_to(inputs_root))
               for p in inputs_root.rglob("*.h5") if p.is_file()}
    return sorted(present - claimed)


__all__ = [
    "BuildContext",
    "INPUTS_ROOT",
    "MissingExternalTool",
    "Recipe",
    "ROOT",
    "build_recipes",
    "unclaimed_files",
]
