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

r"""
Rebuild the AFQMC input files under `tests/functional/afqmc_inputs`.

Each system directory has one recipe in `input_recipes/`. This script runs them,
writing into a directory you name. It never writes into the committed tree: a
rebuild is not byte-identical to what is there, and accepting one obliges
regenerating the reference results too, so installing it is a deliberate
`cp -r` you make after looking at what came out.

    python tests/functional/make_inputs.py --list
    python tests/functional/make_inputs.py all --into /scratch/inputs

See `tests/functional/README.md` for what the recipes need installed.
"""

import argparse
import shutil
import sys
import traceback
from pathlib import Path
from time import perf_counter

sys.path.insert(0, str(Path(__file__).resolve().parent))

from input_recipes import (INPUTS_ROOT, BuildContext, MissingExternalTool,  # noqa: E402
                           Recipe, build_recipes)
from input_recipes._common import missing_tools  # noqa: E402
from input_recipes.solids import resolve_external_tools  # noqa: E402


def build_recipe(recipe: Recipe, build_root: Path, scratch_root: Path,
                 tools: dict, verbose: bool) -> None:
    """Build one recipe into ``build_root``.

    The recipe always writes into an empty directory. That matters because most
    of the afqmctools writers append rather than truncate - ``write_dense`` and
    the ``write_wfn`` family open their files ``'a'`` (which is how the Rashba
    recipe puts a hamiltonian and a trial in one file); only
    ``write_model_hamiltonian`` opens ``'w'``. Onto a stale file the appending
    writers would merge into the old datasets instead of replacing them.
    """
    out_dir = build_root / recipe.data_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    scratch = scratch_root / recipe.key
    scratch.mkdir(parents=True, exist_ok=True)

    recipe.build(BuildContext(out_dir=out_dir, scratch=scratch,
                              tools=tools, verbose=verbose))


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("recipe", nargs="*", help="recipe key(s), or 'all'")
    parser.add_argument("--into", type=Path,
                        help="directory to build into; intermediate files land "
                             "under <dir>/_scratch")
    parser.add_argument("--list", action="store_true",
                        help="list the recipes and exit")
    parser.add_argument("-v", "--verbose", action="store_true",
                        help="verbose output from pyscf, afqmctools and AutoHF")
    args = parser.parse_args(argv)

    recipes = build_recipes()
    tools = resolve_external_tools()

    if args.list:
        for key, recipe in recipes.items():
            missing = missing_tools(tools, recipe.external)
            status = f"needs {', '.join(missing)}" if missing else "ready"
            print(f"  {key:<22} -> afqmc_inputs/{recipe.data_dir}/  [{status}]")
        return 0

    if not args.recipe:
        parser.error("give one or more recipe keys, 'all', or --list")
    if args.into is None:
        parser.error("--into DIR is required: this tool never writes into the "
                     "committed inputs tree")

    if args.recipe == ["all"]:
        selected = list(recipes.values())
    else:
        unknown = [key for key in args.recipe if key not in recipes]
        if unknown:
            parser.error(f"unknown recipe(s): {', '.join(unknown)}. "
                         f"Known: {', '.join(recipes)}")
        selected = [recipes[key] for key in args.recipe]

    build_root = args.into.resolve()
    # The committed tree is what both test suites read, and the C++ unit tests
    # compile a path to it. Building into it would leave the tests reading files
    # no reference has been recorded against.
    if build_root == INPUTS_ROOT or build_root.is_relative_to(INPUTS_ROOT):
        parser.error(f"--into {build_root} is inside the committed inputs tree; "
                     "build somewhere else and copy the result in yourself")
    build_root.mkdir(parents=True, exist_ok=True)

    scratch_root = build_root / "_scratch"
    scratch_root.mkdir(parents=True, exist_ok=True)

    # Two recipes can share a data_dir (the diamond pair), so clear each one
    # once for the whole selection rather than per recipe.
    for data_dir in dict.fromkeys(recipe.data_dir for recipe in selected):
        shutil.rmtree(build_root / data_dir, ignore_errors=True)

    print(f"building in {build_root}")

    built = skipped = failed = 0
    for recipe in selected:
        missing = missing_tools(tools, recipe.external)
        if missing:
            print(f"\n--- {recipe.key}\n  SKIP (needs {', '.join(missing)})")
            skipped += 1
            continue

        print(f"\n--- {recipe.key} -> {build_root / recipe.data_dir}", flush=True)
        start = perf_counter()
        try:
            build_recipe(recipe, build_root, scratch_root, tools, args.verbose)
        except MissingExternalTool as exc:
            # A recipe can also ask for a tool it did not declare.
            print(f"  SKIP ({exc})")
            skipped += 1
        except Exception as exc:  # noqa: BLE001 - one bad recipe must not stop the rest
            traceback.print_exc()
            print(f"  FAILED  {type(exc).__name__}: {exc}")
            failed += 1
        else:
            print(f"  OK ({perf_counter() - start:.1f}s)")
            built += 1

    print(f"\n==== {built} built, {failed} failed, {skipped} skipped ====")
    if built:
        print(f"To install: cp -r {build_root}/<system>/. {INPUTS_ROOT}/<system>/")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
