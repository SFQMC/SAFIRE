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
Rebuild the AFQMC input files under `tests/functional/afqmc_inputs` and report
how the result differs from what is committed.

Each system directory has one recipe in `input_recipes/`. This script schedules
them, builds into a directory of its own, and diffs. It never writes into the
committed tree: a rebuild is not byte-identical to what is there, and accepting
one obliges regenerating the reference results as well, so installing it is left
as a deliberate `cp -r` that this script prints for you.

    python tests/functional/make_inputs.py --list
    python tests/functional/make_inputs.py hubbard_kanamori
    python tests/functional/make_inputs.py all --into /scratch/inputs

Exit status is 0 when every recipe reproduced its committed files, 2 when some
differed, and 1 when one failed outright. Differences are normal - see
"Do not expect the files back byte for byte" in `tests/functional/README.md`
for how to tell an orbital-gauge rotation from a real problem, and what the
recipes need installed.
"""

import argparse
import shutil
import sys
import tempfile
import traceback
from pathlib import Path
from time import perf_counter

sys.path.insert(0, str(Path(__file__).resolve().parent))

from input_recipes import (INPUTS_ROOT, BuildContext, MissingExternalTool,  # noqa: E402
                           Recipe, build_recipes, unclaimed_files)
from input_recipes._common import compare_h5, missing_tools  # noqa: E402
from input_recipes.solids import resolve_external_tools  # noqa: E402

RTOL = 1e-8
ATOL = 1e-10


def _print_list(recipes, tools, reference: Path) -> None:
    print("Recipes:\n")
    for key, recipe in recipes.items():
        missing = missing_tools(tools, recipe.external)
        status = f"needs {', '.join(missing)}" if missing else "ready"
        print(f"  {key:<22} [{status}]")
        print(f"      {recipe.description}")
        print(f"      -> afqmc_inputs/{recipe.data_dir}/  "
              f"({len(recipe.produces)} file{'s' if len(recipe.produces) != 1 else ''})")
        if recipe.notes:
            print(f"      note: {recipe.notes}")
        print()

    orphans = unclaimed_files(list(recipes.values()), reference)
    if orphans:
        print("Present in the inputs tree but produced by no recipe:")
        for name in orphans:
            print(f"  {name}")
        print()


def build_recipe(recipe: Recipe, build_root: Path, scratch_root: Path,
                 reference: Path, tools: dict, verbose: bool) -> list:
    """Build one recipe into ``build_root`` and diff it against ``reference``.

    Returns the differences, empty when the rebuild reproduced the committed
    files. ``build_root`` is never the committed tree, so the recipe always
    writes into an empty directory. That matters because most of the afqmctools
    writers append rather than truncate - ``write_dense`` and the ``write_wfn``
    family open their files ``'a'`` (which is how the Rashba recipe puts a
    hamiltonian and a trial in one file); only ``write_model_hamiltonian``
    opens ``'w'``. Onto a stale file the appending writers would merge into the
    old datasets instead of replacing them.
    """
    out_dir = build_root / recipe.data_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    scratch = scratch_root / recipe.key
    scratch.mkdir(parents=True, exist_ok=True)

    recipe.build(BuildContext(out_dir=out_dir, scratch=scratch,
                              tools=tools, verbose=verbose))

    absent = [name for name in recipe.produces if not (out_dir / name).exists()]
    if absent:
        raise RuntimeError(f"recipe finished without writing: {', '.join(absent)}")

    diffs = []
    for name in recipe.produces:
        committed = reference / recipe.data_dir / name
        if not committed.exists():
            diffs.append(f"{name}: not present in {reference}")
            continue
        diffs += [f"{name}: {line}"
                  for line in compare_h5(committed, out_dir / name,
                                         rtol=RTOL, atol=ATOL)]
    return diffs


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("recipe", nargs="*", help="recipe key(s), or 'all'")
    parser.add_argument("--into", type=Path,
                        help="build here and keep it, with intermediate files "
                             "under <dir>/_scratch; a temporary directory is "
                             "used and discarded by default")
    parser.add_argument("--reference", type=Path, default=INPUTS_ROOT,
                        help="inputs tree to compare against (default: the "
                             "committed tests/functional/afqmc_inputs)")
    parser.add_argument("--list", action="store_true",
                        help="list the recipes and exit")
    parser.add_argument("--dry-run", action="store_true",
                        help="report what would be built and exit")
    parser.add_argument("-v", "--verbose", action="store_true",
                        help="verbose output from pyscf, afqmctools and AutoHF")
    args = parser.parse_args(argv)

    recipes = build_recipes()
    tools = resolve_external_tools()

    if args.list:
        _print_list(recipes, tools, args.reference)
        return 0

    if not args.recipe:
        parser.error("give one or more recipe keys, 'all', or --list")

    if args.recipe == ["all"]:
        selected = list(recipes.values())
    else:
        unknown = [key for key in args.recipe if key not in recipes]
        if unknown:
            parser.error(f"unknown recipe(s): {', '.join(unknown)}. "
                         f"Known: {', '.join(recipes)}")
        selected = [recipes[key] for key in args.recipe]

    if args.dry_run:
        for recipe in selected:
            missing = missing_tools(tools, recipe.external)
            mark = f"  [skipped: needs {', '.join(missing)}]" if missing else ""
            print(f"{recipe.key} -> {recipe.data_dir}/{mark}")
            for name in recipe.produces:
                print(f"    {name}")
        return 0

    reference = args.reference.resolve()
    temporary = None
    if args.into is None:
        temporary = Path(tempfile.mkdtemp(prefix="afqmc_inputs_"))
        build_root = temporary
    else:
        build_root = args.into.resolve()
        # The committed tree is what both test suites read, and the C++ unit
        # tests compile a path to it. Building into it would leave the tests
        # reading files no reference has been recorded against.
        if build_root == reference or build_root.is_relative_to(reference):
            parser.error(f"--into {build_root} is inside the inputs tree being "
                         "compared against; build somewhere else and copy the "
                         "result in yourself once you have read the diff")
        build_root.mkdir(parents=True, exist_ok=True)

    scratch_root = build_root / "_scratch"
    scratch_root.mkdir(parents=True, exist_ok=True)

    # Two recipes can share a data_dir (the diamond pair), so clear each one
    # once for the whole selection rather than per recipe.
    for data_dir in dict.fromkeys(recipe.data_dir for recipe in selected):
        shutil.rmtree(build_root / data_dir, ignore_errors=True)

    print(f"reference : {reference}")
    print(f"building  : {build_root}")

    identical = differs = failed = skipped = 0
    unclean = []
    try:
        for recipe in selected:
            missing = missing_tools(tools, recipe.external)
            if missing:
                print(f"\n--- {recipe.key}\n  RESULT: SKIP "
                      f"(needs {', '.join(missing)})")
                skipped += 1
                continue

            print(f"\n--- {recipe.key} -> {build_root / recipe.data_dir}",
                  flush=True)
            start = perf_counter()
            try:
                diffs = build_recipe(recipe, build_root, scratch_root,
                                     reference, tools, args.verbose)
            except MissingExternalTool as exc:
                # A recipe can also ask for a tool it did not declare.
                print(f"  RESULT: SKIP ({exc})")
                skipped += 1
                continue
            except Exception as exc:  # noqa: BLE001 - one bad recipe must not stop the rest
                traceback.print_exc()
                print(f"  RESULT: FAILED  {type(exc).__name__}: {exc}")
                failed += 1
                unclean.append(recipe.key)
                continue

            elapsed = perf_counter() - start
            if not diffs:
                print(f"  RESULT: IDENTICAL ({elapsed:.1f}s)")
                identical += 1
                continue

            print(f"  RESULT: DIFFERS ({elapsed:.1f}s)")
            for diff in diffs:
                print(f"    {diff}")
            if recipe.notes:
                print(f"    note: {recipe.notes}")
            if temporary is None:
                print(f"  to accept: cp -r {build_root / recipe.data_dir}/. "
                      f"{reference / recipe.data_dir}/")
            differs += 1
            unclean.append(recipe.key)
    finally:
        if temporary is not None:
            shutil.rmtree(temporary, ignore_errors=True)

    print(f"\n==== {identical} identical, {differs} differs, {failed} failed, "
          f"{skipped} skipped ====")
    if unclean:
        print("  " + ", ".join(unclean))
    if differs:
        print('\nDifferences are expected. See "Do not expect the files back '
              'byte for byte"\nin tests/functional/README.md before acting on '
              "them.")
        if temporary is not None:
            print("Rerun with --into DIR to keep the rebuilt files and the "
                  "logs behind them.")

    if failed:
        return 1
    return 2 if differs else 0


if __name__ == "__main__":
    sys.exit(main())
