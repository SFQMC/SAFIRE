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
Regenerate the AFQMC input files under `tests/functional/afqmc_inputs`.

Those files - hamiltonians and trial wavefunctions in HDF5 - are the fixed
starting point of both test suites in this repository: `run_functional.py`
reads them through `functional_cases.py`, and the C++ unit tests read the same
directory through `unit_test_base()` in `tests/test_common.hpp`. They are
committed rather than built, so without a tool like this the only record of how
they were made is whatever scratch directory they came out of.

Each system directory has one recipe in `input_recipes/`, declaring the files
it writes and how to write them. This script schedules the recipes, clears the
targets, and can check a rebuild against what is committed.

Usage
-----
List the recipes and say which are runnable here::

    python tests/functional/make_inputs.py --list

Rebuild everything in place (the default output path is the inputs tree)::

    python tests/functional/make_inputs.py all

Rebuild one system somewhere else and keep the intermediate files::

    python tests/functional/make_inputs.py BH \
        --output-path /scratch/inputs --scratch /scratch/work --keep-scratch

Rebuild into a temporary tree and report how it differs from what is committed,
without touching the committed files::

    python tests/functional/make_inputs.py all --check

Requirements
------------
`afqmctools` must be importable - installing `utils` (`pip install -e utils`) is
the usual way, otherwise put `utils` on PYTHONPATH. The molecular recipes need
pyscf; the 4x4 Hubbard UHF trial needs AutoHF (`utils/AutoHF`).

The two diamond recipes drive Quantum ESPRESSO and CoQui and are skipped unless
those resolve, from these environment variables::

    QE_BIN_DIR        directory holding pw.x and pw2coqui.x
                      (or PW_X / PW2COQUI_X individually)
    COQUI_EXEC        the CoQui executable (or COQUI_BIN_DIR)
    MPIEXEC           launcher prefix, e.g. "mpirun -n 8" or "srun -n 8"
    QE_MPIEXEC        launcher for pw.x, overriding MPIEXEC
    COQUI_MPIEXEC     launcher for coqui, overriding MPIEXEC
    QE_ENV_SCRIPT     sourced before each Quantum ESPRESSO step
    COQUI_ENV_SCRIPT  sourced before each CoQui step

The two env scripts are usually both needed: the codes are built against
different toolchains, and at Flatiron they live in module trees that cannot
both be loaded in one shell. Have each script set up its own environment from
scratch (`module purge` first). For example::

    export QE_BIN_DIR=/path/to/q-e/build/bin
    export QE_ENV_SCRIPT=/path/to/q-e/build/env.sh
    export COQUI_EXEC=/path/to/coqui.build/bin/coqui
    export COQUI_ENV_SCRIPT=/path/to/coqui.build/env_2.4.sh
    export COQUI_MPIEXEC="mpirun -n 8"

CoQui wants roughly one rank per irreducible k-point. Quantum ESPRESSO is left
serial by default here because the whole diamond mean field takes under a
second; `pw2coqui.x` is always run serially regardless, since it deadlocks on
more than one rank.

Known gaps
----------
Two pieces of the inputs tree have no generator anywhere in this repository and
so cannot be rebuilt:

- `square_2x2_hubbard_Beta3_nt100/wfn_collinear.h5`, a thermal propagator
  factorisation in a format afqmctools does not write.
- the `TEST_RESULTS` group inside that directory's `ham_collinear.h5`, holding
  the arrays the finite-temperature C++ unit test asserts against. It is copied
  across a regeneration rather than recomputed.

`--list` prints both, and `--check` will tell you whether the regenerated
hamiltonian still matches the one those results were computed from.
"""

import argparse
import shutil
import sys
import tempfile
import traceback
from pathlib import Path
from time import perf_counter
from typing import Dict, List, Optional, Sequence

import h5py as h5

sys.path.insert(0, str(Path(__file__).resolve().parent))

from input_recipes import (INPUTS_ROOT, BuildContext, ExternalTools,  # noqa: E402
                           MissingExternalTool, Recipe, build_recipes,
                           unclaimed_files)
from input_recipes._common import compare_h5, copy_groups  # noqa: E402


# ============================================================================
# Reporting
# ============================================================================

class Outcome:
    BUILT = "built"
    CHECKED = "ok"
    DIFFERS = "differs"
    SKIPPED = "skipped"
    FAILED = "failed"


class Result:
    def __init__(self, key: str, outcome: str, detail: str = "",
                 seconds: float = 0.0, diffs: Optional[List[str]] = None):
        self.key = key
        self.outcome = outcome
        self.detail = detail
        self.seconds = seconds
        self.diffs = diffs or []


def _print_list(recipes: Dict[str, Recipe], tools: ExternalTools,
                inputs_root: Path) -> None:
    print("Recipes:\n")
    for key, recipe in recipes.items():
        missing = tools.missing(recipe.external)
        status = f"needs {', '.join(missing)}" if missing else "ready"
        print(f"  {key:<22} [{status}]")
        print(f"      {recipe.description}")
        print(f"      -> afqmc_inputs/{recipe.data_dir}/  "
              f"({len(recipe.produces)} file{'s' if len(recipe.produces) != 1 else ''})")
        if recipe.notes:
            print(f"      note: {recipe.notes}")
        print()

    orphans = unclaimed_files(list(recipes.values()), inputs_root)
    if orphans:
        print("Present in the inputs tree but produced by no recipe:")
        for name in orphans:
            print(f"  {name}")
        print()


CHECK_LEGEND = """
A recipe reporting DIFFERS is not necessarily broken. Differences fall into
three groups, and only the third is a problem:

  Orbital gauge. Any SCF, CASSCF or variational solve is free to return a
  different rotation within a degenerate shell, or a different sign per
  orbital. `Hamiltonian/X` and everything written in that basis then differ,
  while the physics does not - check that the eigenvalues of `Hamiltonian/hcore`
  and the Cholesky rank still agree.

  Writer changes. `maximum_connectivity` in the model hamiltonians now has a
  floor of 12 in `write_model_hamiltonian`; the committed files predate it and
  store the smaller computed value.

  Everything else. A changed nuclear energy, Cholesky rank, matrix shape or
  electron count means the recipe and the committed file describe different
  systems, and one of them is wrong.
"""


def _print_summary(results: Sequence[Result], check: bool,
                   recipes: Dict[str, Recipe]) -> None:
    print("\n" + "=" * 72)
    for result in results:
        line = f"  {result.key:<22} {result.outcome.upper():<8}"
        if result.seconds:
            line += f" {result.seconds:6.1f}s"
        if result.detail:
            line += f"  {result.detail}"
        print(line)
        for diff in result.diffs:
            print(f"        {diff}")
        note = recipes[result.key].notes
        if note and result.outcome == Outcome.DIFFERS:
            print(f"      note: {note}")

    tally: Dict[str, int] = {}
    for result in results:
        tally[result.outcome] = tally.get(result.outcome, 0) + 1
    print("=" * 72)
    print("  " + ", ".join(f"{count} {name}" for name, count in sorted(tally.items())))

    if check and any(result.outcome == Outcome.DIFFERS for result in results):
        print(CHECK_LEGEND)


# ============================================================================
# Running one recipe
# ============================================================================

def _clear_targets(recipe: Recipe, out_dir: Path, stash: Optional[Path]) -> None:
    """Delete the recipe's outputs, first stashing any preserved groups.

    The afqmctools writers open HDF5 files in append mode, so leaving a stale
    file in place would merge the new datasets into the old ones instead of
    replacing them.
    """
    for name in recipe.produces:
        target = out_dir / name
        if not target.exists():
            continue
        if name in recipe.preserve and stash is not None:
            shutil.copy(target, stash / name)
        target.unlink()


def _restore_preserved(recipe: Recipe, out_dir: Path, stash: Path) -> List[str]:
    """Copy preserved groups back into the freshly written files."""
    restored: List[str] = []
    for name, groups in recipe.preserve.items():
        # _assert_preservable has already established that this exists.
        source = stash / name
        copied = copy_groups(source, out_dir / name, groups)
        restored += [f"{name}:{group}" for group in copied]
    return restored


def _assert_preservable(recipe: Recipe, out_dir: Path) -> None:
    """Refuse to start if a preserved group has nothing to be preserved from.

    Checked before anything is cleared: those groups have no generator, so
    overwriting the file that holds them and only then discovering the problem
    would destroy data.
    """
    for name, groups in recipe.preserve.items():
        target = out_dir / name
        if not target.exists():
            raise RuntimeError(
                f"{recipe.key}: {name} carries group(s) {', '.join(groups)} "
                "that no recipe can regenerate, and there is no existing file "
                f"at {target} to copy them from. Restore the committed file "
                "before rebuilding, or build somewhere else with --output-path."
            )
        with h5.File(target, "r") as fh5:
            absent = [group for group in groups if group not in fh5]
        if absent:
            raise RuntimeError(
                f"{recipe.key}: {target} is missing group(s) "
                f"{', '.join(absent)}, which cannot be regenerated."
            )


def run_recipe(recipe: Recipe, out_root: Path, scratch_root: Path,
               tools: ExternalTools, verbose: bool) -> List[str]:
    """Build one recipe into ``out_root``. Returns notes worth reporting."""
    out_dir = out_root / recipe.data_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    _assert_preservable(recipe, out_dir)

    scratch = scratch_root / recipe.key
    scratch.mkdir(parents=True, exist_ok=True)

    stash = scratch / "_preserved"
    stash.mkdir(exist_ok=True)
    _clear_targets(recipe, out_dir, stash)

    recipe.build(BuildContext(out_dir=out_dir, scratch=scratch,
                              tools=tools, verbose=verbose))

    missing = [name for name in recipe.produces if not (out_dir / name).exists()]
    if missing:
        raise RuntimeError(f"recipe finished without writing: {', '.join(missing)}")

    notes = []
    restored = _restore_preserved(recipe, out_dir, stash)
    if restored:
        notes.append("carried over " + ", ".join(restored))
    return notes


def check_recipe(recipe: Recipe, reference_root: Path, build_root: Path,
                 scratch_root: Path, tools: ExternalTools, verbose: bool,
                 rtol: float, atol: float) -> List[str]:
    """Build into ``build_root`` and describe how it differs from the committed tree.

    Preserved groups are seeded from the committed files so that a check does
    not report them as missing.
    """
    out_dir = build_root / recipe.data_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    for name in recipe.preserve:
        committed = reference_root / recipe.data_dir / name
        if committed.exists():
            shutil.copy(committed, out_dir / name)

    run_recipe(recipe, build_root, scratch_root, tools, verbose)

    diffs: List[str] = []
    for name in recipe.produces:
        reference = reference_root / recipe.data_dir / name
        candidate = out_dir / name
        if not reference.exists():
            diffs.append(f"{name}: not present in {reference_root}")
            continue
        for line in compare_h5(reference, candidate, rtol=rtol, atol=atol):
            diffs.append(f"{name}: {line}")
    return diffs


# ============================================================================
# Entry point
# ============================================================================

def main(argv=None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("recipe", nargs="*",
                        help="recipe key(s), or 'all'")
    parser.add_argument("--output-path", type=Path, default=INPUTS_ROOT,
                        help="inputs tree to write into (default: the committed "
                             "tests/functional/afqmc_inputs)")
    parser.add_argument("--scratch", type=Path,
                        help="directory for intermediate files; a temporary one "
                             "is used and discarded by default")
    parser.add_argument("--keep-scratch", action="store_true",
                        help="do not delete the temporary scratch directory")
    parser.add_argument("--check", action="store_true",
                        help="build into a temporary tree and report differences "
                             "against --output-path without modifying it")
    parser.add_argument("--rtol", type=float, default=1e-8,
                        help="relative tolerance for --check (default: 1e-8)")
    parser.add_argument("--atol", type=float, default=1e-10,
                        help="absolute tolerance for --check (default: 1e-10)")
    parser.add_argument("--dry-run", action="store_true",
                        help="report what would be built and exit")
    parser.add_argument("--list", action="store_true",
                        help="list the recipes and exit")
    parser.add_argument("--require-external", action="store_true",
                        help="fail instead of skipping when an external code is "
                             "not configured")
    parser.add_argument("-v", "--verbose", action="store_true",
                        help="verbose output from pyscf, afqmctools and AutoHF")
    args = parser.parse_args(argv)

    recipes = build_recipes()
    tools = ExternalTools.from_env()

    if args.list:
        _print_list(recipes, tools, args.output_path)
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
            missing = tools.missing(recipe.external)
            mark = f"  [skipped: needs {', '.join(missing)}]" if missing else ""
            print(f"{recipe.key} -> {args.output_path / recipe.data_dir}{mark}")
            for name in recipe.produces:
                print(f"    {name}")
        return 0

    scratch_root = args.scratch
    temporary = None
    if scratch_root is None:
        temporary = tempfile.mkdtemp(prefix="afqmc_inputs_")
        scratch_root = Path(temporary)
    scratch_root.mkdir(parents=True, exist_ok=True)

    build_root = args.output_path
    check_root = None
    if args.check:
        check_root = scratch_root / "_rebuild"
        check_root.mkdir(parents=True, exist_ok=True)
        build_root = check_root

    print(f"inputs tree : {args.output_path}")
    print(f"scratch     : {scratch_root}")
    if args.check:
        print(f"mode        : check only (rebuilding into {build_root})")

    results: List[Result] = []
    try:
        for recipe in selected:
            missing = tools.missing(recipe.external)
            if missing and not args.require_external:
                results.append(Result(recipe.key, Outcome.SKIPPED,
                                      f"needs {', '.join(missing)}"))
                print(f"\n--- {recipe.key}: skipped, needs {', '.join(missing)}")
                continue

            print(f"\n--- {recipe.key} -> {build_root / recipe.data_dir}", flush=True)
            start = perf_counter()
            try:
                if args.check:
                    diffs = check_recipe(recipe, args.output_path, build_root,
                                         scratch_root, tools, args.verbose,
                                         args.rtol, args.atol)
                    outcome = Outcome.DIFFERS if diffs else Outcome.CHECKED
                    results.append(Result(recipe.key, outcome,
                                          seconds=perf_counter() - start,
                                          diffs=diffs))
                else:
                    notes = run_recipe(recipe, build_root, scratch_root,
                                       tools, args.verbose)
                    results.append(Result(recipe.key, Outcome.BUILT,
                                          detail="; ".join(notes),
                                          seconds=perf_counter() - start))
            except MissingExternalTool as exc:
                # A tool can also go missing part-way in (a recipe asks for one
                # it did not declare). Honour --require-external there too.
                outcome = (Outcome.FAILED if args.require_external
                           else Outcome.SKIPPED)
                results.append(Result(recipe.key, outcome, str(exc),
                                      seconds=perf_counter() - start))
            except Exception as exc:  # noqa: BLE001 - one bad recipe must not stop the rest
                traceback.print_exc()
                results.append(Result(recipe.key, Outcome.FAILED,
                                      f"{type(exc).__name__}: {exc}",
                                      seconds=perf_counter() - start))
    finally:
        if temporary is not None and not args.keep_scratch:
            shutil.rmtree(temporary, ignore_errors=True)
        elif temporary is not None:
            print(f"\nscratch kept at {temporary}")

    _print_summary(results, args.check, recipes)

    bad = {Outcome.FAILED, Outcome.DIFFERS}
    return 1 if any(result.outcome in bad for result in results) else 0


if __name__ == "__main__":
    sys.exit(main())
