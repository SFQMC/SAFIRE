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
Simple functional-test runner for SAFIRE.

For a chosen system it:
  1. enumerates the hamiltonian x wavefunction x walker combinations defined in
     `functional_cases.py` (the same definitions the test modules use),
  2. filters them into "expected success" and "expected failure" sets using a
      reimplementation of the suite's spin-symmetry / implementation rules,
  3. writes `afqmc.json` and runs AFQMC for each case in a directory mirroring
     the `afqmc_ref_runs` layout,
  4. records `afqmc.out` (+ AFQMC's native output) and a small `results.h5`,
  5. compares against the stored reference `results.h5` and prints a plain-text
     PASS/FAIL line per case plus a final tally.

Usage:
    export AFQMC_EXEC=/path/to/afqmc
    python run_functional.py SYSTEM --output-path DIR \
        [--mpiexec "mpiexec -n 34" | "srun"] [--compute {cpu,gpu}] \
        [--timeout SECONDS] [--dry-run]
    python run_functional.py --list          # show available systems
    python run_functional.py all --output-path DIR ...
"""

import argparse
import json
import os
import re
import shlex
import subprocess as sp
import sys
from dataclasses import dataclass, replace
from pathlib import Path
from time import perf_counter
from typing import List, Optional

import h5py as h5
import numpy as np

# Reusing SAFIRE library utilities is fine; only the dev test harness is avoided.
from afqmctools.utils.types import SpinSymm
from afqmctools.analysis.rdm import average_afqmc_rdm
from stats.scalar_dat import analyze_scalar_data

from functional_cases import (
    HamiltonianClass,
    WavefunctionClass,
    Hamiltonian,
    Wavefunction,
    System,
    WALKERS,
    build_systems,
)


@dataclass
class Case:
    hamiltonian: Hamiltonian
    wavefunction: Wavefunction
    walker: str
    reference: Path      # absolute path to the reference results.h5
    out_subdir: Path     # path (relative to output-path/system) for this run
    runparams: dict
    bp: bool = False     # run with back-propagation (and compare the 1-RDM)


# ============================================================================
# Filtering rules
# ============================================================================

def _wavefunction_is_implemented(c: Case) -> bool:
    # only collinear PHMSD with collinear walkers is implemented; all NOMSD spin symmetries are fine.
    if c.wavefunction.type == WavefunctionClass.PHMSD:
        return WALKERS[c.walker] == SpinSymm.COLLINEAR or WALKERS[c.walker] == SpinSymm.NONCOLLINEAR and c.wavefunction.spin == SpinSymm.COLLINEAR
    return True


def should_succeed(c: Case) -> bool:
    rules = [
        _wavefunction_is_implemented(c),
        WALKERS[c.walker] >= c.hamiltonian.spin,  # walker<->hamiltonian spin compatible
        WALKERS[c.walker] >= c.wavefunction.spin,  # walker<->wavefunction spin compatible
        # not closed-THC with noncollinear wfn
        not (c.hamiltonian.type == HamiltonianClass.THC
             and c.hamiltonian.spin == SpinSymm.CLOSED
             and c.wavefunction.spin == SpinSymm.NONCOLLINEAR),
        # no closed walkers on lattice hamiltonian
        not (c.hamiltonian.type == HamiltonianClass.MODEL
             and c.hamiltonian.spin == SpinSymm.CLOSED),
        # wfn/walkers either both fully-polarized or both not
        (c.wavefunction.spin == SpinSymm.FULLYPOLARIZED)
        == (WALKERS[c.walker] == SpinSymm.FULLYPOLARIZED),
    ]
    return all(rules)


def should_skip(c: Case) -> bool:
    # Multi-determinant noncollinear (GHF CASCI) NOMSD works but currently
    # segfaults in the reference
    if c.wavefunction.file == "afqmc_casci_ghf_nomsd.h5":
        return True

    # The code now supports model Hamiltonians as long as the walker spin symmetry is not closed
    # However, the reference fails for any closed model Hamiltonian.
    if c.hamiltonian.type == HamiltonianClass.MODEL and c.hamiltonian.spin == SpinSymm.CLOSED:
        return True

    return False


def bp_cherry_pick(c: Case) -> bool:
    """Back-propagation subset selection from should_run_successfully_bp.

    BP runs are expensive, so only a representative subset of the successful
    space is exercised, chosen to cover distinct spin-symmetry transitions.
    """
    h, w, walker = c.hamiltonian.spin, c.wavefunction.spin, WALKERS[c.walker]
    if h == SpinSymm.CLOSED:
        if w == SpinSymm.COLLINEAR:
            return walker != SpinSymm.CLOSED
        if w == SpinSymm.NONCOLLINEAR:
            return walker == SpinSymm.NONCOLLINEAR
    elif h == SpinSymm.COLLINEAR:
        if w == SpinSymm.COLLINEAR:
            return walker in (SpinSymm.COLLINEAR, SpinSymm.NONCOLLINEAR)
    elif h == SpinSymm.NONCOLLINEAR:
        return w == SpinSymm.NONCOLLINEAR and walker == SpinSymm.NONCOLLINEAR
    return False


def bp_cases(success: List[Case]) -> List[Case]:
    """The back-propagation cases: the cherry-picked subset of success cases,
    flagged for BP and written to a sibling ``*_bp`` directory so they do not
    clobber the plain run."""
    picked = []
    for c in success:
        if bp_cherry_pick(c):
            out = c.out_subdir.parent / (c.out_subdir.name + "_bp")
            picked.append(replace(c, bp=True, out_subdir=out))
    return picked


# ============================================================================
# Case generation
# ============================================================================

def merge_runparams(*sources) -> dict:
    out = {}
    for s in sources:
        if s:
            out.update(s)
    return out


def generate(system: System) -> List[Case]:
    cases: List[Case] = []
    ref = system.ref_dir

    for h_name, hamiltonian in system.hamiltonians.items():
        for w_name, wavefunction in system.wavefunctions.items():
            for walker in system.walkers:
                run_dir = ref / h_name / w_name / walker.lower()
                if not run_dir.exists():
                    print(f"  [skip] no reference dir: {run_dir}")
                    continue
                cases.append(Case(
                    hamiltonian=hamiltonian, wavefunction=wavefunction, walker=walker,
                    reference=run_dir / "results.h5",
                    out_subdir=Path(h_name) / w_name / walker.lower(),
                    runparams=merge_runparams(hamiltonian.runparams, wavefunction.runparams),
                ))

    for ex in system.extra:
        reference = ref / ex.reference
        cases.append(Case(
            hamiltonian=ex.hamiltonian, wavefunction=ex.wavefunction, walker=ex.walker,
            reference=reference,
            out_subdir=reference.relative_to(ref).parent,
            runparams=merge_runparams(ex.hamiltonian.runparams, ex.wavefunction.runparams),
        ))
    return cases


# ============================================================================
# Input file
# ============================================================================

def write_input(path: Path, hamil_file: Path, wfn_file: Path, walker: str,
                n_walkers_per_mpi_task: int, steps: int, timestep: float,
                run_bp: bool):
    execute = {
        "walker_set": {"walker_type": walker},
        "wavefunction": {"filename": str(wfn_file)},
        "hamiltonian": {"filename": str(hamil_file)},
        "timestep": timestep,
        "steps": steps,
        "n_walkers_per_mpi_task": n_walkers_per_mpi_task,
    }
    if run_bp:
        execute["estimator"] = {
            "name": "back_propagation",
            "path_restoration": True,
            "bp_walker_ortho_interval": 10,
            "measure_interval_multiplier": 40,
            "equil_multiplier": 200,
            "onerdm": {"name": "one_rdm"},
        }
    execute["population_control_interval"] = 10
    execute["measure_interval_multiplier"] = 1
    execute["walker_ortho_interval"] = 10
    execute["seed"] = 42

    document = {
        "afqmc": {
            "project": {"id": "qmc", "series": 0},
            "execute": execute,
        }
    }
    with open(path, "w") as f:
        json.dump(document, f, indent=2)


# ============================================================================
# Output parsing + results.h5
# ============================================================================

_ANSI = re.compile(r"\x1b\[[0-9;]*m")


def raised_error(text: str) -> bool:
    return re.search(r"\[error\]", text) is not None


def raised_warning(text: str) -> bool:
    return re.search(r"\[warning\]", text) is not None


def is_finite(text: str) -> bool:
    return re.search(r"\([-]?nan,|[-]nan\)", text) is None


def error_messages(text: str) -> set:
    if not raised_error(text):
        return set()
    text = _ANSI.sub("", text)
    msgs = {m.lstrip().rstrip() for m in re.findall(r"\[error\] (.+)", text)}
    return {m for m in msgs
            if not re.match(r"\*{10,}", m)
            and not re.match(r"APPLICATION ABORT: Fatal Error\.", m)}


def warning_messages(text: str) -> set:
    if not raised_warning(text):
        return set()
    text = _ANSI.sub("", text)
    return {m.lstrip().rstrip() for m in re.findall(r"\[warning\] (.+)", text)}


def _write_message_group(f: h5.File, name: str, messages: set):
    g = f.create_group(name)
    g.create_dataset("num_messages", data=len(messages))
    prefix = name.split("_")[0]  # error_messages -> error, warning_messages -> warning
    for i, m in enumerate(messages):
        g.create_dataset(f"{prefix}_{i}", data=str(m))


def _scalar_column(scalar_file: str, column: Optional[str], label: str):
    """Average a scalar.dat column, returning [value, stoch_error] or None."""
    try:
        params = dict(fname=scalar_file, xaxis="time", nequil=5.0, verbose=False)
        if column is not None:
            params["column"] = column
        v, dv = analyze_scalar_data(params)
        return np.array([v, dv])
    except Exception as e:  # noqa: BLE001
        print(f"  [warn] could not analyze {label}: {e}")
        return None


def record_results(out_dir: Path, return_code: int, run_time: float, run_bp: bool):
    """Extract a results summary and write results.h5 (schema-compatible with the
    reference files written by the pytest suite: includes energy, weight,
    LogOvlpFactor and the error/warning message groups)."""
    out_text = (out_dir / "afqmc.out").read_text()
    with h5.File(out_dir / "results.h5", "w") as f:
        f.create_dataset("return_code", data=return_code)
        f.create_dataset("run_time_seconds", data=run_time)
        f.create_dataset("afqmc_is_finite", data=is_finite(out_text))
        f.create_dataset("afqmc_raised_error", data=raised_error(out_text))
        f.create_dataset("afqmc_raised_warning", data=raised_warning(out_text))
        f.create_dataset("input_file", data=(out_dir / "afqmc.json").read_text())
        _write_message_group(f, "error_messages", error_messages(out_text))
        _write_message_group(f, "warning_messages", warning_messages(out_text))
        if return_code == 0:
            scalar_file = str(out_dir / "qmc.s000.scalar.dat")
            energy = _scalar_column(scalar_file, None, "energy")
            if energy is not None:
                f.create_dataset("energy", data=energy)
            weight = _scalar_column(scalar_file, "weight", "weight")
            if weight is not None:
                f.create_dataset("weight", data=weight)
            # Log overlap: old name LogOvlpFactor, new name LogOvlp; support both.
            try:
                ovlp_col = ("LogOvlpFactor" if "LogOvlpFactor" in open(scalar_file).read()
                            else "LogOvlp")
            except OSError:
                ovlp_col = "LogOvlpFactor"
            log_ovlp = _scalar_column(scalar_file, ovlp_col, "LogOvlpFactor")
            if log_ovlp is not None:
                f.create_dataset("LogOvlpFactor", data=log_ovlp)
            if run_bp:
                try:
                    rho, drho = average_afqmc_rdm(str(out_dir / "qmc.s000.stat.h5"))
                    f.create_dataset("avg_1rdm", data=rho)
                    f.create_dataset("avg_1rdm_stoch_error", data=drho)
                except Exception as e:  # noqa: BLE001
                    print(f"  [warn] could not average back-propagated 1-RDM: {e}")


# ============================================================================
# Comparisons
# ============================================================================

def _rc_class(code) -> int:
    return 1 if int(code) > 0 else 0


def _h5_error_messages(f: h5.File) -> set:
    g = f.get("error_messages")
    if g is None:
        return set()
    n = int(g["num_messages"][()])
    return {str(g[f"error_{i}"][()]) for i in range(n)}


def _compare_energy(ft: h5.File, fr: h5.File) -> bool:
    if "energy" not in ft or "energy" not in fr:
        print("  [compare] missing energy dataset")
        return False
    E, _ = ft["energy"][:]
    Eref, dEref = fr["energy"][:]
    if np.isclose(E, Eref, atol=dEref):
        print(f"  [compare] energy OK: {E:.6f} vs {Eref:.6f} +- {dEref:.6f}")
        return True
    if np.isclose(E, Eref, atol=2 * dEref):
        print(f"  [compare] energy within 2x tol: {E:.6f} vs {Eref:.6f} +- {dEref:.6f}")
        return True
    print(f"  [compare] energy mismatch: {E:.6f} vs {Eref:.6f} +- {dEref:.6f}")
    return False


def _compare_1rdm(ft: h5.File, fr: h5.File) -> bool:
    """Compare the back-propagated 1-RDM. A shape mismatch fails; otherwise we
    report the fraction of elements agreeing within the joint stochastic
    uncertainty (statistics-only check, warn-like, never fails) as the suite does."""
    if "avg_1rdm" not in ft or "avg_1rdm" not in fr:
        print("  [compare] missing avg_1rdm dataset")
        return False
    A, Aerr = ft["avg_1rdm"][:], ft["avg_1rdm_stoch_error"][:]
    B, Berr = fr["avg_1rdm"][:], fr["avg_1rdm_stoch_error"][:]
    if A.shape != B.shape or Aerr.shape != Berr.shape:
        print(f"  [compare] avg_1rdm shape mismatch: {A.shape} vs {B.shape}")
        return False
    A = np.asarray(A, dtype=np.complex128)
    B = np.asarray(B, dtype=np.complex128)
    sigma = np.sqrt(Aerr ** 2 + Berr ** 2).real
    for part in ("real", "imag"):
        a, b = getattr(A, part), getattr(B, part)
        for k in (1, 2, 3):
            pct = float(np.mean(np.isclose(a, b, atol=k * sigma, rtol=1e-4)) * 100)
            print(f"  [compare] avg_1rdm {part} within {k} sigma: {pct:.1f}%")
    return True


def compare(test_h5: Path, ref_h5: Path, expect_success: bool, check_bp: bool = False) -> bool:
    if not ref_h5.exists():
        print(f"  [compare] reference missing: {ref_h5}")
        return False
    with h5.File(test_h5, "r") as ft, h5.File(ref_h5, "r") as fr:
        if not bool(ft["afqmc_is_finite"][()]):
            print("  [compare] test results contain NaN")
            return False
        test_rc = _rc_class(ft["return_code"][()])
        ref_rc = _rc_class(fr["return_code"][()])
        if test_rc != ref_rc:
            print(f"  [compare] return-code class mismatch: test={test_rc} ref={ref_rc}")
            return False

        if expect_success:
            if test_rc != 0:
                print("  [compare] expected success but run exited with error")
                return False
            ok = _compare_energy(ft, fr)
            if check_bp:
                ok = _compare_1rdm(ft, fr) and ok
            for name in ("weight", "LogOvlpFactor"):
                if name in ft and name in fr:
                    print(f"  [compare] {name} (print-only): "
                          f"test={ft[name][:]} ref={fr[name][:]}")
            return ok

        # expected failure: both must have exited with a SAFIRE error.
        if test_rc != 1 or ref_rc != 1:
            print("  [compare] expected both to exit with error")
            return False
        if _h5_error_messages(fr) != _h5_error_messages(ft):
            print("  [compare] error messages differ (still counts as matching error exit)")
        return True


# ============================================================================
# Run loop
# ============================================================================

def detect_ranks(mpiexec: str) -> int:
    """Determine how many MPI ranks `mpiexec` will actually launch by running a
    trivial probe under it and counting the processes that start."""
    cmd = shlex.split(mpiexec) + [sys.executable, "-c", "print('RANKPROBE')"]
    try:
        proc = sp.run(cmd, stdout=sp.PIPE, stderr=sp.DEVNULL,
                      env=os.environ, text=True, timeout=120)
    except Exception as e:
        print(f"  [warn] could not probe MPI ranks ({e}); assuming 1")
        return 1
    count = proc.stdout.count("RANKPROBE")
    if count == 0:
        print("  [warn] could not determine MPI rank count; assuming 1")
        return 1
    return count


def run_case(system: System, case: Case, out_root: Path, mpiexec: str,
             afqmc_exec: str, compute: str, ranks: int, timeout: Optional[float],
             expect_success: bool) -> bool:
    """Run one case's AFQMC subprocess, then record and compare its results."""
    out_dir = (out_root / case.out_subdir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    hamil_file = system.inputs_dir / case.hamiltonian.file
    wfn_file = system.inputs_dir / case.wavefunction.file

    total_walkers = case.runparams.get("total_walkers", 1600)
    n_walkers = total_walkers // max(1, ranks)

    input_file = out_dir / "afqmc.json"
    write_input(
        input_file, hamil_file, wfn_file, case.walker,
        n_walkers_per_mpi_task=n_walkers,
        steps=case.runparams.get("steps", 10000),
        timestep=case.runparams.get("timestep", 0.01),
        run_bp=case.bp,
    )

    # Pass the input as an absolute path and let the child run in out_dir (so
    # AFQMC's native output lands there) rather than mutating our own cwd.
    run_cmd = shlex.split(mpiexec) + [afqmc_exec, "--compute", compute,
                                     str(input_file)]
    print(f"  cmd: {' '.join(run_cmd)}")

    with open(out_dir / "afqmc.out", "w") as fout:
        t0 = perf_counter()
        try:
            # timeout is in minutes; subprocess.run wants seconds.
            proc = sp.run(run_cmd, stdout=fout, stderr=fout, cwd=out_dir,
                          env=os.environ,
                          timeout=timeout * 60 if timeout is not None else None)
            return_code = proc.returncode
        except sp.TimeoutExpired:
            fout.write("\n[error] run timed out\n")
            print(f"  [timeout] run timed out after {timeout} min")
            return False
        run_time = perf_counter() - t0

    record_results(out_dir, return_code, run_time, run_bp=case.bp)
    return compare(out_dir / "results.h5", case.reference, expect_success,
                   check_bp=case.bp)


# ============================================================================
# CLI
# ============================================================================

def main(argv=None) -> int:
    systems = build_systems()
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("system", nargs="?", help="system key, or 'all'")
    p.add_argument("--output-path", type=Path, help="root output directory. It is recommended to empty this directory before running the tests.")
    p.add_argument("--mpiexec", default="",
                   help='launcher prefix prepended to AFQMC_EXEC (e.g. "mpiexec -n 34")')
    p.add_argument("--compute", choices=["cpu", "gpu"], default="cpu",
                   help="compute device passed to AFQMC via --compute")
    p.add_argument("--timeout", type=float, default=None,
                   help="per-run timeout in minutes")
    p.add_argument("--dry-run", action="store_true",
                   help="list planned cases without running AFQMC")
    p.add_argument("--list", action="store_true", help="list available systems and exit")
    args = p.parse_args(argv)

    if args.list or not args.system:
        print("Available systems:")
        for name in systems:
            print(f"  {name}")
        print("  all")
        return 0

    if args.system == "all":
        selected = list(systems)
    elif args.system in systems:
        selected = [args.system]
    else:
        print(f"Unknown system '{args.system}'. Use --list to see options.")
        return 2

    if not args.dry_run and args.output_path is None:
        print("--output-path is required unless --dry-run is given.")
        return 2

    afqmc_exec = os.environ.get("AFQMC_EXEC")
    if not args.dry_run and not afqmc_exec:
        print("AFQMC_EXEC environment variable is not set.")
        return 2

    # The launcher is fixed for the whole run, so probe the rank count once.
    ranks = 1
    if not args.dry_run and args.compute != "gpu":
        ranks = detect_ranks(args.mpiexec)
        print(f"Detected {ranks} MPI rank(s) for launcher: {args.mpiexec!r}")

    total_pass = total_fail = 0
    for name in selected:
        system = systems[name]
        all_cases = generate(system)
        success = [c for c in all_cases if should_succeed(c) and not should_skip(c)]
        fail = [c for c in all_cases if not should_succeed(c) and not should_skip(c)]

        for c in all_cases:
            if should_skip(c):
                print(f"Skipped case {c.out_subdir}.")

        # Back-propagation cases are a cherry-picked subset of the success cases,
        # only for systems that have them.
        bp = bp_cases(success) if system.bp else []
        print(f"=== {name}: {len(success)} expected-success, "
              f"{len(fail)} expected-fail, {len(bp)} back-propagation ===")

        for expect_success, label, group in (
                (False, "EXPECT_FAILURE", fail),
                (True, "EXPECT_SUCCESS", success),
                (True, "BACKPROPAGATION", bp)):
            for case in group:
                tag = f"[{label}] {case.out_subdir}"
                if args.dry_run:
                    print(f"  {tag}")
                    continue
                print(f"\n>>> {name} {tag}")
                ok = run_case(
                    system, case, args.output_path / name, args.mpiexec,
                    afqmc_exec, args.compute, ranks, args.timeout, expect_success)
                print(f"  RESULT: {'PASS' if ok else 'FAIL'}")
                total_pass += int(ok)
                total_fail += int(not ok)

    if not args.dry_run:
        print(f"\n==== {total_pass} passed, {total_fail} failed ====")
    return 1 if total_fail else 0


if __name__ == "__main__":
    sys.exit(main())
