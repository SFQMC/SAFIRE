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
     `functional_cases.py`,
  2. filters them into "expected success" and "expected failure" sets using the
     spin-symmetry / implementation rules,
  3. writes `afqmc.json` and runs AFQMC for each case in a directory mirroring
     the `statistical_references` layout,
  4. records the console output to `afqmc.out` and a digested version of the measurement results to `results.h5`,
  5. compares against the stored reference `results.h5` and prints a plain-text
     PASS/FAIL line per case plus a final tally.

The comparison in step 5 comes in two flavours. By default a long run is compared
statistically against `statistical_references`, which tests the physics but tolerates
any change that stays within the stochastic error. With `--snapshot` a short, seeded
run is compared to `snapshot_references` for numerically exact agreement, which
catches changes the statistical test cannot see (a reordered random-number stream,
say) but only reproduces at a fixed rank count, and on gpu only for an executable built
with -DDEVICE_RNG_FROM_HOST=ON.

With `--regenerate` step 5 is replaced by copying each freshly recorded `results.h5`
into the reference tree, which is how the stored references are produced in the first
place.

The AFQMC executable is taken from the AFQMC_EXEC environment variable.
"""

import argparse
import enum
import json
import os
import re
import shlex
import shutil
import subprocess as sp
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path
from time import perf_counter
from typing import List, Optional

import h5py as h5
import numpy as np
import scipy.stats

# Reusing SAFIRE library utilities is fine; only the dev test harness is avoided.
from afqmctools.utils.types import SpinSymm
from afqmctools.analysis.measurements import average_measurements

from functional_cases import (
    HamiltonianClass,
    WavefunctionClass,
    Hamiltonian,
    Wavefunction,
    System,
    build_systems,
    INPUTS_ROOT,
    REFERENCES_ROOT,
    SNAPSHOT_REFERENCES_ROOT
)


class TestType(enum.Enum):
    EXPECT_SUCCESS = enum.auto()
    EXPECT_FAILURE = enum.auto()
    BACKPROPAGATION = enum.auto()


SIGNIFICANCE_LEVEL = 0.001
MACHINE_EPS = 1e-9

# AFQMC's own output, named after the stem of the input file write_input writes. One run writes
# one file, with the observables of each execute block below a Stage<N> group of its own.
AFQMC_RESULTS = "afqmc.results.h5"


@dataclass
class Case:
    hamiltonian: Hamiltonian
    wavefunction: Wavefunction
    walker: SpinSymm
    data_dir: str              # system dir shared by afqmc_inputs/ and both reference roots
    out_subdir: Path           # path (relative to output-path/system) for this run
    runparams: dict
    observables: dict          # back-propagation observable blocks; empty means no BP run

    def reference(self, snapshot: bool = False) -> Path:
        """The stored results.h5 for this case, in the snapshot or statistical tree."""
        root = SNAPSHOT_REFERENCES_ROOT if snapshot else REFERENCES_ROOT
        return root / self.data_dir / self.out_subdir / "results.h5"


# ============================================================================
# Filtering rules
# ============================================================================

def _wavefunction_is_implemented(c: Case) -> bool:
    # only collinear PHMSD with collinear walkers is implemented; all NOMSD spin symmetries are fine.
    if c.wavefunction.type == WavefunctionClass.PHMSD:
        return c.walker == SpinSymm.COLLINEAR and c.wavefunction.spin == SpinSymm.COLLINEAR
    return True


def should_succeed(c: Case) -> bool:
    rules = [
        _wavefunction_is_implemented(c),
        c.walker >= c.hamiltonian.spin,  # walker<->hamiltonian spin compatible
        c.walker >= c.wavefunction.spin,  # walker<->wavefunction spin compatible
        # no closed walkers on lattice hamiltonian
        not (c.hamiltonian.type == HamiltonianClass.MODEL
             and c.walker == SpinSymm.CLOSED),
    ]
    return all(rules)


def should_skip(c: Case) -> bool:
    return False


def should_backprop(c: Case) -> bool:
    """Back-propagation subset selection from should_succeed.

    BP runs are expensive, so only a representative subset of the successful
    space is exercised, chosen to cover distinct spin-symmetry transitions.
    """
    h, w, walker = c.hamiltonian.spin, c.wavefunction.spin, c.walker
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
    """Every hamiltonian x wavefunction x walker combination of a system, each
    keyed to the reference at `<hamiltonian>/<wavefunction>/<walker>/results.h5`."""
    cases: List[Case] = []
    for h_name, hamiltonian in system.hamiltonians.items():
        for w_name, wavefunction in system.wavefunctions.items():
            for walker in system.walkers:
                subdir = Path(h_name) / w_name / walker.name.lower()
                cases.append(Case(
                    hamiltonian=hamiltonian, wavefunction=wavefunction, walker=walker,
                    data_dir=system.data_dir,
                    out_subdir=subdir,
                    runparams=merge_runparams(hamiltonian.runparams, wavefunction.runparams),
                    observables=system.observables,
                ))
    return cases


def resolve_observable_inputs(observables: dict, inputs_dir: Path) -> dict:
    """`observables` with every stored-input `filename` made absolute.

    The blocks name their inputs relative to the system's afqmc_inputs directory, but AFQMC
    runs in the case output directory. A stored input is an h5 path block, i.e. a nested
    `{"filename": ..., "group": ...}`, so the walk goes one level down.
    """
    resolved = {}
    for name, block in observables.items():
        block = {key: dict(value) if isinstance(value, dict) else value
                 for key, value in block.items()}
        for value in block.values():
            if isinstance(value, dict) and "filename" in value:
                value["filename"] = str(inputs_dir / value["filename"])
        resolved[name] = block
    return resolved


# ============================================================================
# Input file
# ============================================================================

def write_input(path: Path, hamil_file: Path, wfn_file: Path, walker: SpinSymm,
                n_walkers_per_mpi_task: int, timestep: float, observables: dict,
                snapshot: bool):
    steps = 10000
    equilibration_steps = 2000
    population_control_interval = 10
    bp_measure_interval_multiplier = 40
    if snapshot:
        steps = 20
        equilibration_steps = 0
        population_control_interval = 1
        bp_measure_interval_multiplier = 2

    execute = {
        "walker_set": {"walker_type": walker.name},
        "wavefunction": {"filename": str(wfn_file)},
        "hamiltonian": {"filename": str(hamil_file)},
        "timestep": timestep,
        "steps": steps,
        "n_walkers_per_mpi_task": n_walkers_per_mpi_task,
    }
    if observables:
        execute["estimators"] = {
            "mixed": {
                "measure_interval_multiplier": bp_measure_interval_multiplier,
                **observables,
            },
            "backprop": {
                "path_restoration": True,
                "measure_interval_multiplier": bp_measure_interval_multiplier,
                **observables,
            },
        }
    execute["population_control_interval"] = population_control_interval
    execute["measure_interval_multiplier"] = 1
    execute["walker_ortho_interval"] = 10
    execute["equilibration_steps"] = equilibration_steps

    document = {
        "seed": 42,
        "execute": execute,
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


def _average_observables(results: Path) -> dict:
    """(mean, stochastic error) for every observable AFQMC measured, keyed by the full
    '/'-separated path it was measured under, e.g. `Stage0/Energy` or
    `Stage0/BackPropEstimator/Steps=40/OneRDM`.

    Nothing is discarded here: the driver measures nothing before `equilibration_steps`, so
    every bin in the file is already equilibrated.
    """
    try:
        return average_measurements(results)
    except Exception as e:  # noqa: BLE001
        print(f"  [warn] could not average {results.name}: {e}")
        return {}


def _write_measurements(f: h5.File, averaged: dict):
    """Store each observable as `mean` and `error` below a group named after the path it was
    measured under, so a recorded file mirrors AFQMC's own observable tree."""
    group = f.create_group("measurements")
    for name, (mean, error) in averaged.items():
        observable = group.create_group(name)
        for dataset, value in (("mean", mean), ("error", error)):
            value = np.asarray(value)
            observable.create_dataset(dataset, data=value)


def _read_measurements(f: h5.File) -> dict:
    """{path: (mean, error)} for every observable `_write_measurements` stored."""
    recorded = {}
    group = f.get("measurements")
    if group is None:
        return recorded

    def visit(name, obj):
        if isinstance(obj, h5.Group) and "mean" in obj:
            recorded[name] = (obj["mean"][()], obj["error"][()])

    group.visititems(visit)
    return recorded


def record_results(out_dir: Path, return_code: int, ranks: int, run_time: float):
    """Extract a results summary and write results.h5: the run metadata, the error/warning
    message groups, and every observable AFQMC measured, averaged over its bins."""
    out_text = (out_dir / "afqmc.out").read_text()
    with h5.File(out_dir / "results.h5", "w") as f:
        f.create_dataset("return_code", data=return_code)
        f.create_dataset("num_ranks", data=ranks)
        f.create_dataset("run_time_seconds", data=run_time)
        f.create_dataset("afqmc_raised_error", data=raised_error(out_text))
        f.create_dataset("afqmc_raised_warning", data=raised_warning(out_text))
        f.create_dataset("input_file", data=(out_dir / "afqmc.json").read_text())
        _write_message_group(f, "error_messages", error_messages(out_text))
        _write_message_group(f, "warning_messages", warning_messages(out_text))

        finite = is_finite(out_text)
        if return_code == 0:
            averaged = _average_observables(out_dir / AFQMC_RESULTS)
            _write_measurements(f, averaged)
            finite = finite and all(np.all(np.isfinite(v))
                                    for pair in averaged.values() for v in pair)
        f.create_dataset("afqmc_is_finite", data=finite)


# ============================================================================
# Comparisons
# ============================================================================

def _rc_class(code) -> int:
    return 1 if int(code) > 0 else 0


def _h5_messages(f: h5.File, group: str = "error_messages") -> set:
    g = f.get(group)
    if g is None:
        return set()
    n = int(g["num_messages"][()])
    prefix = group.split("_")[0]  # error_messages -> error_0, error_1, ...
    msgs = (g[f"{prefix}_{i}"][()] for i in range(n))
    return {m.decode() if isinstance(m, bytes) else str(m) for m in msgs}


def _compare_measurement(name: str, test, ref) -> bool:
    """Whether one observable agrees with its reference within the stochastic error.

    Any tensor rank, a scalar being the one-component case. A shape mismatch fails; otherwise
    we do a Bonferroni adjusted test on the worst mismatch.
    """
    A, Aerr = np.atleast_1d(test[0]).astype(np.complex128), np.abs(np.atleast_1d(test[1]))
    B, Berr = np.atleast_1d(ref[0]).astype(np.complex128), np.abs(np.atleast_1d(ref[1]))
    if A.shape != B.shape or Aerr.shape != Berr.shape:
        print(f"  [compare] {name} shape mismatch: {A.shape} vs {B.shape}")
        return False
    sigma = np.sqrt(Aerr ** 2 + Berr ** 2)
    # Only test components with a meaningful stochastic error. Off-diagonal spin
    # blocks that are identically zero (e.g. a collinear-derived noncollinear
    # reference) have sigma ~ machine epsilon, where (a - b)/sigma is a
    # tiny/tiny ratio that spuriously inflates the z-score.
    valid = sigma > MACHINE_EPS
    n_valid = int(np.count_nonzero(valid))

    det = ~valid
    if det.any():
        if not np.allclose(A[det], B[det], rtol=MACHINE_EPS, atol=MACHINE_EPS):
            d = np.abs(A - B)
            d[valid] = 0.0
            worst = tuple(map(int, np.unravel_index(np.argmax(d), d.shape)))
            print(f"  [compare] {name} deterministic (sigma <= {MACHINE_EPS}) mismatch: |Δ| = {d[worst]:.3e} > {MACHINE_EPS} at idx = {worst}")
            return False

    if n_valid == 0:
        print(f"  [compare] {name}: no components with sigma > {MACHINE_EPS}; matched to machine precision")
        return True
    z_crit = scipy.stats.norm.ppf(1 - SIGNIFICANCE_LEVEL / (2 * n_valid))

    z = np.zeros_like(sigma)
    z[valid] = np.abs(A[valid] - B[valid]) / sigma[valid]
    worst = tuple(map(int, np.unravel_index(np.argmax(z), z.shape)))
    values = (f"{A[worst]:.6f} ± {Aerr[worst]:.6f} vs "
              f"{B[worst]:.6f} ± {Berr[worst]:.6f} at idx = {worst}")

    if z[worst] <= z_crit:
        print(f"  [compare] {name} OK: worst component z = {z[worst]:.2f} <= {z_crit:.2f}, {values}")
        return True
    print(f"  [compare] {name} mismatch: worst component z = {z[worst]:.2f} > {z_crit:.2f}, {values}")
    return False


def _compare_measurements(ft: h5.File, fr: h5.File) -> bool:
    """Whether every observable of the run agrees with the reference. The same treatment for
    all of them: an observable is a tensor of some rank, and the energy is the rank-0 case."""
    test, ref = _read_measurements(ft), _read_measurements(fr)
    if set(test) != set(ref):
        print(f"  [compare] recorded observables differ: "
              f"only in test = {sorted(set(test) - set(ref))}, "
              f"only in reference = {sorted(set(ref) - set(test))}")
        return False
    if not test:
        print("  [compare] no observables were recorded")
        return False
    # a list, not a generator: report every observable rather than stopping at the first bad one
    return all([_compare_measurement(name, test[name], ref[name]) for name in sorted(test)])


def compare_statistically(test_h5: Path, ref_h5: Path, test_type: TestType) -> bool:
    """Whether the run agrees with the reference within the stochastic error."""
    with h5.File(test_h5, "r") as ft, h5.File(ref_h5, "r") as fr:
        if not bool(ft["afqmc_is_finite"][()]):
            print("  [compare] test results contain NaN")
            return False
        test_rc = _rc_class(ft["return_code"][()])
        ref_rc = _rc_class(fr["return_code"][()])
        if test_rc != ref_rc:
            print(f"  [compare] return-code class mismatch: test={test_rc} ref={ref_rc}")
            return False

        if test_type != TestType.EXPECT_FAILURE:
            if test_rc != 0:
                print("  [compare] expected success but run exited with error")
                return False
            return _compare_measurements(ft, fr)

        # expected failure: both must have exited with a SAFIRE error.
        if test_rc != 1 or ref_rc != 1:
            print("  [compare] expected both to exit with error")
            return False
        if _h5_messages(fr) != _h5_messages(ft):
            print("  [compare] error messages differ (still counts as matching error exit)")
        return True


def _exact_mismatch(a, b) -> Optional[str]:
    """None if a and b agree to MACHINE_EPS, else a description of the disagreement."""
    a, b = np.asarray(a), np.asarray(b)
    if a.shape != b.shape:
        return f"shape {a.shape} vs {b.shape}"
    if a.dtype.kind in "SUO" or b.dtype.kind in "SUO":
        return None if np.array_equal(a, b) else f"{a} vs {b}"
    if np.array_equal(a, b) or np.allclose(a, b, rtol=MACHINE_EPS, atol=MACHINE_EPS):
        return None
    d = np.abs(a.astype(np.complex128) - b.astype(np.complex128))
    if d.ndim == 0:
        return f"{a} vs {b} (|Δ| = {d:.3e} > {MACHINE_EPS})"
    worst = tuple(map(int, np.unravel_index(np.argmax(d), d.shape)))
    return f"|Δ| = {d[worst]:.3e} > {MACHINE_EPS} at idx = {worst}"


def compare_exactly(test_h5: Path, snapshot_h5: Path) -> bool:
    """Whether the run reproduces the snapshot to MACHINE_EPS.

    Snapshot runs are short, seeded and unequilibrated, so every recorded quantity is a
    deterministic function of the input and the number of ranks. That makes the whole
    file comparable, which is a stricter test than compare_statistically. The
    error/warning message text is the one exception, treated as in the statistical test:
    a difference is reported but does not fail the case.
    """
    with h5.File(test_h5, "r") as ft, h5.File(snapshot_h5, "r") as fs:
        if not bool(ft["afqmc_is_finite"][()]):
            print("  [compare] test results contain NaN")
            return False

        # run_time_seconds is timing; input_file is compared as parsed settings below; the
        # observables are a group tree, compared below rather than as one dataset.
        ignored = {"run_time_seconds", "input_file", "measurements"}
        test_keys, snap_keys = set(ft.keys()) - ignored, set(fs.keys()) - ignored
        if test_keys != snap_keys:
            print(f"  [compare] recorded quantities differ: "
                  f"only in test = {sorted(test_keys - snap_keys)}, "
                  f"only in snapshot = {sorted(snap_keys - test_keys)}")
            return False

        # A snapshot only reproduces at the rank count it was recorded with: walkers are
        # split over the ranks and every rank draws its own auxiliary fields. Report that
        # before anything else, since it explains every other difference.
        if "num_ranks" in test_keys:
            mismatch = _exact_mismatch(ft["num_ranks"][()], fs["num_ranks"][()])
            if mismatch is not None:
                print(f"  [compare] rank count differs ({mismatch}); rerun with the "
                      f"launcher the snapshot was recorded with, or re-record it")
                return False

        message_groups = {"error_messages", "warning_messages"}
        for name in sorted(message_groups & test_keys):
            if _h5_messages(ft, name) != _h5_messages(fs, name):
                print(f"  [compare] {name} differ (not compared)")

        ok = True
        compared = sorted(test_keys - message_groups)
        for name in compared:
            mismatch = _exact_mismatch(ft[name][()], fs[name][()])
            if mismatch is not None:
                print(f"  [compare] {name} mismatch: {mismatch}")
                ok = False

        test_obs, snap_obs = _read_measurements(ft), _read_measurements(fs)
        if set(test_obs) != set(snap_obs):
            print(f"  [compare] recorded observables differ: "
                  f"only in test = {sorted(set(test_obs) - set(snap_obs))}, "
                  f"only in snapshot = {sorted(set(snap_obs) - set(test_obs))}")
            return False
        for name in sorted(test_obs):
            for i, part in enumerate(("mean", "error")):
                mismatch = _exact_mismatch(test_obs[name][i], snap_obs[name][i])
                if mismatch is not None:
                    print(f"  [compare] {name} {part} mismatch: {mismatch}")
                    ok = False
                compared.append(f"{name}/{part}")

        if ok:
            print(f"  [compare] all {len(compared)} recorded quantities match to "
                  f"{MACHINE_EPS}")
        return ok


def store_reference(results: Path, dest: Path, test_type: TestType) -> bool:
    """Copy a freshly recorded results.h5 over the stored reference at `dest`.

    A run is only frozen as a reference when its outcome matches what the case is
    expected to do: recording a crashed run as a success reference (or a successful
    run as an expected-failure reference) would bake the wrong behaviour in.
    """
    with h5.File(results, "r") as f:
        rc = _rc_class(f["return_code"][()])
        finite = bool(f["afqmc_is_finite"][()])
        has_measurements = bool(_read_measurements(f))

    if test_type == TestType.EXPECT_FAILURE:
        if rc == 0:
            print("  [regenerate] refusing to store: expected failure but the run "
                  "exited successfully")
            return False
    elif rc != 0:
        print("  [regenerate] refusing to store: expected success but the run exited "
              "with error")
        return False
    elif not finite:
        print("  [regenerate] refusing to store: results contain NaN")
        return False
    elif not has_measurements:
        print("  [regenerate] refusing to store: no observables were recorded, so AFQMC's "
              "output file was not read")
        return False

    try:
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(results, dest)
    except OSError as e:
        print(f"  [regenerate] could not write {dest}: {e}")
        return False
    print(f"  [regenerate] wrote {dest}")
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


def device_rng_from_host_enabled(afqmc_exec: str) -> bool:
    """Whether `afqmc_exec` was built with -DDEVICE_RNG_FROM_HOST=ON, read off its
    --version feature list. Without it the device samples its own random numbers and
    cannot reproduce a snapshot recorded on the host."""
    try:
        proc = sp.run([afqmc_exec, "--version"], stdout=sp.PIPE, stderr=sp.DEVNULL,
                      env=os.environ, text=True, timeout=120)
    except Exception as e:
        print(f"  [error] could not run '{afqmc_exec} --version' ({e})")
        return False
    for line in proc.stdout.splitlines():
        if line.startswith("Features:"):
            return "DeviceRNGFromHost" in line
    print(f"  [error] no 'Features:' line in '{afqmc_exec} --version' output")
    return False


def run_case(case: Case, test_type: TestType, out_root: Path, mpiexec: str,
             afqmc_exec: str, compute: str, ranks: int, timeout: Optional[float],
             snapshot: bool, regenerate: bool) -> Optional[bool]:
    """Run one case's AFQMC subprocess, then record and compare its results (or, with
    `regenerate`, store them as the new reference). None if the case has no reference
    and could not be compared."""
    out_dir = (out_root / case.out_subdir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    hamil_file = INPUTS_ROOT / case.data_dir / case.hamiltonian.file
    wfn_file = INPUTS_ROOT / case.data_dir / case.wavefunction.file

    observables = (resolve_observable_inputs(case.observables, INPUTS_ROOT / case.data_dir)
                   if test_type == TestType.BACKPROPAGATION else {})
    total_walkers = case.runparams.get("total_walkers", 1600)
    if snapshot:
        total_walkers = 50
    n_walkers = total_walkers // max(1, ranks)

    input_file = out_dir / "afqmc.json"
    write_input(
        input_file, hamil_file, wfn_file, case.walker,
        n_walkers_per_mpi_task=n_walkers,
        timestep=case.runparams.get("timestep", 0.01),
        observables=observables,
        snapshot=snapshot,
    )

    # Pass the input as an absolute path and let the child run in out_dir (so
    # AFQMC's native output lands there) rather than mutating our own cwd.
    run_cmd = shlex.split(mpiexec) + [afqmc_exec, "--compute", compute,
                                     str(input_file)]

    # for expected failures we can save time by not generating stack traces
    if test_type == TestType.EXPECT_FAILURE:
        run_cmd += ["--verbosity", "0"]

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

    results = out_dir / "results.h5"
    record_results(out_dir, return_code, ranks, run_time)
    reference = case.reference(snapshot)
    if regenerate:
        return store_reference(results, reference, test_type)
    if not reference.exists():
        print(f"  [compare] reference missing: {reference}")
        return None
    if snapshot:
        return compare_exactly(results, reference)
    return compare_statistically(results, reference, test_type)


# ============================================================================
# CLI
# ============================================================================

def main(argv=None) -> int:
    systems = build_systems()
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("system", nargs="?", help="system key, or 'all'")
    p.add_argument("--output-path", type=Path, help="root output directory, temporary by default. It is recommended to empty this directory before running the tests.")
    p.add_argument("--mpiexec", default="",
                   help='launcher prefix prepended to AFQMC_EXEC (e.g. "mpiexec -n 34")')
    p.add_argument("--compute", choices=["cpu", "gpu"], default="cpu",
                   help="compute device passed to AFQMC via --compute")
    p.add_argument("--timeout", type=float, default=None,
                   help="per-run timeout in minutes")
    p.add_argument("--dry-run", action="store_true",
                   help="list planned cases without running AFQMC")
    p.add_argument("--list", action="store_true", help="list available systems and exit")
    p.add_argument("--snapshot", action="store_true", help="run each test for a few steps and check for numerically exact agreement against a seeded snapshot")
    p.add_argument("--regenerate", action="store_true",
                   help="instead of comparing, copy each run's results.h5 into the "
                        "reference tree (snapshot_references with --snapshot, "
                        "statistical_references otherwise)")
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

    temp_dir = None
    if not args.dry_run and args.output_path is None:
        temp_dir = tempfile.TemporaryDirectory()
        args.output_path = Path(temp_dir.name)

    afqmc_exec = os.environ.get("AFQMC_EXEC")
    if not args.dry_run and not afqmc_exec:
        print("AFQMC_EXEC environment variable is not set.")
        return 2

    # Snapshots are exact comparisons, so a GPU run has to draw the same random numbers
    # as the host build the references were recorded with.
    if not args.dry_run and args.snapshot and args.compute == "gpu":
        if not device_rng_from_host_enabled(afqmc_exec):
            print("Snapshot tests on gpu need a build configured with "
                  "-DDEVICE_RNG_FROM_HOST=ON; otherwise the device draws its own "
                  "random numbers and no case can reproduce its snapshot.")
            return 2

    # The launcher is fixed for the whole run, so probe the rank count once.
    ranks = 1
    if not args.dry_run:
        ranks = detect_ranks(args.mpiexec)
        print(f"Detected {ranks} MPI rank(s) for launcher: {args.mpiexec!r}")

    total_pass = total_fail = total_skip = 0
    for name in selected:
        system = systems[name]
        all_cases = generate(system)
        has_bp = bool(system.observables)
        success = [c for c in all_cases if should_succeed(c) and not should_skip(c) and not (has_bp and should_backprop(c))]
        fail = [c for c in all_cases if not should_succeed(c) and not should_skip(c)]
        backprop = [c for c in all_cases if should_succeed(c) and has_bp and should_backprop(c) and not should_skip(c)]

        print(f"=== {name}: {len(success)} expected-success, "
              f"{len(fail)} expected-fail, {len(backprop)} back-propagation ===")

        for c in all_cases:
            if should_skip(c):
                print(f"  [SKIPPED] {c.out_subdir}")

        for test_type, group in (
                (TestType.EXPECT_FAILURE, fail),
                (TestType.EXPECT_SUCCESS, success),
                (TestType.BACKPROPAGATION, backprop)):
            for case in group:
                tag = f"[{test_type.name}] {case.out_subdir}"
                if args.dry_run:
                    print(f"  {tag}")
                    continue
                print(f"\n>>> {name} {tag}")

                ok = run_case(
                    case, test_type, args.output_path / name, args.mpiexec,
                    afqmc_exec, args.compute, ranks=ranks, timeout=args.timeout,
                    snapshot=args.snapshot, regenerate=args.regenerate)
                if ok is None:
                    print("  RESULT: SKIP")
                    total_skip += 1
                    continue
                if args.regenerate:
                    print(f"  RESULT: {'REGENERATED' if ok else 'NOT REGENERATED'}")
                else:
                    print(f"  RESULT: {'PASS' if ok else 'FAIL'}")
                total_pass += int(ok)
                total_fail += int(not ok)

    if not args.dry_run:
        if args.regenerate:
            print(f"\n==== {total_pass} regenerated, {total_fail} not regenerated, "
                  f"{total_skip} skipped ====")
        else:
            print(f"\n==== {total_pass} passed, {total_fail} failed, "
                  f"{total_skip} skipped ====")
    return 1 if total_fail else 0


if __name__ == "__main__":
    sys.exit(main())
