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
  4. records `afqmc.out` (+ AFQMC's native output) and a small `results.h5`,
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
from afqmctools.analysis.rdm import average_afqmc_rdm
from stats.scalar_dat import analyze_scalar_data

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

# Length of the equilibration phase discarded before averaging a scalar.dat column, in
# units of imaginary time. Snapshot runs are too short to equilibrate and are compared
# as they are, so they keep every block.
NEQUIL = 5.0
SNAPSHOT_NEQUIL = 0.0


@dataclass
class Case:
    hamiltonian: Hamiltonian
    wavefunction: Wavefunction
    walker: SpinSymm
    data_dir: str              # system dir shared by afqmc_inputs/ and both reference roots
    out_subdir: Path           # path (relative to output-path/system) for this run
    runparams: dict

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
                ))
    return cases


# ============================================================================
# Input file
# ============================================================================

def write_input(path: Path, hamil_file: Path, wfn_file: Path, walker: SpinSymm,
                n_walkers_per_mpi_task: int, timestep: float, run_bp: bool,
                snapshot: bool):
    steps = 10000
    equil_multiplier = 200
    population_control_interval = 10
    bp_measure_interval_multiplier = 40
    if snapshot:
        steps = 20
        equil_multiplier = 0
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
    if run_bp:
        execute["estimator"] = {
            "name": "back_propagation",
            "path_restoration": True,
            "bp_walker_ortho_interval": 10,
            "measure_interval_multiplier": bp_measure_interval_multiplier,
            "equil_multiplier": equil_multiplier,
            "onerdm": {"name": "one_rdm"},
        }
    execute["population_control_interval"] = population_control_interval
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


def _scalar_column(scalar_file: str, column: Optional[str], label: str, nequil: float):
    """Average a scalar.dat column, returning [value, stoch_error] or None."""
    try:
        params = dict(fname=scalar_file, xaxis="time", nequil=nequil, verbose=False)
        if column is not None:
            params["column"] = column
        v, dv = analyze_scalar_data(params)
        return np.array([v, dv])
    except Exception as e:  # noqa: BLE001
        print(f"  [warn] could not analyze {label}: {e}")
        return None


def record_results(out_dir: Path, return_code: int, ranks: int, run_time: float,
                   run_bp: bool, nequil: float):
    """Extract a results summary and write results.h5 (schema-compatible with the
    stored reference files: includes energy, weight, LogOvlpFactor and the
    error/warning message groups)."""
    out_text = (out_dir / "afqmc.out").read_text()
    with h5.File(out_dir / "results.h5", "w") as f:
        f.create_dataset("return_code", data=return_code)
        f.create_dataset("num_ranks", data=ranks)
        f.create_dataset("run_time_seconds", data=run_time)
        f.create_dataset("afqmc_is_finite", data=is_finite(out_text))
        f.create_dataset("afqmc_raised_error", data=raised_error(out_text))
        f.create_dataset("afqmc_raised_warning", data=raised_warning(out_text))
        f.create_dataset("input_file", data=(out_dir / "afqmc.json").read_text())
        _write_message_group(f, "error_messages", error_messages(out_text))
        _write_message_group(f, "warning_messages", warning_messages(out_text))
        if return_code == 0:
            scalar_file = str(out_dir / "qmc.s000.scalar.dat")
            energy = _scalar_column(scalar_file, None, "energy", nequil)
            if energy is not None:
                f.create_dataset("energy", data=energy)
            weight = _scalar_column(scalar_file, "weight", "weight", nequil)
            if weight is not None:
                f.create_dataset("weight", data=weight)
            # Log overlap: old name LogOvlpFactor, new name LogOvlp; support both.
            try:
                ovlp_col = ("LogOvlpFactor" if "LogOvlpFactor" in open(scalar_file).read()
                            else "LogOvlp")
            except OSError:
                ovlp_col = "LogOvlpFactor"
            log_ovlp = _scalar_column(scalar_file, ovlp_col, "LogOvlpFactor", nequil)
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


def _h5_messages(f: h5.File, group: str = "error_messages") -> set:
    g = f.get(group)
    if g is None:
        return set()
    n = int(g["num_messages"][()])
    prefix = group.split("_")[0]  # error_messages -> error_0, error_1, ...
    msgs = (g[f"{prefix}_{i}"][()] for i in range(n))
    return {m.decode() if isinstance(m, bytes) else str(m) for m in msgs}


def _compare_energy(ft: h5.File, fr: h5.File) -> bool:
    if "energy" not in ft or "energy" not in fr:
        print("  [compare] missing energy dataset")
        return False
    E, dE = ft["energy"][:]
    Eref, dEref = fr["energy"][:]
    sigma = np.sqrt(dE**2 + dEref**2)

    z = (E-Eref)/sigma
    p = 2 * scipy.stats.norm.sf(abs(z))
    if np.isnan(p) or p < SIGNIFICANCE_LEVEL:
        print(f"  [compare] energy mismatch: {E:.6f} ± {dE:.6f} vs {Eref:.6f} ± {dEref:.6f} (p = {p:.2g} < {SIGNIFICANCE_LEVEL})")
        return False

    print(f"  [compare] energy OK: {E:.6f} ± {dE:.6f} vs {Eref:.6f} ± {dEref:.6f} (p = {p:.2g} > {SIGNIFICANCE_LEVEL})")
    return True


def _compare_1rdm(ft: h5.File, fr: h5.File) -> bool:
    """Compare the back-propagated 1-RDM. A shape mismatch fails; otherwise we do Bonferroni adjusted test on the worst mismatch."""
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
            print(f"  [compare] avg_1rdm deterministic (sigma <= {MACHINE_EPS}) mismatch: |Δ| = {d[worst]:.3e} > {MACHINE_EPS} at idx = {worst}")
            return False

    if n_valid == 0:
        print(f"  [compare] avg_1rdm: no components with sigma > {MACHINE_EPS}; matched to machine precision")
        return True
    z_crit = scipy.stats.norm.ppf(1 - SIGNIFICANCE_LEVEL / (2 * n_valid))
    for part in ("real", "imag"):
        a, b = getattr(A, part), getattr(B, part)

        z = np.zeros_like(sigma)
        z[valid] = (a[valid] - b[valid]) / sigma[valid]
        worst = tuple(map(int, np.unravel_index(np.argmax(np.abs(z)), z.shape)))

        if np.abs(z[worst]) <= z_crit:
            print(f"  [compare] avg_1rdm {part} OK: worst component z = {z[worst]:.2f} <= {z_crit:.2f} at idx = {worst}")
        else:
            print(f"  [compare] avg_1rdm {part} mismatch: worst component z = {z[worst]:.2f} > {z_crit:.2f} at idx = {worst}")
            return False
    return True


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
            ok = _compare_energy(ft, fr)
            if test_type == TestType.BACKPROPAGATION:
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


def _input_settings(f: h5.File) -> dict:
    """The recorded input file, with the hamiltonian/wavefunction paths reduced to bare
    filenames: those are absolute, so they differ between checkouts even when the run
    settings are identical."""
    raw = f["input_file"][()]
    document = json.loads(raw.decode() if isinstance(raw, bytes) else raw)
    execute = document["afqmc"]["execute"]
    for key in ("hamiltonian", "wavefunction"):
        if key in execute and "filename" in execute[key]:
            execute[key]["filename"] = Path(execute[key]["filename"]).name
    return document


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

        # run_time_seconds is timing; input_file is compared as parsed settings below.
        ignored = {"run_time_seconds", "input_file"}
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

        if _input_settings(ft) != _input_settings(fs):
            print("  [compare] the snapshot was recorded with different run settings; "
                  "re-record it")
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

    run_bp = test_type == TestType.BACKPROPAGATION
    total_walkers = case.runparams.get("total_walkers", 1600)
    if snapshot:
        total_walkers = 50
    n_walkers = total_walkers // max(1, ranks)

    input_file = out_dir / "afqmc.json"
    write_input(
        input_file, hamil_file, wfn_file, case.walker,
        n_walkers_per_mpi_task=n_walkers,
        timestep=case.runparams.get("timestep", 0.01),
        run_bp=run_bp,
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
    record_results(out_dir, return_code, ranks, run_time, run_bp,
                   nequil=SNAPSHOT_NEQUIL if snapshot else NEQUIL)
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
        success = [c for c in all_cases if should_succeed(c) and not should_skip(c) and not (system.bp and should_backprop(c))]
        fail = [c for c in all_cases if not should_succeed(c) and not should_skip(c)]
        backprop = [c for c in all_cases if should_succeed(c) and system.bp and should_backprop(c) and not should_skip(c)]

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
