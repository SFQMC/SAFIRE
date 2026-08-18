# This file is distributed under the Apache License, Version 2.0 License.
# See LICENSE file in top directory for details.
#
# Copyright (c) 2021-2025 The Simons Foundation, Inc.
#
# You may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0

"""Convert VAFQMC (``hafqmc``) training artifacts into SAFIRE HDF5 inputs.

Reads a training ``hamiltonian.pkl`` and a ``checkpoint.pkl`` and writes

- ``ham.h5``     -- the physical Hamiltonian, ``/Hamiltonian/DenseFactorized``
- ``wfn.h5``     -- a single-determinant NOMSD trial
- ``ham_var.h5`` -- optional; the *inner* (variational) Hamiltonian rebuilt from the
  trained propagator parameters, carrying its timestep as the
  ``/Hamiltonian`` attribute ``inner_timestep``
- ``inner_timestep.json`` -- optional sidecar recording ``walker_type`` and the spin
  layout, neither of which SAFIRE can infer from the HDF5 files alone

``hafqmc`` is an optional dependency; install with the ``VAFQMC`` extra.

Examples
--------
::

    vafqmc_to_afqmc --hamiltonian hamiltonian.pkl --checkpoint checkpoint.pkl \\
                    --out-dir ./export --variational

The resulting SAFIRE input block::

    "hamiltonian":   { "filename": "ham.h5" },
    "wavefunction":  { "filename": "wfn.h5", "stochastic": true,
                       "inner_hamiltonian": { "filename": "ham_var.h5" } }
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any, Mapping, Sequence

import h5py
import numpy as np

from afqmctools.hamiltonian.io import write_dense
from afqmctools.utils.slater_types import _SlaterType, _slater_enum_map
from afqmctools.wavefunction.common import modified_gram_schmidt, write_wfn

from hafqmc.afqmc.utils import extract_params, normalize_ansatz_params
from hafqmc.hamiltonian import Hamiltonian, _has_spin
from hafqmc.utils import load_pickle


def _as_numpy(x: Any) -> np.ndarray:
    return np.asarray(x)


def _chol_to_safire_layout(chol: np.ndarray) -> np.ndarray:
    """Map a VAFQMC ``ceri`` [nchol, nmo, nmo] onto SAFIRE's ``L`` [nmo*nmo, nchol]."""
    chol = np.asarray(chol)
    if chol.ndim == 3:
        nchol, nmo, _ = chol.shape
        return chol.reshape(nchol, nmo * nmo).T.copy()
    if chol.ndim == 2:
        return chol.copy()
    raise ValueError(f"Expected Cholesky rank-2 or rank-3 array, got shape {chol.shape}.")


def write_hamiltonian_dense(
    hcore: np.ndarray,
    chol: np.ndarray,
    nelec: Sequence[int],
    nmo: int,
    enuc: float,
    filename: str | Path,
    *,
    real_chol: bool | None = None,
    ortho: np.ndarray | None = None,
) -> None:
    """Write ``/Hamiltonian/DenseFactorized`` after reshaping the Cholesky vectors.

    Thin wrapper over :func:`afqmctools.hamiltonian.io.write_dense`; it exists only to
    accept VAFQMC's rank-3 Cholesky layout. The file is removed first because
    ``write_dense`` opens in append mode, and these exports are re-run in place.
    """
    filename = Path(filename)
    if filename.exists():
        filename.unlink()
    write_dense(
        np.asarray(hcore),
        _chol_to_safire_layout(chol),
        (int(nelec[0]), int(nelec[1])),
        nmo,
        enuc=float(enuc),
        filename=filename,
        real_chol=real_chol,
        ortho=ortho,
    )


def write_nomsd_wfn(
    filename: str | Path,
    wfn_a: np.ndarray,
    wfn_b: np.ndarray,
    nelec: Sequence[int],
    *,
    walker_type: str = "closed",
    ci_coeffs: Sequence[complex] | None = None,
    orthonormalize: bool = True,
    thresh: float = 1e-8,
) -> None:
    """Write a single-determinant NOMSD trial wavefunction.

    Assembles the alpha/beta blocks into the ``(ndet, nmo, na + nb)`` array
    :func:`afqmctools.wavefunction.common.write_wfn` expects. Orthonormalization is done
    here rather than delegated because ``write_wfn`` only orthonormalizes matrices that
    fail its check, whereas an exported trial is always put through Gram-Schmidt.
    """
    walker = _slater_enum_map(walker_type.lower() if isinstance(walker_type, str) else walker_type)
    if walker == _SlaterType.NONCOLLINEAR:
        raise ValueError(
            "A spin-mixed trial is written on the 2*nmo spin-orbital space as a "
            "COLLINEAR problem with ndown == 0, not as NONCOLLINEAR; see "
            "to_polarized_operator.")
    uhf = walker == _SlaterType.COLLINEAR

    nalpha, nbeta = int(nelec[0]), int(nelec[1])
    nmo = wfn_a.shape[0]

    psi_a = _as_numpy(wfn_a)[:, :nalpha].astype(np.complex128, copy=True)
    psi_b = _as_numpy(wfn_b)[:, :nbeta].astype(np.complex128, copy=True)
    psi_a[np.abs(psi_a) < thresh] = 0.0
    psi_b[np.abs(psi_b) < thresh] = 0.0

    if orthonormalize:
        psi_a = modified_gram_schmidt(psi_a)
        if uhf and nbeta > 0:
            psi_b = modified_gram_schmidt(psi_b)

    if uhf:
        wfn_block = np.zeros((1, nmo, nalpha + nbeta), dtype=np.complex128)
        wfn_block[0, :, :nalpha] = psi_a
        wfn_block[0, :, nalpha:nalpha + nbeta] = psi_b
    else:
        wfn_block = psi_a[None, :, :]

    coeffs = np.array([1.0 + 0.0j] if ci_coeffs is None else ci_coeffs, dtype=np.complex128)

    filename = Path(filename)
    if filename.exists():
        filename.unlink()
    # Gram-Schmidt has already run above, so write_wfn must not repeat it.
    write_wfn(filename, (coeffs, wfn_block), walker, (nalpha, nbeta), nmo,
              orthonormalize=False)


def load_hamiltonian_pickle(path: str | Path) -> Hamiltonian:
    h1e, ceri, enuc, wfn0, aux = load_pickle(path)
    return Hamiltonian(h1e, ceri, enuc, wfn0, aux if aux is not None else {})


def load_ansatz_params(checkpoint_path: str | Path) -> Mapping[str, Any]:
    payload = load_pickle(checkpoint_path)
    params = normalize_ansatz_params(extract_params(payload))
    ansatz = params.get("params", params)
    if not isinstance(ansatz, Mapping):
        raise ValueError("Checkpoint does not contain a mapping of ansatz parameters.")
    return ansatz


def _propagator_block(ansatz: Mapping[str, Any]) -> Mapping[str, Any]:
    if "propagators_0" not in ansatz:
        raise KeyError("Checkpoint ansatz is missing 'propagators_0'.")
    return ansatz["propagators_0"]


def _report_split_error(ts_h: np.ndarray, hmf_slices: Sequence[np.ndarray]) -> dict[str, float]:
    """Report the one approximation this conversion makes.

    The trained propagator applies ``exp(-ts_h[1] hmf_1) exp(V) exp(-ts_h[0] hmf_0)``,
    with distinct left and right one-body operators. SAFIRE applies a symmetric
    ``exp(-G/2) exp(V) exp(-G/2)`` built from a single generator ``G = sum_i ts_h[i]
    hmf_i``. The two agree only if the slices commute AND ``ts_h`` is symmetric.

    Both diagnostics are printed on every conversion because a symmetric ``ts_h`` alone
    does NOT make the fold exact -- the slices must also commute, and in practice they
    do not. An exact conversion would need distinct left/right one-body operators in
    SAFIRE's inner propagator.
    """
    ts = np.asarray(ts_h, dtype=float).ravel()
    ratio = float(ts.max() / ts.min()) if ts.min() > 0 else float("inf")
    cmax = 0.0
    for i in range(len(hmf_slices)):
        for j in range(i + 1, len(hmf_slices)):
            a, b = np.asarray(hmf_slices[i]), np.asarray(hmf_slices[j])
            cmax = max(cmax, float(np.abs(a @ b - b @ a).max()))
    exact = cmax <= 1e-10 and abs(ratio - 1.0) <= 1e-10
    print(f"[from_vafqmc] trained one-body split: ts_h = {np.array2string(ts, precision=5)} "
          f"(asymmetry {ratio:.2f}x), max||[hmf_i, hmf_j]|| = {cmax:.3g}", file=sys.stderr)
    if not exact:
        print("[from_vafqmc] WARNING: the trained left/right one-body asymmetry is folded "
              "into a single generator that SAFIRE splits symmetrically. This is an "
              "approximation, not an identity: the converted propagator does not "
              "reproduce the trained one field-sample by field-sample. See "
              "_report_split_error.__doc__.", file=sys.stderr)
    return dict(ts_h_asymmetry=ratio, max_commutator=cmax, split_is_exact=bool(exact))


def _all_hmf(prop: Mapping[str, Any]) -> list[np.ndarray]:
    """Every trained one-body slice ``hmf_ops_i['hmf']``, in slice order.

    A single B_T slice (``init_tsteps = [dt]``) with ``timevarying: hmf`` has ``nts_h ==
    2`` distinct slices that diverge under training. Both are needed; keeping only the
    last silently discards half the trained one-body operator.
    """
    hmf_keys = sorted((k for k in prop if k.startswith("hmf_ops_")),
                      key=lambda k: int(k.rsplit("_", 1)[1]))
    if not hmf_keys:
        raise KeyError("Propagator block has no hmf_ops_* parameters.")
    return [_as_numpy(prop[k]["hmf"]) for k in hmf_keys]


def _select_vhs(prop: Mapping[str, Any]) -> np.ndarray:
    if "vhs_ops_0" not in prop or "vhs" not in prop["vhs_ops_0"]:
        raise KeyError("Propagator block is missing vhs_ops_0/vhs.")
    return _as_numpy(prop["vhs_ops_0"]["vhs"])


def _trained_timestep(prop: Mapping[str, Any], *, sqrt_tsvpar: bool = True,
                      init_dt: float | None = None) -> float:
    """The B_T two-body timestep SAFIRE must use for its inner propagator.

    The trained propagator applies the auxiliary-field exponent with prefactor
    ``step_v``: ``1j*ts_v`` when ``sqrt_tsvpar`` (so ``ts_v`` holds ``sqrt(dt)``), else
    ``sqrt(-ts_v)``. SAFIRE rebuilds it as ``sqrt(-dt) sum_k x_k L_k``, so matching the
    trained field weight requires ``dt = ts_v**2`` (or ``dt = ts_v``). Only a single
    B_T slice (``nts_v == 1``) is supported.

    The nominal ``init_tsteps`` is NOT the trained timestep: ``ts_v`` is trainable under
    ``parametrize: all`` and drifts substantially. ``init_dt`` is the fallback for
    checkpoints trained with frozen timesteps, where ``ts_v`` never enters the params
    tree and the nominal value is therefore the correct one.
    """
    if "ts_v" not in prop:
        if init_dt is not None:
            return float(init_dt)
        raise KeyError("Propagator block is missing ts_v and no init_dt fallback was provided; "
                       "cannot recover the trained timestep.")
    ts_v = _as_numpy(prop["ts_v"]).astype(float).ravel()
    if ts_v.size != 1:
        raise ValueError(
            f"Expected a single B_T slice (nts_v == 1) but ts_v has {ts_v.size} entries; "
            "the (hcore, L, dt) inner-Hamiltonian conversion assumes one slice.")
    dt = float(ts_v[0] ** 2) if sqrt_tsvpar else float(abs(ts_v[0]))
    if not (1e-4 < dt < 1.0):
        print(f"WARNING: recovered inner timestep dt={dt:.4g} is outside the expected "
              f"(1e-4, 1) range (ts_v={ts_v[0]:.4g}, sqrt_tsvpar={sqrt_tsvpar}). "
              "Check the sqrt_tsvpar convention against the training config.", file=sys.stderr)
    return dt


def is_spin_doubled(hmf: np.ndarray, vhs: np.ndarray, nmo: int) -> bool:
    """Were these operators trained with ``spin_mixing: true`` (GHF, 2*nmo basis)?"""
    return hmf.shape[-1] == 2 * nmo and vhs.shape[-1] == 2 * nmo


def spin_mixing_magnitude(hmf: np.ndarray, vhs: np.ndarray, nmo: int) -> dict[str, float]:
    """Frobenius norms of the spin-off-diagonal blocks against the alpha-alpha block.

    Reported so that the size of what a spin-restricted conversion would discard is a
    measured number in the log rather than an assertion.
    """
    def blocks(a):
        return np.linalg.norm(a[..., :nmo, :nmo]), np.linalg.norm(a[..., :nmo, nmo:]), \
            np.linalg.norm(a[..., nmo:, :nmo]), np.linalg.norm(a[..., nmo:, nmo:])

    h_aa, h_ab, h_ba, h_bb = blocks(np.asarray(hmf))
    v_aa, v_ab, v_ba, v_bb = blocks(np.asarray(vhs))
    return dict(hmf_aa=float(h_aa), hmf_ab=float(h_ab), hmf_ba=float(h_ba), hmf_bb=float(h_bb),
                vhs_aa=float(v_aa), vhs_ab=float(v_ab), vhs_ba=float(v_ba), vhs_bb=float(v_bb),
                offdiag_ratio=float(max(v_ab, v_ba) / v_aa) if v_aa > 0 else float("inf"))


def _refuse_restricted_export(nmo: int, mags: dict[str, float]) -> None:
    """A spin-mixed trial cannot be written in the spin-restricted layout: fail, do not truncate.

    The spin-restricted layout can hold only the alpha-alpha block of each operator.
    Continuing with a warning would ship a trial that silently omits the rest, so this
    is a hard error. ``--spin-layout polarized`` writes the same operator exactly.
    """
    raise SystemExit(
        "FATAL: this checkpoint was trained with spin_mixing: true, so its propagator\n"
        "  operators are\n"
        f"  GHF spin-doubled ({2*nmo} x {2*nmo}). The spin-restricted layout can only hold the\n"
        "  alpha-alpha block, and for this checkpoint that block is NOT the whole operator:\n"
        f"    ||L_aa||_F = {mags['vhs_aa']:.3f}   ||L_ab||_F = {mags['vhs_ab']:.3f}   "
        f"||L_ba||_F = {mags['vhs_ba']:.3f}   "
        f"(off-diagonal / diagonal = {mags['offdiag_ratio']:.2f})\n"
        f"    ||h_aa||_F = {mags['hmf_aa']:.3f}   ||h_ab||_F = {mags['hmf_ab']:.3f}   "
        f"||h_bb||_F = {mags['hmf_bb']:.3f}\n"
        "  Use --spin-layout polarized: it writes the SAME operator exactly, on the 2*nmo\n"
        "  spin-orbital space, as a COLLINEAR/ndown=0 problem that SAFIRE already supports.")


def to_polarized_operator(a: np.ndarray, nmo: int) -> np.ndarray:
    """Embed a spin-restricted operator into the 2*nmo spin-orbital space as blockdiag(a, a).

    An operator that is already spin-doubled is returned untouched. This is the identity
    map on physics: for a spin-independent one-body ``h``, ``blockdiag(h, h)`` is the
    same operator written on spin orbitals, and for Cholesky vectors ``blockdiag(L, L)``
    reproduces the spin-independent two-body term exactly -- including the ``v0 = -1/2
    sum_k L_k L_k`` that SAFIRE re-adds internally, since ``blockdiag(L, L)**2 ==
    blockdiag(L**2, L**2)``.
    """
    a = np.asarray(a)
    if a.shape[-1] == 2 * nmo:
        return a
    if a.shape[-1] != nmo:
        raise ValueError(
            f"operator has trailing dimension {a.shape[-1]}, expected {nmo} or {2*nmo}")
    n2 = 2 * nmo
    out = np.zeros(a.shape[:-2] + (n2, n2), dtype=a.dtype)
    out[..., :nmo, :nmo] = a
    out[..., nmo:, nmo:] = a
    return out


def to_polarized_orbitals(wfn_a: np.ndarray, wfn_b: np.ndarray, nmo: int) -> np.ndarray:
    """Block-diagonal spin-orbital Slater matrix [[wfn_a, 0], [0, wfn_b]], shape (2*nmo, na+nb)."""
    wfn_a = _as_numpy(wfn_a)
    wfn_b = _as_numpy(wfn_b)
    if wfn_a.shape[0] == 2 * nmo:
        return wfn_a          # already a spin-orbital (GHF) matrix
    na, nb = wfn_a.shape[1], wfn_b.shape[1]
    out = np.zeros((2 * nmo, na + nb), dtype=np.complex128)
    out[:nmo, :na] = wfn_a
    out[nmo:, na:] = wfn_b
    return out


def _maybe_unspin_orbitals(
    wfn_a: np.ndarray,
    wfn_b: np.ndarray,
    nmo: int,
) -> tuple[np.ndarray, np.ndarray]:
    if wfn_a.shape[0] == 2 * nmo:
        print(
            "Detected GHF-doubled orbitals in checkpoint; exporting spatial rows only.",
            file=sys.stderr,
        )
        wfn_a = wfn_a[:nmo]
        if wfn_b.shape[0] == 2 * nmo:
            wfn_b = wfn_b[:nmo]
    return wfn_a, wfn_b


def extract_orbitals(
    ansatz: Mapping[str, Any],
    wfn0: Any,
) -> tuple[np.ndarray, np.ndarray]:
    """Return (wfn_a, wfn_b) Slater matrices [nmo, nelec]."""
    if "wfn_a" in ansatz:
        wfn_a = _as_numpy(ansatz["wfn_a"])
        if "wfn_b" in ansatz:
            wfn_b = _as_numpy(ansatz["wfn_b"])
        else:
            wfn_b = wfn_a
        return wfn_a, wfn_b

    if _has_spin(wfn0):
        return _as_numpy(wfn0[0]), _as_numpy(wfn0[1])
    wfn = _as_numpy(wfn0)
    return wfn, wfn


def nelec_from_orbitals(wfn_a: np.ndarray, wfn_b: np.ndarray) -> tuple[int, int]:
    return int(wfn_a.shape[1]), int(wfn_b.shape[1])


def infer_walker_type(nelec: Sequence[int], requested: str,
                      wfn_a: np.ndarray | None = None, wfn_b: np.ndarray | None = None,
                      tol: float = 1e-8) -> str:
    """SAFIRE walker type for this trial, decided on the orbitals where possible.

    Electron counts alone are not enough: a broken-symmetry singlet has ``na == nb`` and
    is COLLINEAR, not CLOSED. Deciding on counts would write such a trial as CLOSED,
    silently replacing its beta orbitals with its alpha ones.

    So when the orbitals are available, compare them: identical to ``tol`` means CLOSED,
    otherwise COLLINEAR. The count-based fallback is kept only for callers that have no
    orbitals, and it announces itself.
    """
    if requested != "auto":
        return requested
    if int(nelec[0]) != int(nelec[1]):
        return "collinear"
    if wfn_a is None or wfn_b is None:
        print("WARNING: infer_walker_type called without orbitals; falling back to electron "
              "counts, which cannot see a broken-symmetry singlet (na == nb but COLLINEAR).",
              file=sys.stderr)
        return "closed"
    dmax = float(np.abs(_as_numpy(wfn_a) - _as_numpy(wfn_b)).max())
    if dmax > tol:
        print(f"  broken-symmetry trial detected (max|wfn_a - wfn_b| = {dmax:.3g} > {tol:g}) "
              f"-> walker_type = collinear", file=sys.stderr)
        return "collinear"
    return "closed"


def build_variational_hamiltonian(
    hamil: Hamiltonian,
    ansatz: Mapping[str, Any],
    wfn_a: np.ndarray,
    wfn_b: np.ndarray,
    *,
    sqrt_tsvpar: bool = True,
    abs_tshpar: bool = True,
    init_dt: float | None = None,
    spin_layout: str = "restricted",
) -> tuple[np.ndarray, np.ndarray, float, float]:
    """Inner (variational) ``hcore`` / Cholesky ``L`` / ``enuc`` / trained timestep ``dt``.

    Per outer step the trained propagator applies (single-slice ansatz, ``nts_v == 1``
    so ``nts_h == 2``)::

        exp(-ts_h[1] hmf_1) exp(1j ts_v sum_k x_k vhs_k) exp(-ts_h[0] hmf_0)

    SAFIRE rebuilds a symmetric physical Trotter step from ``(hcore, L)`` and one ``dt``::

        B_S = exp(-H1/2) exp(sqrt(-dt) sum_k x_k L_k) exp(-H1/2)
        H1  = dt*hcore + dt*vexx + mf,   vexx = -1/2 sum_k L_k L_k

    so its total one-body generator is ``-(dt*hcore + dt*vexx + mf)``. ``vexx`` and the
    force bias ``mf`` are added by SAFIRE unconditionally and cannot be controlled from
    the HDF5 file, which fixes the conversion:

    * ``dt = ts_v**2``, so the field weight ``sqrt(-dt)`` equals the trained ``1j*ts_v``.
    * ``L = vhs``, exact given that ``dt``.
    * ``hcore = (sum_i ts_h[i] hmf_i) / dt - vexx``. The trained total one-body generator
      is ``sum_i ts_h[i] hmf_i`` (each slice already carries the ``-1/2 v0`` folded in by
      ``make_proj_op``); equating SAFIRE's total to it gives this expression. The
      ``- vexx`` is essential -- without it SAFIRE re-adds a second ``-1/2 v0``, and the
      double count scaled by ``dt`` drives the energy far below exact. ``mf`` is left in:
      mean-field subtraction is an exact identity on the propagator, affecting variance
      and not the energy.

    This folds in BOTH ``hmf`` slices and the trained (asymmetric) ``ts_h``. At the
    untrained initialisation (``ts_h == [dt/2, dt/2]``, ``hmf_0 == hmf_1``) SAFIRE's
    one-body reduces to ``sum ts_h hmf == hmf``, the physically correct value. The only
    residual approximation is the symmetric-Trotter treatment of the trained left/right
    asymmetry; see :func:`_report_split_error`.
    """
    prop = _propagator_block(ansatz)
    hmf_slices = _all_hmf(prop)
    chol = _select_vhs(prop)
    dt = _trained_timestep(prop, sqrt_tsvpar=sqrt_tsvpar, init_dt=init_dt)

    if "ts_h" in prop:
        ts_h = _as_numpy(prop["ts_h"]).astype(float).ravel()
    elif init_dt is not None:
        # frozen timesteps: ts_h is the init half-step split convolve([dt],[.5,.5]) =
        # [dt/2, dt/2] -- symmetric, so the symmetric-Trotter fold has no asymmetry error.
        ts_h = np.array([init_dt / 2.0, init_dt / 2.0])
    else:
        ts_h = None
    if ts_h is not None and abs_tshpar:
        # match the propagator's abs_tshpar guard: the applied one-body coefficient is
        # |ts_h| (forward imaginary time), not the raw signed parameter.
        ts_h = np.abs(ts_h)
    if ts_h is not None and ts_h.size == len(hmf_slices):
        one_body_total = sum(float(t) * h for t, h in zip(ts_h, hmf_slices))
        _report_split_error(ts_h, hmf_slices)
    else:
        # ts_h absent or shape-mismatched (not expected for this ansatz): treat the mean
        # hmf as if the full dt were split evenly across the slices.
        print(f"WARNING: ts_h {'missing' if ts_h is None else f'has {ts_h.size} entries'} "
              f"but there "
              f"are {len(hmf_slices)} hmf slices; assuming an even dt split over the mean hmf.",
              file=sys.stderr)
        one_body_total = dt * (sum(hmf_slices) / len(hmf_slices))

    # SAFIRE's vexx from the exported L (== vhs). Subtracting it here makes SAFIRE's
    # unconditional re-addition of dt*vexx net exactly the trained one-body generator.
    vexx = -0.5 * np.einsum("kpr,krs->ps", chol, chol)
    hcore = one_body_total / dt - vexx

    nmo = int(hamil.h1e.shape[-1])
    doubled = is_spin_doubled(np.asarray(hcore), np.asarray(chol), nmo)
    if spin_layout == "polarized":
        # Keep the trained operators exactly as trained: a no-op when they are already
        # spin-doubled, and an embedding as blockdiag(a, a) when they are not. Either
        # way nothing is discarded, which is the point of this layout.
        hcore = to_polarized_operator(hcore, nmo)
        chol = to_polarized_operator(chol, nmo)
    elif doubled:
        _refuse_restricted_export(
            nmo, spin_mixing_magnitude(np.asarray(hcore), np.asarray(chol), nmo))

    trial = (wfn_a, wfn_b) if _has_spin((wfn_a, wfn_b)) else wfn_a
    _, _, enuc_var = hamil.make_proj_op(trial)
    return hcore, chol, float(np.real(enuc_var)), dt


def vafqmc_to_afqmc(
    hamiltonian_path: str | Path,
    checkpoint_path: str | Path,
    out_dir: str | Path,
    *,
    variational: bool = False,
    walker_type: str = "auto",
    orthonormalize: bool = True,
    init_dt: float | None = None,
    spin_layout: str = "auto",
) -> dict[str, Path]:
    """Convert a VAFQMC (pickle) checkpoint into SAFIRE HDF5 inputs.

    Returns a mapping of output label to path.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    hamil = load_hamiltonian_pickle(hamiltonian_path)
    ansatz = load_ansatz_params(checkpoint_path)

    h1e = _as_numpy(hamil.h1e)
    ceri = _as_numpy(hamil.ceri)
    enuc = float(hamil.enuc)
    nmo = int(h1e.shape[-1])

    # ---- resolve the spin layout before anything is written -----------------------
    prop = _propagator_block(ansatz)
    doubled = is_spin_doubled(_all_hmf(prop)[0], _select_vhs(prop), nmo)
    if spin_layout == "auto":
        spin_layout = "polarized" if doubled else "restricted"
    if spin_layout not in {"restricted", "polarized"}:
        raise ValueError(f"unknown spin_layout {spin_layout!r}")
    if doubled and spin_layout == "restricted":
        _refuse_restricted_export(
            nmo, spin_mixing_magnitude(_all_hmf(prop)[0], _select_vhs(prop), nmo))
    if doubled:
        mags = spin_mixing_magnitude(_all_hmf(prop)[0], _select_vhs(prop), nmo)
        print(f"  spin-mixed checkpoint (spin_mixing: true): ||L_ab||/||L_aa|| = "
              f"{mags['offdiag_ratio']:.3f} -> spin_layout = polarized", file=sys.stderr)

    wfn_a, wfn_b = extract_orbitals(ansatz, hamil.wfn0)
    wfn_a, wfn_b = _maybe_unspin_orbitals(wfn_a, wfn_b, nmo)
    if wfn_a.shape[0] != nmo:
        raise ValueError(
            f"Orbital dimension mismatch: h1e has nmo={nmo}, wfn has {wfn_a.shape[0]} rows."
        )

    nelec = nelec_from_orbitals(wfn_a, wfn_b)
    ortho = hamil.aux.get("orth_mat") if hamil.aux else None
    if ortho is not None:
        ortho = _as_numpy(ortho)

    outputs: dict[str, Path] = {}

    if spin_layout == "polarized":
        # The spin-orbital (COLLINEAR, ndown == 0) layout, at nmo' = 2*nmo.
        #
        # This is how a spin-MIXED trial reaches SAFIRE at all. SAFIRE stores Cholesky
        # vectors as Likn(nspin*npol, NMO, NMO, ncv) and vHS returns npol stacked
        # NMO x NMO blocks, block-diagonal in polarization -- there is no slot for L_ab,
        # so a NONCOLLINEAR (npol=2) Hamiltonian cannot carry spin mixing. Writing the
        # same operator on the 2*nmo spin-orbital space as a polarized problem makes
        # Likn (1, 2NMO, 2NMO, ncv), which holds the spin-mixed operator with no change
        # to SAFIRE. COLLINEAR with ndown == 0 is the supported spelling for a polarized
        # system; it replaced the removed FULLYPOLARIZED walker type.
        n2 = 2 * nmo
        nel = (nelec[0] + nelec[1], 0)
        wfn_pol = to_polarized_orbitals(wfn_a, wfn_b, nmo)
        walker = "collinear"
        # `Hamiltonian/X` is an AO -> orthonormal-basis matrix of shape (nao, nmo). It
        # has no meaning at 2*nmo and nothing in this path consumes it, so it is
        # deliberately not written.
        ham_path = out_dir / "ham.h5"
        write_hamiltonian_dense(
            to_polarized_operator(h1e, nmo),
            to_polarized_operator(ceri, nmo),
            nel, n2, enuc, ham_path,
        )
        outputs["ham"] = ham_path
        wfn_path = out_dir / "wfn.h5"
        write_nomsd_wfn(wfn_path, wfn_pol, wfn_pol[:, :0], nel,
                        walker_type=walker, orthonormalize=orthonormalize)
        outputs["wfn"] = wfn_path
    else:
        walker = infer_walker_type(nelec, walker_type, wfn_a, wfn_b)
        ham_path = out_dir / "ham.h5"
        write_hamiltonian_dense(h1e, ceri, nelec, nmo, enuc, ham_path, ortho=ortho)
        outputs["ham"] = ham_path
        wfn_path = out_dir / "wfn.h5"
        write_nomsd_wfn(wfn_path, wfn_a, wfn_b, nelec,
                        walker_type=walker, orthonormalize=orthonormalize)
        outputs["wfn"] = wfn_path

    if variational:
        hcore_var, chol_var, enuc_var, dt_var = build_variational_hamiltonian(
            hamil, ansatz, wfn_a, wfn_b, init_dt=init_dt, spin_layout=spin_layout
        )
        ham_var_path = out_dir / "ham_var.h5"
        # The inner Hamiltonian must be written on the same space as ham.h5/wfn.h5:
        # SAFIRE builds the inner propagator against the outer walkers, so an (nmo)
        # inner operator beside (2*nmo) walkers is a dimension mismatch.
        var_nelec = (nelec[0] + nelec[1], 0) if spin_layout == "polarized" else nelec
        var_nmo = 2 * nmo if spin_layout == "polarized" else nmo
        write_hamiltonian_dense(
            hcore_var,
            chol_var,
            var_nelec,
            var_nmo,
            enuc_var,
            ham_var_path,
            ortho=None if spin_layout == "polarized" else ortho,
        )
        outputs["ham_var"] = ham_var_path

        # Stamp the trained timestep into ham_var.h5 itself. dt parameterizes the
        # propagator built FROM THIS HAMILTONIAN, so the operator and its timestep must
        # travel in one file: SAFIRE reads this attribute and has no input key and no
        # default for it.
        with h5py.File(ham_var_path, "r+") as fh5:
            fh5["Hamiltonian"].attrs["inner_timestep"] = float(dt_var)

        # The sidecar carries walker_type, which SAFIRE cannot infer (a broken-symmetry
        # singlet has na == nb but is COLLINEAR). inner_timestep is duplicated here for
        # humans; ham_var.h5 is the authority and the only thing SAFIRE reads.
        meta_path = out_dir / "inner_timestep.json"
        meta_path.write_text(json.dumps({
            "inner_timestep": dt_var,
            "walker_type": walker,
            "spin_layout": spin_layout,
            "nmo": var_nmo,
            "nelec": list(var_nelec),
            "spin_layout_note": (
                "'polarized' means every file here is written on the 2*nmo SPIN-ORBITAL space as a "
                "COLLINEAR problem with ndown == 0 -- the spelling that replaced FULLYPOLARIZED. "
                "It is required whenever the checkpoint was trained with spin_mixing: true, "
                "because "
                "SAFIRE's Cholesky is block-diagonal in polarization and cannot hold L_ab. Set "
                "walker_set.walker_type = COLLINEAR and take nmo/nelec from this file, not "
                "from the "
                "training Hamiltonian."),
            "note": "Trained B_T two-body timestep dt = ts_v**2 (sqrt_tsvpar=True). Use as SAFIRE "
                    "wavefunction.inner_propagator.timestep so the inner field weight sqrt(-dt) "
                    "matches the trained 1j*ts_v. See from_vafqmc.build_variational_hamiltonian. "
                    "walker_type ('closed'/'collinear') is recorded so the run driver sets "
                    "walker_set.walker_type without re-inferring from electron counts (a "
                    "broken-symmetry singlet has na == nb but is COLLINEAR).",
        }, indent=2))
        outputs["inner_timestep_meta"] = meta_path
        print(f"  trained inner timestep dt = {dt_var:.6g} (from ts_v**2) -> {meta_path.name}",
              file=sys.stderr)

    return outputs


#: Alias for the name this conversion carried before it moved into afqmctools.
export_safire = vafqmc_to_afqmc
