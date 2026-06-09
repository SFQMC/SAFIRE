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
Results class for storing and managing AFQMC calculation results.

This module provides a flexible way to create, store, and retrieve AFQMC
calculation results in HDF5 format.
"""

import re
from typing import Optional, Union, Dict, Any
from pathlib import Path

import h5py as h5
import numpy as np

from stats.scalar_dat import analyze_scalar_data
from afqmctools.analysis.rdm import average_afqmc_rdm


def afqmc_raised_warning(fname: str) -> bool:
    """Check if AFQMC output file contains warnings."""
    with open(fname, 'r') as f:
        result = re.search(r"\[warning\]", f.read())
    return result is not None


def afqmc_raised_error(fname: str) -> bool:
    """Check if AFQMC output file contains errors."""
    with open(fname, 'r') as f:
        result = re.search(r"\[error\]", f.read())
    return result is not None


def afqmc_out_is_finite(fname: str) -> bool:
    """Check that AFQMC output contains finite values (not NaN)."""
    with open(fname, 'r') as f:
        result = re.search(r"\([-]?nan,|[-]nan\)", f.read())
    return result is None


def afqmc_warning_message(fname: str) -> set:
    """Extract all warning messages from AFQMC output file."""
    if not afqmc_raised_warning(fname):
        return set()
    with open(fname, 'r') as f:
        results = re.findall(r"(\[warning\]) (.+)", f.read())
    results = [w[1].lstrip().rstrip() for w in results]
    return set(results)


def afqmc_error_message(fname: str) -> set:
    """Extract all error messages from AFQMC output file."""
    if not afqmc_raised_error(fname):
        return set()
    with open(fname, 'r') as f:
        text = re.sub(r'\x1b\[[0-9;]*m', '', f.read())
        results = re.findall(r"(\[error\]) (.+)", text)

    results = set(results)
    results = [
        re.sub(r'\x1b\[[0-9;]*m', '', str(w[1])).lstrip().rstrip() 
        for w in results
    ]
    results = [
        w for w in results
        if not re.match(r"\*{10,}", w) and 
           not re.match(r"APPLICATION ABORT: Fatal Error\.", w)
    ]
    return set(results)


class Results:
    """
    A class to store and manage AFQMC calculation results.

    This class can be constructed in three ways:
    1. From AFQMC run output files
    2. From user-provided data dictionary
    3. From an existing results.h5 file

    Attributes
    ----------
    return_code : int
        Exit code from AFQMC run (0 = success)
    run_time_seconds : float
        Runtime of the calculation in seconds
    afqmc_raised_warning : bool
        Whether warnings were raised
    afqmc_raised_error : bool
        Whether errors were raised
    afqmc_is_finite : bool
        Whether all output values are finite
    input_file : str
        Contents of the AFQMC input file
    warning_messages : set
        Set of warning messages (if any)
    error_messages : set
        Set of error messages (if any)
    energy : np.ndarray
        Array [E, dE] with energy and uncertainty
    weight : np.ndarray
        Array [weight, dweight] with walker weight and uncertainty
    log_ovlp_factor : np.ndarray
        Array [log_ovlp, dlog_ovlp] with overlap factor and uncertainty
    avg_1rdm : np.ndarray
        Averaged 1-particle reduced density matrix (if available)
    avg_1rdm_stoch_error : np.ndarray
        Stochastic error in 1-RDM (if available)
    """

    def __init__(self):
        """Initialize empty Results object."""
        self.return_code: Optional[int] = None
        self.run_time_seconds: float = -1.0
        self.afqmc_raised_warning: bool = False
        self.afqmc_raised_error: bool = False
        self.afqmc_is_finite: bool = True
        self.input_file: Optional[str] = None
        self.warning_messages: set = set()
        self.error_messages: set = set()
        self.energy: Optional[np.ndarray] = None
        self.weight: Optional[np.ndarray] = None
        self.log_ovlp_factor: Optional[np.ndarray] = None
        self.avg_1rdm: Optional[np.ndarray] = None
        self.avg_1rdm_stoch_error: Optional[np.ndarray] = None

    @classmethod
    def from_afqmc_run(
        cls,
        return_code: int,
        run_time_secs: float = -1.0,
        nequil: float = 5.0,
        run_bp: bool = False,
        output_file: str = "afqmc.out",
        input_file: str = "afqmc.json",
        scalar_file: str = "qmc.s000.scalar.dat"
    ) -> "Results":
        """
        Construct Results from AFQMC run output files.

        Parameters
        ----------
        return_code : int
            Exit code from the AFQMC calculation
        run_time_secs : float, optional
            Runtime in seconds (default: -1.0)
        nequil : float, optional
            Equilibration time for data analysis (default: 5.0)
        run_bp : bool, optional
            Whether back-propagation was run (default: False)
        output_file : str, optional
            Name of AFQMC output file (default: "afqmc.out")
        input_file : str, optional
            Name of AFQMC input file (default: "afqmc.json")
        scalar_file : str, optional
            Name of scalar data file (default: "qmc.s000.scalar.dat")

        Returns
        -------
        Results
            Populated Results object
        """
        results = cls()
        results.return_code = return_code
        results.run_time_seconds = run_time_secs

        # Check for warnings and errors
        results.afqmc_raised_warning = afqmc_raised_warning(output_file)
        results.warning_messages = afqmc_warning_message(output_file)

        results.afqmc_raised_error = afqmc_raised_error(output_file)
        results.error_messages = afqmc_error_message(output_file)

        results.afqmc_is_finite = afqmc_out_is_finite(output_file)

        # Read input file
        with open(input_file, 'r') as f:
            results.input_file = f.read()

        # Process results if run was successful
        if return_code == 0:
            # Energy
            E, dE = analyze_scalar_data(
                dict(fname=scalar_file, xaxis="time", nequil=nequil)
            )
            results.energy = np.array([E, dE])

            # Walker weight
            weight, dweight = analyze_scalar_data(
                dict(fname=scalar_file, xaxis="time", nequil=nequil, column="weight")
            )
            results.weight = np.array([weight, dweight])

            # LogOvlpFactor
            log_ovlp, dlog_ovlp = analyze_scalar_data(
                dict(fname=scalar_file, xaxis="time", nequil=nequil, column="LogOvlpFactor")
            )
            results.log_ovlp_factor = np.array([log_ovlp, dlog_ovlp])

            # 1-RDM if back-propagation was run
            if run_bp:
                rho_avg, delta_rho = average_afqmc_rdm()
                results.avg_1rdm = rho_avg
                results.avg_1rdm_stoch_error = delta_rho

        return results

    @classmethod
    def from_data(
        cls,
        return_code: int = 0,
        run_time_seconds: float = -1.0,
        energy: Optional[Union[tuple, list, np.ndarray]] = None,
        weight: Optional[Union[tuple, list, np.ndarray]] = None,
        log_ovlp_factor: Optional[Union[tuple, list, np.ndarray]] = None,
        input_file: Optional[str] = None,
        avg_1rdm: Optional[np.ndarray] = None,
        avg_1rdm_stoch_error: Optional[np.ndarray] = None,
        **kwargs
    ) -> "Results":
        """
        Construct Results from user-provided data.

        This is useful for reproducing results from literature or external sources.

        Parameters
        ----------
        return_code : int, optional
            Exit code (default: 0 for success)
        run_time_seconds : float, optional
            Runtime in seconds (default: -1.0)
        energy : array-like, optional
            Energy [E, dE] or just [E]
        weight : array-like, optional
            Walker weight [weight, dweight] or just [weight]
        log_ovlp_factor : array-like, optional
            Log overlap factor [log_ovlp, dlog_ovlp] or just [log_ovlp]
        input_file : str, optional
            Input file contents
        avg_1rdm : np.ndarray, optional
            Averaged 1-RDM
        avg_1rdm_stoch_error : np.ndarray, optional
            Stochastic error in 1-RDM
        **kwargs : dict
            Additional fields (e.g., afqmc_raised_warning, afqmc_raised_error)

        Returns
        -------
        Results
            Populated Results object

        Notes
        -----
        At minimum, you should provide either energy or weight data to create
        a meaningful Results object.
        """
        results = cls()
        results.return_code = return_code
        results.run_time_seconds = run_time_seconds

        # Process energy data
        if energy is not None:
            energy_array = np.atleast_1d(energy)
            if len(energy_array) == 1:
                energy_array = np.array([energy_array[0], 0.0])
            results.energy = energy_array

        # Process weight data
        if weight is not None:
            weight_array = np.atleast_1d(weight)
            if len(weight_array) == 1:
                weight_array = np.array([weight_array[0], 0.0])
            results.weight = weight_array

        # Process log_ovlp_factor data
        if log_ovlp_factor is not None:
            log_ovlp_array = np.atleast_1d(log_ovlp_factor)
            if len(log_ovlp_array) == 1:
                log_ovlp_array = np.array([log_ovlp_array[0], 0.0])
            results.log_ovlp_factor = log_ovlp_array

        results.input_file = input_file
        results.avg_1rdm = avg_1rdm
        results.avg_1rdm_stoch_error = avg_1rdm_stoch_error

        # Handle additional keyword arguments
        for key, value in kwargs.items():
            if hasattr(results, key):
                setattr(results, key, value)

        return results

    @classmethod
    def from_hdf5(cls, filename: Union[str, Path] = "results.h5") -> "Results":
        """
        Construct Results from an existing HDF5 file.

        Parameters
        ----------
        filename : str or Path, optional
            Path to the HDF5 file (default: "results.h5")

        Returns
        -------
        Results
            Populated Results object
        """
        results = cls()

        with h5.File(filename, 'r') as f:
            # Load basic metadata
            if "return_code" in f:
                results.return_code = int(f["return_code"][()])
            if "run_time_seconds" in f:
                results.run_time_seconds = float(f["run_time_seconds"][()])

            # Load flags
            if "afqmc_raised_warning" in f:
                results.afqmc_raised_warning = bool(f["afqmc_raised_warning"][()])
            if "afqmc_raised_error" in f:
                results.afqmc_raised_error = bool(f["afqmc_raised_error"][()])
            if "afqmc_is_finite" in f:
                results.afqmc_is_finite = bool(f["afqmc_is_finite"][()])

            # Load input file
            if "input_file" in f:
                results.input_file = f["input_file"][()].decode() if isinstance(f["input_file"][()], bytes) else str(f["input_file"][()])

            # Load warning messages
            if "warning_messages" in f:
                warning_group = f["warning_messages"]
                num_warnings = int(warning_group["num_messages"][()])
                results.warning_messages = set(
                    warning_group[f"warning_{i}"][()].decode() if isinstance(warning_group[f"warning_{i}"][()], bytes) else str(warning_group[f"warning_{i}"][()])
                    for i in range(num_warnings)
                )

            # Load error messages
            if "error_messages" in f:
                error_group = f["error_messages"]
                num_errors = int(error_group["num_messages"][()])
                results.error_messages = set(
                    error_group[f"error_{i}"][()].decode() if isinstance(error_group[f"error_{i}"][()], bytes) else str(error_group[f"error_{i}"][()])
                    for i in range(num_errors)
                )

            # Load calculated results
            if "energy" in f:
                results.energy = f["energy"][:]
            if "weight" in f:
                results.weight = f["weight"][:]
            if "LogOvlpFactor" in f:
                results.log_ovlp_factor = f["LogOvlpFactor"][:]
            if "avg_1rdm" in f:
                results.avg_1rdm = f["avg_1rdm"][:]
            if "avg_1rdm_stoch_error" in f:
                results.avg_1rdm_stoch_error = f["avg_1rdm_stoch_error"][:]

        return results

    def to_hdf5(self, filename: Union[str, Path] = "results.h5") -> None:
        """
        Write Results to an HDF5 file.

        Parameters
        ----------
        filename : str or Path, optional
            Path to the output HDF5 file (default: "results.h5")
        """
        # Explicitly remove existing file to ensure clean overwrite
        filepath = Path(filename)
        if filepath.exists():
            filepath.unlink()
        
        with h5.File(filename, 'w') as f:
            # Write basic metadata
            f.create_dataset("return_code", data=self.return_code if self.return_code is not None else -1)
            f.create_dataset("run_time_seconds", data=self.run_time_seconds)

            # Write flags
            f.create_dataset("afqmc_raised_warning", data=self.afqmc_raised_warning)
            f.create_dataset("afqmc_raised_error", data=self.afqmc_raised_error)
            f.create_dataset("afqmc_is_finite", data=self.afqmc_is_finite)

            # Write input file if available
            if self.input_file is not None:
                f.create_dataset("input_file", data=self.input_file)

            # Write warning messages (always create group, even if empty)
            warning_group = f.create_group("warning_messages")
            warning_group.create_dataset("num_messages", data=len(self.warning_messages))
            for i, warning in enumerate(self.warning_messages):
                warning_group.create_dataset(f"warning_{i}", data=str(warning))

            # Write error messages (always create group, even if empty)
            error_group = f.create_group("error_messages")
            error_group.create_dataset("num_messages", data=len(self.error_messages))
            for i, error in enumerate(self.error_messages):
                error_group.create_dataset(f"error_{i}", data=str(error))

            # Write calculated results if available
            if self.energy is not None:
                f.create_dataset("energy", data=self.energy)
            if self.weight is not None:
                f.create_dataset("weight", data=self.weight)
            if self.log_ovlp_factor is not None:
                f.create_dataset("LogOvlpFactor", data=self.log_ovlp_factor)
            if self.avg_1rdm is not None:
                f.create_dataset("avg_1rdm", data=self.avg_1rdm)
            if self.avg_1rdm_stoch_error is not None:
                f.create_dataset("avg_1rdm_stoch_error", data=self.avg_1rdm_stoch_error)

    def __repr__(self) -> str:
        """String representation of Results object."""
        parts = [f"Results(return_code={self.return_code}"]
        if self.energy is not None:
            parts.append(f"energy={self.energy[0]:.6f}±{self.energy[1]:.6f}")
        if self.weight is not None:
            parts.append(f"weight={self.weight[0]:.6f}±{self.weight[1]:.6f}")
        if self.afqmc_raised_warning:
            parts.append(f"warnings={len(self.warning_messages)}")
        if self.afqmc_raised_error:
            parts.append(f"errors={len(self.error_messages)}")
        return ", ".join(parts) + ")"
