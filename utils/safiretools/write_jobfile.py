# This file is distributed under the Apache License, Version 2.0 License.
# See LICENSE file in top directory for details.
#
# Copyright (c) 2021-2025 The Simons Foundation, Inc.
#
# You may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0

"""Writing the JSON input file the AFQMC executable is run on, together with the
HDF5 file holding the Hamiltonian and wavefunction it refers to."""

import json
from pathlib import Path

from safiretools.hamiltonian.base import Hamiltonian
from safiretools.wavefunction.base import Wavefunction


def write_jobfile(filename, parameters) -> None:
    """
    Write the SAFIRE input file `filename` from `parameters`.

    Parameters
    ----------
    filename : str or pathlib.Path
        JSON file to write. Its directory must exist.
    parameters : dict
        The input deck, in the format of the SAFIRE afqmc.json input file, except
        that a `Hamiltonian` or `Wavefunction` object may stand wherever that
        format expects the name of the file holding it. Every such object is
        written to ``<filename without suffix>_inputs.h5`` next to the input file
        and replaced by that file's name, which is the path to it relative to the
        input file.

    Examples
    --------
    >>> write_jobfile(run_dir / 'afqmc.json', {
    ...     'execute': {
    ...         'wavefunction': {'filename': wfn},
    ...         'hamiltonian': {'filename': ham},
    ...         'timestep': 0.01,
    ...         'steps': 10000,
    ...         'n_walkers_per_mpi_task': 10,
    ...     },
    ... })
    """
    filename = Path(filename)
    inputs = filename.with_name(f'{filename.stem}_inputs.h5')
    with open(filename, 'w') as f:
        json.dump(_resolve_inputs(parameters, inputs), f, indent=2)


def _resolve_inputs(node, inputs: Path):
    """`node` with every Hamiltonian and wavefunction in it written to the HDF5
    file `inputs` and replaced by its name. The original is left untouched."""
    if isinstance(node, (Hamiltonian, Wavefunction)):
        node.to_hdf5(inputs)
        return inputs.name
    if isinstance(node, dict):
        return {key: _resolve_inputs(value, inputs) for key, value in node.items()}
    if isinstance(node, list):
        return [_resolve_inputs(item, inputs) for item in node]
    return node
