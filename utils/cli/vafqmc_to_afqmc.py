#! /usr/bin/env python3

# This file is distributed under the Apache License, Version 2.0 License.
# See LICENSE file in top directory for details.
#
# Copyright (c) 2021-2025 The Simons Foundation, Inc.
#
# You may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0

"""Convert VAFQMC (hafqmc) pickle checkpoints into SAFIRE HDF5 input files."""

import argparse
import sys
from pathlib import Path

from afqmctools.inputs.from_vafqmc import vafqmc_to_afqmc


def parse_args(args=None):
    """Parse command-line arguments.

    Parameters
    ----------
    args : list of strings
        command-line arguments.

    Returns
    -------
    options : :class:`argparse.Namespace`
        Command line arguments.
    """

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--hamiltonian', dest='hamiltonian', type=Path,
                        required=True,
                        help='Input hamiltonian.pkl from VAFQMC training.')
    parser.add_argument('--checkpoint', dest='checkpoint', type=Path,
                        required=True,
                        help='Input checkpoint.pkl (or ckpt_*.pkl) from VAFQMC training.')
    parser.add_argument('-o', '--out-dir', dest='out_dir', type=Path,
                        required=True,
                        help='Output directory for ham.h5, wfn.h5 and, with '
                             '--variational, ham_var.h5.')
    parser.add_argument('--variational', dest='variational',
                        action='store_true', default=False,
                        help='Also write ham_var.h5, the inner (variational) '
                             'Hamiltonian rebuilt from the trained propagator '
                             'parameters, with its timestep stamped on it.')
    parser.add_argument('--walker-type', dest='walker_type',
                        default='auto', choices=['auto', 'closed', 'collinear', 'uhf'],
                        help='SAFIRE walker / NOMSD layout. "auto" (default) compares '
                             'the alpha and beta orbitals, which is the only way to '
                             'see a broken-symmetry singlet (na == nb but COLLINEAR).')
    parser.add_argument('--no-orthonormalize', dest='no_orthonormalize',
                        action='store_true', default=False,
                        help='Skip Gram-Schmidt orthonormalization of the orbitals.')
    parser.add_argument('--init-dt', dest='init_dt', type=float, default=None,
                        help='Fallback timestep for checkpoints trained with FROZEN '
                             'timesteps (parametrize excludes "tsteps"), where ts_v '
                             'never enters the params tree and the trained dt is just '
                             'the init value. Use the training hparams.yml '
                             'ansatz.propagators[0].init_tsteps[0]. Such a checkpoint '
                             'cannot be converted without it.')
    parser.add_argument('--spin-layout', dest='spin_layout',
                        default='auto', choices=['auto', 'restricted', 'polarized'],
                        help='Basis the output is written on. "restricted" = nmo '
                             'spatial orbitals; "polarized" = 2*nmo spin orbitals as a '
                             'COLLINEAR/ndown=0 problem, the only layout that can carry '
                             'a spin_mixing:true trial. "auto" (default) picks polarized '
                             'iff the trained operators are spin-doubled. A spin-mixed '
                             'checkpoint with --spin-layout restricted is a hard error, '
                             'not a truncation.')
    parser.add_argument('-v', '--verbose', dest='verbose',
                        action='store_true', default=False,
                        help='Print the files written.')

    options = parser.parse_args(args)

    if not options.hamiltonian.is_file():
        parser.error(f'Hamiltonian file not found: {options.hamiltonian}')
    if not options.checkpoint.is_file():
        parser.error(f'Checkpoint file not found: {options.checkpoint}')

    return options


def main(args=None):
    options = parse_args(args)

    outputs = vafqmc_to_afqmc(
        options.hamiltonian,
        options.checkpoint,
        options.out_dir,
        variational=options.variational,
        walker_type=options.walker_type,
        orthonormalize=not options.no_orthonormalize,
        init_dt=options.init_dt,
        spin_layout=options.spin_layout,
    )

    if options.verbose:
        print('Wrote SAFIRE input:')
        for name, path in outputs.items():
            print(f'  {name}: {path}')
        if not options.variational:
            print('  (ham_var.h5 not written; pass --variational to include it)')

    return 0


if __name__ == '__main__':
    sys.exit(main())
