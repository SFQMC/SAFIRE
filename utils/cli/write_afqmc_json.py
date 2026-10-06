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

from argparse import ArgumentParser

from afqmctools.inputs.from_hdf import write_json

# TODO: This isn't a very convenient tool. Redesign with:
#          1. most of the functionality living within the afqmctools package (just expose functionality here)
#          2. Add some *validation* - mostly generate warnings
#          3. Add visualization
#          4. Maybe use some *recipes* to guide users. For example: a GPU run will prioritize running many walkers per task/gpu (~ 10,000 ),
#                   but an mpi run using CPUs will prioritize many MPI tasks (10s of walkers per task)
#                - recipes might also distinguish between application domain ()

def get_args(description=None):
    parser = ArgumentParser(description=description)
    
    # file settings
    parser.add_argument('--fout', '-o', default='afqmc.json')
    parser.add_argument('--fwfn', '-i', default='afqmc.h5')
    parser.add_argument('--fham', '-ih')
    parser.add_argument('--verbose', '-v', action='store_true')

    # basic projection settings
    parser.add_argument('--steps', '-s', type=int, default=10000)
    parser.add_argument('--timestep', '-ts', type=float, default=0.01)
    parser.add_argument('--n_walkers', '-nw', type=int, default=200,
        help='total number of walkers, split evenly over the MPI ranks')
    
    # estimator settings
    parser.add_argument('--mixed_est', '-me', action='store_true')
    parser.add_argument('--time_bp', '-tbp', type=float,
        help='maximum back propagation time')
    parser.add_argument('--num_bp', '-nbp', type=int, default=1)
    parser.add_argument('--path_restoration', '-pr', default='0',
        help='path restoration type "0", "1", or "e" for no, yes, and extra')
    
    args = parser.parse_args()
    return args

def main():
    description = """
    Writes an AFQMC json input file based on the settings provided.
    """

    args = get_args(description)
    
    fham = args.fham
    if fham is None:
        fham = args.fwfn

    exec_opts = dict(
        steps = args.steps,
        timestep = args.timestep,
        num_walkers = args.n_walkers
    )
    write_json(args.fout, args.fwfn, fham, exec_opts=exec_opts, args_namespace=args)


if __name__ == '__main__':
    main()
