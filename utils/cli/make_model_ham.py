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
command line script for generating model Hamiltonians
"""
import argparse

import numpy as np
import matplotlib.pyplot as plt

from afqmctools.hamiltonian.model.builder import HamiltonianBuilder
from afqmctools.wavefunction.model import write_free_electron_wfn
import afqmctools.utils.io as io
import afqmctools.utils.visualize as vis

def make_hk_model_infile(
    fname=None,
    outfile="afqmc.h5",
    plot_lattice=False,
    plot_one_body=False,
    plot_hubbard_U=False,
    save_plots=False,
    write_H=True,
    write_fe_wfn=False,
    nelec=None,
    make_pair_correlators=False,
    U=None,
    spin_symm=None
    ):

    builder = HamiltonianBuilder.from_input(source=fname)

    hamiltonian = builder.hamiltonian

    lattice = builder.lattice

    params = io.read_input_params(fname).get('misc_params',{})

    if nelec is None:
        nelec = params.get('nelec',(0,0))

    for key,param in params.items():
        print(f"{key} -> {param} ({type(param)})")

    if plot_lattice:
        vis.plot_lattice(
            lattice=lattice,
            save=save_plots
            )

    if plot_one_body:
        
        one_body_mat = hamiltonian['tij'][0].csr_array.toarray()
        if np.iscomplexobj(one_body_mat):
            plt.matshow(one_body_mat.real)
            plt.title("hopping matrix - real")
            plt.colorbar()
            if save_plots:
                plt.savefig("hopping_matrix_real.png")
            else:
                plt.show()
            plt.matshow(one_body_mat.imag)
            plt.title("hopping matrix - imag")
            plt.colorbar()
            if save_plots:
                plt.savefig("hopping_matrix_imag.png")
            else:
                plt.show()

        else:
            plt.matshow(one_body_mat)
            plt.title("hopping matrix")
            plt.colorbar()
            if save_plots:
                plt.savefig("hopping_matrix.png")
            else:
                plt.show()

    if plot_hubbard_U:
        plt.matshow(
            hamiltonian['Uij'][0].csr_array.toarray()
        )
        plt.title("Hubbard-Kanamori 2-body interaction matrix")
        plt.colorbar()
        if save_plots:
            plt.savefig("2bodyHK__matrix.png")
        else:
            plt.show()


    if write_H:
        io.write_model_hamiltonian(
            hamiltonian=hamiltonian,
            fname=outfile,
            nelec=nelec,
            spin_symm=spin_symm
        )

    if write_fe_wfn:
        write_free_electron_wfn(
            hamiltonian_fname=outfile,
            nelec=nelec,
            spin_symm=spin_symm
        )

    if make_pair_correlators:
        print(
            "Warning: Super Conducting Pair Correlators currently assume "
            "a rectangular lattice and fully PBCs in both directions."
            )
        io.write_pair_correlators(
            fname=outfile,
            pairs_dict=lattice.get_directed_pairs(
                directions=["s","+x","-x","+y","-y"]
            )
        )
    

def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        '--input','-i',
        metavar='input',
        help="Input file containing Hamiltonan params (toml)",
        type=str,
        default="input.toml"
        )
    parser.add_argument(
        '--output','-o',
        metavar='outfile',
        help="Name of HDF5 file to save model Hamiltonian in",
        type=str,
        default="afqmc.h5"
        )
    parser.add_argument(
        '--free-elec',
        help="Save free-electron wavefunction as well",
        action='store_true',
        default=False
    )
    parser.add_argument(
        '--slater-type',
        help="type of free-electron Slater determinant (closed,collinear,noncollinear)",
        choices=["closed","collinear","noncollinear"],
        default="collinear"
    )
    parser.add_argument(
        '--plotlat','-pl',
        help="plot the lattice",
        action="store_true",
        default=False
    )
    parser.add_argument(
        '--plot1b',
        help="plot the 1-body Hamiltonian",
        action="store_true",
        default=False
    )
    parser.add_argument(
        '--plot2b',
        help="plot the 2-body Hamiltonian",
        action="store_true",
        default=False
    )
    parser.add_argument(
        '--saveplots','-sp',
        help="save plots instead of displaying (useful for remote connections)",
        action="store_true",
        default=False
    )
    parser.add_argument(
        '--pair-corr','-pc',
        help="Add pair-correlators to AFQMC input file (s,+x,-x,+y,-y)",
        action="store_true",
        default=False
    )
    parser.add_argument(
        '--hubbardU','-U',
        help="Hubbard U, if given and '--free-elec' is set, the hubbard U energy "
             "is evaluated using the free-electron wavefunction (Note: this is only "
             "for reference - does not effect the output wavefunction!) ",
        type=float,
        default=0.0
    )
    
    return parser.parse_args()

def main():
    """
    Builds a lattice model Hamiltonian in SAFIRE format based on the settings in a .toml
    input file and saves to HDF5.
    """
    args = parse_args()

    make_hk_model_infile(
        fname=args.input,
        outfile=args.output,
        write_fe_wfn=args.free_elec,
        plot_lattice=args.plotlat,
        plot_one_body=args.plot1b,
        plot_hubbard_U=args.plot2b,
        save_plots=args.saveplots,
        make_pair_correlators=args.pair_corr, # TODO: read from input file!!
        U=args.hubbardU,
        spin_symm=args.slater_type
    )
if __name__ == '__main__':
      main()
