#!/bin/bash

# filenames
fcidump_file=FCIDUMP
dice_output=output.dat
afqmc_file=afqmc.h5

# trial wavefunction settings
ndet=100


# use verbose mode to see Cholesky information
fcidump_to_afqmc -i $fcidump_file -o $afqmc_file -v &> make_hamil.out
dice_to_hdf5 -i $dice_output -o $afqmc_file  -n $ndet -v &> make_wfn.out

write_afqmc_json -i $afqmc_file -b 400

