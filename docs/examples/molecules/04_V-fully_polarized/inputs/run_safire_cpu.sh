#!/bin/bash -l
#SBATCH -J V_atom
#SBATCH --partition=ccq
#SBATCH --constraint=icelake
#! Number of MPI ranks (= tasks for Slurm)
#SBATCH --ntasks=64
#SBATCH --time=1:00:00

#export AFQMC_PATH=/mnt/home/beskridge/ceph/software/AuxiliaryFields/build/CPU_mods2.3
#source $AFQMC_PATH/env.sh

module load safire

date
# Launch MPI code...
srun --cpu-bind=cores safire --filenames afqmc.json &> afqmc.out
date
