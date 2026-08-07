#!/bin/bash -l
#SBATCH -J V_atom
#SBATCH --partition=ccq,gen
#SBATCH --constraint=icelake
#! Number of MPI ranks (= tasks for Slurm)
#SBATCH --ntasks=64
#SBATCH --time=4:00:00

module load safire-run

date
# Launch MPI code...
srun --cpu-bind=cores safire afqmc.json &> afqmc.out
date
