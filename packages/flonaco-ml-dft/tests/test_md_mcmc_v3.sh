#!/bin/bash

#SBATCH -o ./md_mcmc_mix_%j.out
#SBATCH -e ./md_mcmc_mix_%j.err

#SBATCH -D ./

#SBATCH -J md_mcmc_mix

#SBATCH --constraint=skylake
#SBATCH --partition=ccm

#SBATCH --ntasks-per-node=16
#SBATCH --nodes=1
#SBATCH --cpus-per-task=1

#SBATCH --time=7-0:00

ml load modules/1.59-20220201
ml load slurm
module load gcc/7.5.0
ml load openmpi

source /mnt/home/amolina/ceph/envs/dft-0.1/bin/activate

time mpiexec -np $SLURM_NTASKS --bind-to core gpaw python test_md_mcmc_v3.py

deactivate
