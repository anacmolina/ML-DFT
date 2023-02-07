module load slurm
module add disBatch

sbatch --nodes=1 --ntasks-per-node=2 --cpus-per-task=8 -p ccm disBatch disbatch_train_flow