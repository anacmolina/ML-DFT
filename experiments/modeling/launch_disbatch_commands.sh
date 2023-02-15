module load slurm
module add disBatch

sbatch --nodes=1 --ntasks-per-node=1 --cpus-per-task=8 --constraint=rome -p ccm disBatch disbatch_train_flow.sh

sbatch --nodes=1 --ntasks-per-node=8 --cpus-per-task=16 --constraint=rome -p ccm disBatch disbatch_train_mlp.sh