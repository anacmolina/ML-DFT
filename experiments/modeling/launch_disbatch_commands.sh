module load slurm
module add disBatch

sbatch --nodes=1 --ntasks-per-node=32 --cpus-per-task=2 --constraint=rome -p ccm disBatch disbatch_train_flow.sh

sbatch --nodes=1 --ntasks-per-node=16 --cpus-per-task=16 --constraint=rome -p ccm disBatch disbatch_train_mlp.sh