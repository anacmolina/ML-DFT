# Note there is a space at the end of the next line.
#DISBATCH PREFIX cd /mnt/home/amolina/ceph/ML-DFT/experiments/modeling ;  echo "Running on $(hostname)" ; module purge ; module load gcc slurm ; source /mnt/home/amolina/miniconda3/bin/activate ml-dft; echo $PYTHON_PATH ; 
#DISBATCH SUFFIX > /mnt/ceph/users/amolina/database/models/mlp_tracking/outputs/${SLURM_JOB_ID}_${DISBATCH_TASKID}.log 2>&1

python3 slurm_train_mlp_model.py -np ${SLURM_NPROCS}  -ml 0 -lr 5e-5 -ni 10000 -hdm 16 -hdp 4 -id ${DISBATCH_JOBID}_${DISBATCH_TASKID}
python3 slurm_train_mlp_model.py -np ${SLURM_NPROCS} -ml 0 -lr 5e-5 -ni 10000 -hdm 32 -hdp 4 -id ${DISBATCH_JOBID}_${DISBATCH_TASKID}
python3 slurm_train_mlp_model.py -np ${SLURM_NPROCS} -ml 0 -lr 5e-5 -ni 10000 -hdm 64 -hdp 4 -id ${DISBATCH_JOBID}_${DISBATCH_TASKID}
python3 slurm_train_mlp_model.py -np ${SLURM_NPROCS} -ml 0 -lr 5e-5 -ni 10000 -hdm 128 -hdp 4 -id ${DISBATCH_JOBID}_${DISBATCH_TASKID}
python3 slurm_train_mlp_model.py -np ${SLURM_NPROCS} -ml 0 -lr 5e-5 -ni 10000 -hdm 16 -hdp 8 -id ${DISBATCH_JOBID}_${DISBATCH_TASKID}
python3 slurm_train_mlp_model.py -np ${SLURM_NPROCS} -ml 0 -lr 5e-5 -ni 10000 -hdm 32 -hdp 8 -id ${DISBATCH_JOBID}_${DISBATCH_TASKID}
python3 slurm_train_mlp_model.py -np ${SLURM_NPROCS} -ml 0 -lr 5e-5 -ni 10000 -hdm 64 -hdp 8 -id ${DISBATCH_JOBID}_${DISBATCH_TASKID}
python3 slurm_train_mlp_model.py -np ${SLURM_NPROCS} -ml 0 -lr 5e-5 -ni 10000 -hdm 128 -hdp 8 -id ${DISBATCH_JOBID}_${DISBATCH_TASKID}
