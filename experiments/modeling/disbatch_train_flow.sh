# Note there is a space at the end of the next line.
#DISBATCH PREFIX cd /mnt/home/amolina/ceph/ML-DFT/experiments/modeling ;  echo "Running on $(hostname)" ; module purge ; module load gcc slurm ; source /mnt/home/amolina/miniconda3/bin/activate ml-dft ; echo $PYTHON_PATH ; 
#DISBATCH SUFFIX > /mnt/ceph/users/amolina/database/models/flow_tracking/outputs/${SLURM_JOB_ID}_${DISBATCH_TASKID}.log 2>&1

python3 slurm_train_flow_model.py -np ${SLURM_NPROCS} -ml 0 -lr 5e-5 -nb 2 -ni 10000 -hdm 4 -hdp 2 -id ${SLURM_JOBID}_${DISBATCH_TASKID}
python3 slurm_train_flow_model.py -np ${SLURM_NPROCS} -ml 0 -lr 5e-5 -nb 2 -ni 10000 -hdm 8 -hdp 2 -id ${SLURM_JOBID}_${DISBATCH_TASKID}
python3 slurm_train_flow_model.py -np ${SLURM_NPROCS} -ml 0 -lr 5e-5 -nb 2 -ni 10000 -hdm 12 -hdp 2 -id ${SLURM_JOBID}_${DISBATCH_TASKID}
python3 slurm_train_flow_model.py -np ${SLURM_NPROCS} -ml 0 -lr 5e-5 -nb 2 -ni 10000 -hdm 16 -hdp 2 -id ${SLURM_JOBID}_${DISBATCH_TASKID}
