# Note there is a space at the end of the next line.
#DISBATCH PREFIX cd /mnt/home/amolina/ceph/ML-DFT/experiments/modeling ;  echo "Running on $(hostname)" ; module purge ; module load gcc slurm ; module load python3 ; which python3 ; source /mnt/home/amolina/miniconda3/bin/activate ml-dft; echo $PYTHON_PATH ; 
#DISBATCH SUFFIX > /mnt/ceph/users/amolina/database/models/tracking/outputs/${SLURM_JOB_ID}_${DISBATCH_TASKID}.log 2>&1

python slurm_train_flow_model.py  -lr 1e-3 -ni 1000 -hdm 32  -id ${DISBATCH_JOBID}_${DISBATCH_TASKID}
python slurm_train_flow_model.py  -lr 1e-3 -ni 1000 -hdm 64  -id ${DISBATCH_JOBID}_${DISBATCH_TASKID}
python slurm_train_flow_model.py  -lr 1e-3 -ni 1000 -hdm 128  -id ${DISBATCH_JOBID}_${DISBATCH_TASKID}
python slurm_train_flow_model.py -lr 1e-2 -ni 1000 -hdm 64  -id ${DISBATCH_JOBID}_${DISBATCH_TASKID}
python slurm_train_flow_model.py -lr 1e-4 -ni 1000 -hdm 64  -id ${DISBATCH_JOBID}_${DISBATCH_TASKID}
