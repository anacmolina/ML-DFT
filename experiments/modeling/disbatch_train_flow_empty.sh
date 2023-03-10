# Note there is a space at the end of the next line.
#DISBATCH PREFIX cd /mnt/home/amolina/ceph/ML-DFT/experiments/modeling ;  echo "Running on $(hostname)" ; module purge ; module load gcc slurm ; source /mnt/home/amolina/miniconda3/bin/activate ml-dft ; echo $PYTHON_PATH ; 
#DISBATCH SUFFIX > /mnt/ceph/users/amolina/database/models/flow_tracking/outputs/${SLURM_JOB_ID}_${DISBATCH_TASKID}.log 2>&1

