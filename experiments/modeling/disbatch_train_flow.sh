# Note there is a space at the end of the next line.
#DISBATCH PREFIX cd /mnt/home/mgabrie/ml-dft/ML-DFT/experiments/modeling ;  echo "Running on $(hostname)" ; module purge ; module load gcc slurm ; module load python3/3.7.3 ; module load cuda cudnn ; which python3 ; source /mnt/home/mgabrie/ml-dft/env-ml-dft/bin/activate ; echo $PYTHON_PATH ; 
#DISBATCH SUFFIX > /mnt/ceph/users/mgabrie/ml-dft/${DISBATCH_NAMETASKS}_${DISBATCH_JOBID}_${DISBATCH_TASKID}.log 2>&1

python3 slurm_train_flow_model.py  -lr 1e-3 -niter 1000 -hdm 32  -id ${DISBATCH_JOBID}_${DISBATCH_TASKID}
python3 slurm_train_flow_model.py  -lr 1e-3 -niter 1000 -hdm 64  -id ${DISBATCH_JOBID}_${DISBATCH_TASKID}
python3 slurm_train_flow_model.py  -lr 1e-3 -niter 1000 -hdm 128  -id ${DISBATCH_JOBID}_${DISBATCH_TASKID}
python3 slurm_train_flow_model.py -lr 1e-2 -niter 1000 -hdm 64  -id ${DISBATCH_JOBID}_${DISBATCH_TASKID}
python3 slurm_train_flow_model.py -lr 1e-4 -niter 1000 -hdm 64  -id ${DISBATCH_JOBID}_${DISBATCH_TASKID}
