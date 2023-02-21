for i in {4..128..4}
do
    echo 'python3 slurm_train_flow_model.py -np ${SLURM_NPROCS} -ml 0 -lr 1e-4 -nb 2 -ni 10000 -hdm '${i}' -hdp 2 -id ${SLURM_JOBID}_${DISBATCH_TASKID}' 
done