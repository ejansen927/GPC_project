#!/bin/bash
############################################
### Submit job by: sbatch slurm-job.sh
############################################

#SBATCH --job-name="gpc sweep"
#SBATCH --mail-user=ejansen@bnl.gov
#SBATCH --mail-type=BEGIN,END
#SBATCH --array=0-7%4
#SBATCH --partition=thal5
#SBATCH --nodes=1
#SBATCH --ntasks=100
#SBATCH --time=3-00:00

echo "At `date`, we are on `hostname`"
export SOURCE="$SLURM_SUBMIT_DIR"
echo "SLURM_WORKDIR $SLURM_SUBMIT_DIR"

cd $SLURM_SUBMIT_DIR

echo "Job name:       $SLURM_JOB_NAME"
echo "Cluster name:   $SLURM_CLUSTER_NAME"
echo "Partition:      $SLURM_JOB_PARTITION"
echo "Job ID:         $SLURM_JOB_ID"
echo "Num tasks:      $SLURM_NTASKS"
echo "Node list:      $SLURM_JOB_NODELIST"

echo "Number of processors running: $SLURM_NTASKS"
echo "Job beginning at `date`."

mkdir -p logs

if [[ ! -f joblist.txt ]]; then
    echo "joblist.txt not found!"
    exit 1
fi

CMD=$(sed -n "$((SLURM_ARRAY_TASK_ID + 1))p" joblist.txt)

echo "Running: $CMD"
eval $CMD

