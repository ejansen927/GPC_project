#!/bin/bash
############################################
### Submit job by: sbatch slurm-job.sh
############################################

#SBATCH --job-name="init points"
#SBATCH --mail-user=ejansen@bnl.gov
#SBATCH --mail-type=begin,end  # send specified email notification

#SBATCH --partition=thal5	#two types: thal5->  memory 187GB; and thal5fat- fat has bigger memory 376GB: 2 nodes only
#SBATCH --nodes=1      # use single node only
#SBATCH --ntasks=100 ###8      # use 8 cores total (including hyperthreading)
##SBATCH --mem=20G       # memory required (max.)
#SBATCH --time=3-00:00  # 3 days total wall runtime (max.)

##(regular shell script section)


echo "At `date`, we are on `hostname`"
export SOURCE="$SLURM_SUBMIT_DIR"	#this is the directory where we are running jobs from; similar to pbs_o_workdir?
echo "SLURM_WORKDIR" $SLURM_SUBMIT_DIR

cd $SLURM_SUBMIT_DIR


echo $SLURM_JOB_NAME
echo $SLURM_CLUSTER_NAME
echo $SLURM_JOB_PARTITION
echo $SLURM_JOB_ID
echo $SLURM_NTASKS
echo $SLURM_JOB_NODELIST
#echo $SLURM_SUBMIT_HOST	#this is the machine hostname of the machines i.e. thalia.phy.bnl.gov
#echo $SLURM_ARRAY_TASK_ID


#===========================================================
#export WORKDIR="/project/$LOGNAME/$SLURM_JOB_ID"
#export SCRATCH=$WORKDIR


NP=$SLURM_NTASKS
echo "Number of processors running $NP"

echo "Job beginning at `date`."

lattice=12
dim=5
n=100
out="init_points_$n"

#python parallel_initialize.py --n $n --dim $dim --lattice $lattice --out $out
python initialize.py --n $n --dim $dim --lattice $lattice --out $out

echo "All jobs completed at `date`."
