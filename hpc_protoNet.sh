#!/bin/sh
#SBATCH --verbose
#SBATCH -p aquila
#SBATCH --time=100:00:00
#SBATCH --nodes=1
#SBATCH --mem=12GB
#SBATCH --mail-type=ALL # select which email types will be sent
#SBATCH --mail-user=yy2564@nyu.edu # put your email here if you want emails

#SBATCH --array=0
# we have 8 jobs indexed from 0 to 7, which will show up in SLURM_ARRAY_TASK_ID(see last line)
# the first cpu is gonna run hw1_1.py --setting 0 etc

#SBATCH --output=run_%A_%a.out # %A is SLURM_ARRAY_JOB_ID, %a is SLURM_ARRAY_TASK_ID
#SBATCH --error=run_%A_%a.err

#SBATCH --gres=gpu:1 # uncomment this line to request for a gpu if your program uses gpu
# #SBATCH --constraint=cpu # use this if you want to only use cpu

# the sleep command will help with hpc issues when you have many jobs loading same files
sleep $(( (RANDOM%10) + 1 ))

echo "SLURM_JOBID: " $SLURM_JOBID
echo "SLURM_ARRAY_JOB_ID: " $SLURM_ARRAY_JOB_ID
echo "SLURM_ARRAY_TASK_ID: " $SLURM_ARRAY_TASK_ID

module load anaconda3 cuda/9.0 glfw/3.3 gcc/7.3 mesa/19.0.5 llvm/7.0.1

echo ${SLURM_ARRAY_TASK_ID}
python -m protoNet.experiments.proto_nets --dataset miniImageNet --distance cosine --k-test 5 --n-test 1 --k-train 20 --n-train 1 --q-train 15 --setting ${SLURM_ARRAY_TASK_ID}
