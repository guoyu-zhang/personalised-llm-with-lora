\#!/bin/sh
# Grid Engine options (lines prefixed with #$)
#$ -N get_accuracy
#$ -cwd
#$ -l h_rt=24:00:00
#$ -l h_vmem=80G
#$ -o logs/acc_user1/train.log
#$ -e logs/acc_user1/train.err
#$ -q gpu
#$ -pe gpu-a100 1
#  These options are:
#  job name: -N
#  use the current working directory: -cwd
#  runtime limit of 5 minutes: -l h_rt
#  memory limit of 1 Gbyte: -l h_vmem

> $(pwd)/logs/acc_user1/train.err
> $(pwd)/logs/acc_user1/train.log

# Initialise the environment modules
. /etc/profile.d/modules.sh
export XDG_CACHE_HOME="/exports/eddie/scratch/s1808795/.cache"

# Load Python
module load cuda
module load python/3.11.4

source /exports/eddie/scratch/s1808795/repo/PEFT-TRL/venv/bin/activate

# Run the program
python ~/user_centric_llms/python_files/accuracy_user.py

deactivate




