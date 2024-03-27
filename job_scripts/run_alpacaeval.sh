\#!/bin/sh
# Grid Engine options (lines prefixed with #$)
#$ -N alpaca
#$ -cwd
#$ -l h_rt=24:00:00
#$ -l h_vmem=100G
#$ -o alpaca/train.log
#$ -e alpaca/train.err
#$ -q gpu
#$ -pe gpu-a100 1
#  These options are:
#  job name: -N
#  use the current working directory: -cwd
#  runtime limit of 5 minutes: -l h_rt
#  memory limit of 1 Gbyte: -l h_vmem

> $(pwd)/logs_alpaca/train.err
> $(pwd)/logs_alpaca/train.log

# Initialise the environment modules
. /etc/profile.d/modules.sh
export XDG_CACHE_HOME="/exports/eddie/scratch/s1808795/.cache"

# Load Python
module load cuda
module load python/3.11.4

source /exports/eddie/scratch/s1808795/repo/PEFT-TRL/venv/bin/activate

alpaca_eval evaluate_from_model \
  --model_configs 'oasst-sft-pythia-12b' \
  --annotators_config 'alpaca_eval_gpt4_turbo_fn'


deactivate





