\#!/bin/sh
# Grid Engine options (lines prefixed with #$)
#$ -N logs_hh_shp_dpoall
#$ -cwd
#$ -l h_rt=24:00:00
#$ -l h_vmem=80G
#$ -o logs_hh_shp_dpoall/train.log
#$ -e logs_hh_shp_dpoall/train.err
#$ -q gpu
#$ -pe gpu-a100 1
#  These options are:
#  job name: -N
#  use the current working directory: -cwd
#  runtime limit of 5 minutes: -l h_rt
#  memory limit of 1 Gbyte: -l h_vmem

> $(pwd)/logs_hh_shp_dpoall/train.err
> $(pwd)/logs_hh_shp_dpoall/train.log

# Initialise the environment modules
. /etc/profile.d/modules.sh
export XDG_CACHE_HOME="/exports/eddie/scratch/s1808795/.cache"

# Load Python
module load cuda
module load python/3.11.4

source /exports/eddie/scratch/s1808795/repo/PEFT-TRL/venv/bin/activate

# Run the program
python ~/user_centric_llms/python_files/dpo_shp.py --beta 0.5 --output_dir "./model_hh_shp2_dpo5" --huggingface_dir_name "hh_shp2_dpo5" --dataset_name_or_path "guoyu-zhang/shp_2"
python ~/user_centric_llms/python_files/dpo_shp.py --beta 0.5 --output_dir "./model_hh_shp3_dpo5" --huggingface_dir_name "hh_shp3_dpo5" --dataset_name_or_path "guoyu-zhang/shp_3"
python ~/user_centric_llms/python_files/dpo_shp.py --beta 0.5 --output_dir "./model_hh_shp4_dpo5" --huggingface_dir_name "hh_shp4_dpo5" --dataset_name_or_path "guoyu-zhang/shp_4"
python ~/user_centric_llms/python_files/dpo_shp.py --beta 0.9 --output_dir "./model_hh_shp2_dpo9" --huggingface_dir_name "hh_shp2_dpo9" --dataset_name_or_path "guoyu-zhang/shp_2"
python ~/user_centric_llms/python_files/dpo_shp.py --beta 0.9 --output_dir "./model_hh_shp3_dpo9" --huggingface_dir_name "hh_shp3_dpo9" --dataset_name_or_path "guoyu-zhang/shp_3"
python ~/user_centric_llms/python_files/dpo_shp.py --beta 0.9 --output_dir "./model_hh_shp4_dpo9" --huggingface_dir_name "hh_shp4_dpo9" --dataset_name_or_path "guoyu-zhang/shp_4"
python ~/user_centric_llms/python_files/dpo_shp.py --beta 0.1 --output_dir "./model_hh_shp2_dpo1" --huggingface_dir_name "hh_shp2_dpo1" --dataset_name_or_path "guoyu-zhang/shp_2"
python ~/user_centric_llms/python_files/dpo_shp.py --beta 0.1 --output_dir "./model_hh_shp3_dpo1" --huggingface_dir_name "hh_shp3_dpo1" --dataset_name_or_path "guoyu-zhang/shp_3"
python ~/user_centric_llms/python_files/dpo_shp.py --beta 0.1 --output_dir "./model_hh_shp4_dpo1" --huggingface_dir_name "hh_shp4_dpo1" --dataset_name_or_path "guoyu-zhang/shp_4"
python ~/user_centric_llms/python_files/dpo_shp.py --beta 0.7 --output_dir "./model_hh_shp1_dpo7" --huggingface_dir_name "hh_shp1_dpo7" --dataset_name_or_path "guoyu-zhang/shp_1"

deactivate




