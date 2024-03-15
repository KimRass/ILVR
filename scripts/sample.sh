#!/bin/sh

source ../../venv/cv/bin/activate
source set_pythonpath.sh

# model_params="/Users/jongbeomkim/Documents/ddpm/kr-ml-test/ddpm_celeba_64×64.pth"
model_params="/home/dmeta0304/Downloads/ddpm_celeba_64×64.pth"
img_size=64
dataset="celeba"
data_dir="/home/dmeta0304/Documents/datasets"
# dataset="metfaces"
# data_dir="/home/dmeta0304/Documents/datasets/metfaces"
ref_idx=0

# scale_factor=8
# batch_size=4
# last_cond_step_idx=250
# python3 ../sample.py\
#     --model_params="$model_params"\
#     --img_size=$img_size\
#     --scale_factor=$scale_factor\
#     --dataset="$dataset"\
#     --data_dir="$data_dir"\
#     --ref_idx=$ref_idx\
#     --batch_size=$batch_size\
#     --last_cond_step_idx=$last_cond_step_idx\
#     --mode="single_ref"\

# last_cond_step_idx=0
# python3 ../sample.py\
#     --model_params="$model_params"\
#     --img_size=$img_size\
#     --dataset="$dataset"\
#     --data_dir="$data_dir"\
#     --ref_idx=$ref_idx\
#     --mode="various_scale_factors"\

scale_factor=32
python3 ../sample.py\
    --model_params="$model_params"\
    --img_size=$img_size\
    --scale_factor=$scale_factor\
    --dataset="$dataset"\
    --data_dir="$data_dir"\
    --ref_idx=$ref_idx\
    --mode="various_cond_range"\
