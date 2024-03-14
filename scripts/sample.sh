#!/bin/sh

source ../../venv/cv/bin/activate
source set_pythonpath.sh

# model_params="/Users/jongbeomkim/Documents/ddpm/kr-ml-test/ddpm_celeba_64×64.pth"
model_params="/home/dmeta0304/Downloads/ddpm_celeba_64×64.pth"
img_size=64
scale_factor=16
dataset="metfaces"
data_dir="/home/dmeta0304/Documents/datasets/metfaces/"
# dataset="celeba"
# data_dir="/home/dmeta0304/Documents/datasets/"
ref_idx=2
batch_size=36

python3 ../sample.py\
    --model_params="$model_params"\
    --img_size=$img_size\
    --scale_factor=$scale_factor\
    --dataset="$dataset"\
    --data_dir="$data_dir"\
    --ref_idx=$ref_idx\
    --batch_size=$batch_size\
    --mode="single_ref"\
