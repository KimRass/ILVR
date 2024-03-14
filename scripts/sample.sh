#!/bin/sh

source ../../venv/cv/bin/activate
source set_pythonpath.sh

# model_params="/Users/jongbeomkim/Documents/ddpm/kr-ml-test/ddpm_celeba_64×64.pth"
model_params="/home/dmeta0304/Downloads/ddpm_celeba_64×64.pth"
# save_dir="/Users/jongbeomkim/Desktop/workspace/ILVR/samples/"
save_dir="/home/dmeta0304/Desktop/workspace/ILVR/samples"
img_size=64

# mode="single_ref"
# ref_idx=100
# scale_factor=16
# python3 ../sample.py\
#     --mode=$mode\
#     --model_params="$model_params"\
#     --img_size=$img_size\
#     --data_dir="/home/dmeta0304/Documents/datasets/"\
#     --ref_idx=$ref_idx\
#     --scale_factor=$scale_factor\
#     --batch_size=35\
#     --save_path="$save_dir/$mode/ref_idx=$ref_idx-scale_factor=$scale_factor-0.jpg"\

mode="denoising_process"
ref_idx=100
scale_factor=16
python3 ../sample.py\
    --mode=$mode\
    --model_params="$model_params"\
    --img_size=$img_size\
    --data_dir="/home/dmeta0304/Documents/datasets/"\
    --ref_idx=$ref_idx\
    --scale_factor=$scale_factor\
    --batch_size=35\
    --save_path="$save_dir/$mode/ref_idx=$ref_idx-scale_factor=$scale_factor-0.jpg"\
