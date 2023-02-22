#!/bin/bash

source ~/anaconda3/etc/profile.d/conda.sh
conda activate control
CUDA_VISIBLE_DEVICES=0 python tool_add_control.py ./models/v1-5-pruned.ckpt ./models/control_sd15_ini.ckpt
