#!/bin/bash

source ~/anaconda3/etc/profile.d/conda.sh
conda activate control
CUDA_VISIBLE_DEVICES=0 python tutorial_train.py
