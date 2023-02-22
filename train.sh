#!/bin/bash

source config.sh
source $condapath
conda activate control

python train.py
