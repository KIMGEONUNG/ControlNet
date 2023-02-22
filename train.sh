#!/bin/bash

source config.sh
source $condapath
conda activate control

if [[ -z $@ ]]; then
    python train.py
else
    python train.py $@
fi
