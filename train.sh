#!/bin/bash

source config.sh
source $condapath
conda activate control

if [[ -z $@ ]]; then
  echo -e "\033[31mError: Specify config like '--configs/default.yaml' \033[0m" >&2
  exit 1
fi
python train.py $@
