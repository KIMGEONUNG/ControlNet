#!/bin/bash

source config.sh
source $condapath
conda activate control
python tool_add_control.py ./models/v1-5-pruned.ckpt ./models/control_sd15_ini.ckpt
