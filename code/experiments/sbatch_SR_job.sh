#!/bin/bash

set -eo pipefail

inp_mat_file=$1
op_mat_file=$2
op_SC_file=$3
min_avg=$4
transpose=$5

source conda.sh

cd /home/utkarshu/prog/crowdjudged.git/code;

python SR_fill.py ${inp_mat_file} ${op_mat_file} ${op_SC_file} ${min_avg} ${transpose}
