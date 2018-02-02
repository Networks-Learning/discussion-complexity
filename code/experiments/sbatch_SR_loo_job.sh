#!/bin/bash

set -eo pipefail

inp_mat_file=$1
seed=$2
i_loo=$3
j_loo=$4
op_mat_file=$5
op_loo_file=$6
op_SC_file=$7

source conda.sh

cd /home/utkarshu/prog/crowdjudged.git/code;

python SR_fill.py ${inp_mat_file} ${op_mat_file} ${op_SC_file} --op-loo ${op_loo_file} -i ${i_loo} -j ${j_loo} --seed ${seed} --min-avg
