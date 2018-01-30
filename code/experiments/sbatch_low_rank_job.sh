#!/bin/bash

set -eo pipefail

in_file=$1
n_dim=$2
sigma=$3
alpha=$4
suffix=$5

source conda.sh

cd /home/utkarshu/prog/crowdjudged.git/code;

python low-rank-completion.py "${in_file}" --init-c-vecs "${in_file}.c_vecs" --init-v-vecs "${in_file}.v_vecs" --dims ${n_dim} --sigma ${sigma} --alpha ${alpha} --suffix ${suffix}
