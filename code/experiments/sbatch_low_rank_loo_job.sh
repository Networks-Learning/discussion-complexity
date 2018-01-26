#!/bin/bash

set -eo pipefail

in_file=$1
seed=$2
rank=$3
i_loo=$4
j_loo=$5

source conda.sh

cd /home/utkarshu/prog/crowdjudged.git/code;

python low-rank-completion.py "${in_file}" --dims ${rank} -i ${i_loo} -j ${j_loo} --seed ${seed} --lbfgs
