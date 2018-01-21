#!/bin/bash

set -eo pipefail

inp_mat_file=$1
seed=$2
i_loo=$3
j_loo=$4
sat_2d=$5
sat_1d=$6

op_mat_file="${inp_mat_file}.r${rank}.s${seed}.i${i_loo}.j${j_loo}.Z3.out.mat"
op_loo_file="${inp_mat_file}.r${rank}.s${seed}.i${i_loo}.j${j_loo}.Z3.out.loo"

source conda.sh

cd /home/utkarshu/prog/crowdjudged.git/code;
python z3_fill.py ${inp_mat_file} ${op_mat_file} ${op_loo_file} -i ${i_loo} -j ${j_loo} --sat_2d ${sat_2d} --sat_1d ${sat_1d} --seed ${seed}
