#!/bin/bash

inp_mat_file=$1
seed=$2
rank=$3
i_loo=$4
j_loo=$5

op_mat_file="${inp_mat_file}.r${rank}.s${seed}.i${i_loo}.j${j_loo}.out.mat"
op_loo_file="${inp_mat_file}.r${rank}.s${seed}.i${i_loo}.j${j_loo}.out.loo"

matlab -nosplash -nodesktop -nodisplay -nojvm -r "maxNumCompThreads(1); cd /home/utkarshu/prog/work/1BMC; fill_matrix_loo('${inp_mat_file}', '${op_mat_file}', '${op_loo_file}', ${rank}, ${i_loo} + 1, ${j_loo} + 1); exit;"
