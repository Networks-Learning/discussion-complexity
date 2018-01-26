#!/usr/bin/env python

import os
import pandas as pd
import sys

output_dir = "/NL/stackexchange/work/matrix-completion/batch-out-low-rank-loo"
os.makedirs(output_dir, exist_ok=True)

df = pd.read_csv(sys.argv[1])
base = sys.argv[2]

incremental = False
if len(sys.argv) > 3:
    if sys.argv[3].find('incr') != -1:
        incremental = True
    else:
        raise NotImplemented("Unknown argument {}.".format(sys.argv[3]))

# Hint: perf_script.csv
for ctx_id, seed, rank, i_loo, j_loo in df[['context_id', 'seed', 'rank', 'i_loo', 'j_loo']].values:
    in_mat_file = os.path.join(base, ctx_id, 'M_partial.mat')

    stdout_file = f'{output_dir}/ctx{ctx_id}.r{rank}.s{seed}.i{i_loo}.j{j_loo}.out'

    if incremental:
        file_tmpl = f'{in_mat_file}.r{rank}.s{seed}.i{i_loo}.j{j_loo}.low-rank.out'
        op_mat_file = file_tmpl + '.mat'
        op_loo_file = file_tmpl + '.loo'

        if os.path.exists(op_mat_file) and os.path.exists(op_loo_file):
            print('Not processing {} because output exists.'.format(in_mat_file))
            continue

    # Setting a 6Gb memory limit
    # Setting a 60 minutes time limit
    cmd = f'sbatch --mem=6000 --time=60 -o "{stdout_file}" ./sbatch_low_rank_loo_job.sh {in_mat_file} {seed} {rank} {i_loo} {j_loo}'
    # print(cmd)
    os.system(cmd)
