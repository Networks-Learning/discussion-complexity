#!/usr/bin/env python

import os
import pandas as pd
import sys
from collections import defaultdict

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


sigma_by_rank = defaultdict(lambda: 1.0)
sigma_by_rank[1] = 100.0

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

    # Setting a 10Gb memory limit
    # Setting a 120 minutes time limit
    alpha = 20.0
    sigma = sigma_by_rank[rank]
    cmd = f'sbatch --mem=10000 --time=120 -o "{stdout_file}" ./sbatch_low_rank_loo_job.sh {in_mat_file} {seed} {rank} {alpha} {sigma} {i_loo} {j_loo}'
    # print(cmd)
    os.system(cmd)
