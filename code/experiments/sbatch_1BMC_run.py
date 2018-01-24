#!/usr/bin/env python

import os
import pandas as pd
import sys

output_dir = "sbatch-out"
os.makedirs(output_dir, exist_ok=True)

df = pd.read_csv(sys.argv[1])
base = sys.argv[2]

for ctx_id, seed, rank, i_loo, j_loo in df[['context_id', 'seed', 'rank', 'i_loo', 'j_loo']].values:
    in_file = os.path.join(base, ctx_id, 'M_partial.mat')
    stdout_file = f'{output_dir}/sbatch.ctx{ctx_id}.r{rank}.s{seed}.i{i_loo}.j{j_loo}.out'
    # Setting a 4Gb memory limit
    # Setting a 20 minutes time limit
    cmd = f'sbatch --mem=4000 --time=20 -o "{stdout_file}" ./sbatch_1BMC_job.sh {in_file} {seed} {rank} {i_loo} {j_loo}'
    # print(cmd)
    os.system(cmd)
