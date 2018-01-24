#!/usr/bin/env python

import os
import pandas as pd
import sys

output_dir = "/NL/stackexchange/work/matrix-completion/batch-out-Z3"
os.makedirs(output_dir, exist_ok=True)

df = pd.read_csv(sys.argv[1])
base = sys.argv[2]
seen = set()

# Hint: merged_df.csv has such a structure.
for sat_2d, sat_1d, ctx_id, seed, rank, i_loo, j_loo in df[['2D_sat', '1D_sat', 'context_id', 'seed', 'rank', 'i_loo', 'j_loo']].values:

    if (ctx_id, seed) in seen:
        continue
    else:
        seen.add((ctx_id, seed))

    in_file = os.path.join(base, ctx_id, 'M_partial.mat')
    stdout_file = f'{output_dir}/sbatch.ctx{ctx_id}.s{seed}.i{i_loo}.j{j_loo}.out'
    # Setting a 4Gb memory limit
    # Setting a 20 minutes time limit
    cmd = f'sbatch --mem=4000 --time=20 -o "{stdout_file}" ./sbatch_Z3_job.sh {in_file} {seed} {i_loo} {j_loo} {sat_2d} {sat_1d}'
    # print(cmd)
    os.system(cmd)
