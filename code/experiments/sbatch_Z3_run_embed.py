#!/usr/bin/env python

import os
import pandas as pd
import sys

output_dir = "/NL/stackexchange/work/matrix-completion/batch-out-Z3-embed"
os.makedirs(output_dir, exist_ok=True)

df = pd.read_csv(sys.argv[1])
base = sys.argv[2]

for sat_1d, sat_2d, ctx_id in df[['1D_sat', '2D_sat', 'context_id']].values:
    if sat_2d == 'sat':
        n_dim = 2 if sat_1d == 'unsat' else 1
        in_file = os.path.join(base, ctx_id, 'M_partial.mat')
        stdout_file = f'{output_dir}/ctx.{ctx_id}.out'
        timeout = 25
        cmd = f'sbatch --mem=4000 --time=30 -o "{stdout_file}" ./sbatch_Z3_job_embed.sh {in_file} {n_dim} {timeout}'
        os.system(cmd)
        # print(cmd)
    else:
        print('Not running for {}'.format(ctx_id))
