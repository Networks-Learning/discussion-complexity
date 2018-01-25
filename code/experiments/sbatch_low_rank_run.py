#!/usr/bin/env python

import os
import pandas as pd
import sys

output_dir = "/NL/stackexchange/work/matrix-completion/batch-out-low-rank"
os.makedirs(output_dir, exist_ok=True)

df = pd.read_csv(sys.argv[1])
base = sys.argv[2]

incremental = False
if len(sys.argv) > 3:
    if sys.argv[3].find('incr') != -1:
        incremental = True
    else:
        raise NotImplemented("Unknown argument {}.".format(sys.argv[3]))

# Hint: ctx_compiled.csv has such headings.
for sat_1d, sat_2d, ctx_id in df[['1D_sat', '2D_sat', 'context_id']].values:
    if sat_2d == 'sat':
        n_dim = 2 if sat_1d == 'unsat' else 1
        in_file = os.path.join(base, ctx_id, 'M_partial.mat')

        if incremental:
            out_c_file = in_file + '.bfgs.c_vecs'
            out_v_file = in_file + '.bfgs.c_vecs'
            if os.path.exists(out_c_file) and os.path.exists(out_v_file):
                print('Not processing {} because output exists.'.format(in_file))
                continue

        stdout_file = f'{output_dir}/ctx.{ctx_id}.out'

        # Giving a generous 40 minutes and 4Gb for the optimization to complete.
        cmd = f'sbatch --mem=4000 --time=40 -o "{stdout_file}" ./sbatch_low_rank_job.sh {in_file} {n_dim}'
        os.system(cmd)
        # print(cmd)
    else:
        print('Not processing {} because it is not 2D_sat'.format(ctx_id))
        # pass
