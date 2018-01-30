#!/usr/bin/env python

import click
import os
import pandas as pd
import sys


@click.command()
@click.argument('in_ctx_compiled_csv')
@click.argument('base_dir')
@click.option('--incremental/--no-incremental', 'incr', help='Do only incremental update', default=False)
@click.option('--incremental-op/--no-incremental-op', 'incr_op', help='Rely on the stdout files to determine which jobs have run.', default=False)
@click.option('--op-dir', 'output_dir', help='Files to read/write stdout files to.', default='/NL/stackexchange/work/matrix-completion/batch-out-low-rank')
@click.option('--suffix', 'suffix', help='Suffix to identify output embeddings.', default='1BMC-r')
def cmd(in_ctx_compiled_csv, base_dir, incr, incr_op, output_dir, suffix):
    """Read data from IN_CTX_COMPILED, run 1BMC-r embedding code on it and save the output to M_partial.{SUFFIX}.{c_vec,v_vec}."""

    os.makedirs(output_dir, exist_ok=True)
    df = pd.read_csv(in_ctx_compiled_csv)

    alpha = 20.0

    # Hint: ctx_compiled.csv has such headings.
    for sat_1d, sat_2d, ctx_id in df[['1D_sat', '2D_sat', 'context_id']].values:
        if sat_2d == 'sat':
            n_dim = 2 if sat_1d == 'unsat' else 1
            sigma = 100.0 if sat_1d == 'sat' else 1.0

            in_file = os.path.join(base_dir, ctx_id, 'M_partial.mat')
            stdout_file = f'{output_dir}/ctx.{ctx_id}.out'

            if incr:
                out_c_file = in_file + f'.{suffix}.c_vecs'
                out_v_file = in_file + f'.{suffix}.c_vecs'

                if os.path.exists(out_c_file) and os.path.exists(out_v_file) and (not incr_op or os.path.exists(stdout_file)):
                    print('Not processing {} because output files exist.'.format(in_file))
                    continue

            # Giving a generous 60 minutes and 5Gb for the optimization to complete.
            cmd = f'sbatch --mem=5000 --time=60 -o "{stdout_file}" ./sbatch_low_rank_job.sh {in_file} {n_dim} {sigma} {alpha} {suffix}'
            os.system(cmd)
            # print(cmd)
        else:
            print('Not processing {} because it is not 2D_sat'.format(ctx_id))
            # pass

if __name__ == '__main__':
    cmd()
