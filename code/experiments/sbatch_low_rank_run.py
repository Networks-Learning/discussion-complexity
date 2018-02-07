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
@click.option('--op-dir', 'output_dir', help='Files to read/write stdout files to.', default='/NL/crowdjudged/work/matrix-completion/batch-out-low-rank')
@click.option('--suffix', 'suffix', help='Suffix to identify output embeddings.', default='1BMC-r')
@click.option('--uv/--no-uv', 'UV', help='Whether to run in UV mode.', default=False)
@click.option('--dry-run/--no-dry-run', 'dry', help='Dry run.', default=True)
def cmd(in_ctx_compiled_csv, base_dir, incr, incr_op, output_dir, suffix, UV, dry):
    """Read data from IN_CTX_COMPILED_CSV, run 1BMC-r embedding code on it and save the output to M_partial.{SUFFIX}.{c_vec,v_vec}."""

    os.makedirs(output_dir, exist_ok=True)
    df = pd.read_csv(in_ctx_compiled_csv)

    alpha = 10.0
    sigma = 10.0

    UV_str = '' if not UV else '.UV'
    UV_arg = '--uv' if UV else '--not-uv'

    # Hint: ctx_compiled.csv has such headings.
    for sat_1d, sat_2d, ctx_id in df[['1D_sat', '2D_sat', 'context_id']].values:
        n_dim = 2 if sat_1d == 'unsat' else 1
        # sigma = 100.0 if sat_1d == 'sat' else 1.0

        in_mat_file = os.path.join(base_dir, ctx_id, 'M_partial.mat')
        stdout_file = f'{output_dir}/ctx.{ctx_id}.out'

        if incr:
            out_c_file = in_mat_file + '.' + suffix + UV_str + '.c_vecs'
            out_v_file = in_mat_file + '.' + suffix + UV_str + '.v_vecs'

            if os.path.exists(out_c_file) and os.path.exists(out_v_file) and (not incr_op or os.path.exists(stdout_file)):
                print('Not processing {} because output files exist.'.format(in_mat_file))
                continue

        # Giving a generous 60 minutes and 5Gb for the optimization to complete.
        cmd = f'sbatch --mem=5000 --time=120 -o "{stdout_file}" ./sbatch_low_rank_job.sh {in_mat_file} {n_dim} {sigma} {alpha} {suffix} {UV_arg}'
        if dry:
            print(cmd)
        else:
            os.system(cmd)

if __name__ == '__main__':
    cmd()
