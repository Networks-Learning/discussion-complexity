#!/usr/bin/env python

import pandas as pd
import sys
import os
import click

OUTPUT_DIR = '/NL/stackexchange/work/matrix-completion/batch-out-nD-sat'


@click.command()
@click.argument('ctx_compiled_csv')
@click.argument('base_dir')
@click.option('--solved-csv', 'output_csv', help='File listing already knowing statuses', default='')
@click.option('--std-dir', 'output_dir', help='Where to save the stdout files.', default=OUTPUT_DIR)
@click.option('--timeout', 'timeout', help='How long to let Z3 run (minutes)?', default=480)
@click.option('--dims', 'dims', help='Number of dimensions to try to fit for.', default=2)
@click.option('--dry/--no-dry', 'dry', help='Whether to do a dry run or a real run.', default=True)
def cmd(ctx_compiled_csv, base_dir, output_csv, output_dir, timeout, dims, dry):
    """Read CTX_COMPILED_CSV, determine whether the M_partial.mat matrix in BASE_DIR has sign-rank of less than `dims` or not and save the result with the M_partial.{dims}_sat file."""

    df = pd.read_csv(ctx_compiled_csv)
    os.makedirs(output_dir, exist_ok=True)

    col_name = f'{dims}D_sat'

    if os.path.exists(output_csv):
        output_df = pd.read_csv(output_csv)
        assert set(output_df.context_id) == set(df.context_id), "Input/Output mismatch."
        if col_name not in output_df.columns and dims > 1:
            prev_col_name = '{}D_sat'.format(dims - 1)
            if prev_col_name in output_df.columns:
                output_df[col_name] = output_df[prev_col_name]
                output_df[col_name][output_df[col_name] == 'unsat'] = 'unknown'
            else:
                output_df[col_name] = 'unknown'
    else:
        output_df = pd.DataFrame([
            {'context_id': ctx_id, col_name: 'unknown'}
            for ctx_id in df.context_id
        ])

    unknown_ctx_id = set(output_df.context_id[output_df[col_name] == 'unknown'])

    for ctx_id in df.context_id.values:
        if ctx_id in unknown_ctx_id:
            in_file = os.path.join(base_dir, ctx_id, 'M_partial.mat')
            op_file = in_file + '{}D_sat'.format(dims)

            stdout_file = f'{output_dir}/ctx{ctx_id}.{dims}D.%j'
            sbatch_timeout = timeout + 10
            command = f'sbatch --time={sbatch_timeout} --mem=5000 -o {stdout_file} ./sbatch_2D_sat_job.sh {in_file} {op_file} {timeout} {dims}'

            if dry:
                print(command)
            else:
                os.system(command)


if __name__ == '__main__':
    cmd()
