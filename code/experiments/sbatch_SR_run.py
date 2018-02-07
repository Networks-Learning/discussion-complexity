#!/usr/bin/env python

import os
import pandas as pd
import click
import sys

OUTPUT_DIR = '/NL/stackexchange/work/matrix-completion/batch-out-SR-pure'


@click.command()
@click.argument('ctx_compiled_csv')
@click.argument('base_dir')
@click.option('--std-dir', 'output_dir', help='Where to save the stdout files.', default=OUTPUT_DIR)
@click.option('--min-avg/--no-min-avg', 'use_min_avg', help='Whether to minimize the average number of SC or worst case.', default=True)
@click.option('--transpose/--no-transpose', 'transpose', help='Whether to transpose M matrix or not.', default=False)
@click.option('--incremental/--no-incremental', help='Run only for cases for which the output file does not exist.', default=False)
@click.option('--dry/--no-dry', help='Whether to just print the commands instead of running them.', default=True)
@click.option('--mem', help='How much memory usage to allow for each process.', default=5000)
@click.option('--timeout', help='How much time to allow for each process (minutes).', default=240)
def cmd(ctx_compiled_csv, base_dir, use_min_avg, incremental, output_dir,
        transpose, dry, mem, timeout):
    """Read parameters from CTX_COMPILED_CSV and run SR on it while reading M_partial.mat from BASE_DIR."""
    df = pd.read_csv(ctx_compiled_csv)
    M_file = 'M_partial.mat'

    os.makedirs(output_dir, exist_ok=True)
    min_avg_str = 'min-avg.' if use_min_avg else ''
    min_avg_cmd = '--min-avg' if use_min_avg else '--no-min-avg'

    T_str = 'T.' if transpose else ''
    T_cmd = '--no-transpose' if not transpose else '--transpose'

    for ctx_id in df.context_id:
        in_file = os.path.join(base_dir, ctx_id, M_file)
        stdout_file = f'{output_dir}/ctx{ctx_id}' + min_avg_str + T_str + 'out'

        op_mat_file = f'{in_file}.{min_avg_str}{T_str}SR.mat'
        op_SC_file = f'{in_file}.{min_avg_str}{T_str}SR.SC'

        if incremental:
            if os.path.exists(op_mat_file) and os.path.exists(op_SC_file):
                print('Not running for {} because output exists.'.format(ctx_id))
                continue

        cmd = f'sbatch --mem={mem} --time={timeout} -o "{stdout_file}.%j" ./sbatch_SR_job.sh {in_file} {op_mat_file} {op_SC_file} {min_avg_cmd} {T_cmd}'

        if dry:
            print(cmd)
        else:
            os.system(cmd)


if __name__ == '__main__':
    cmd()
