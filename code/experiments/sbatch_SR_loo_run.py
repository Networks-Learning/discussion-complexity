#!/usr/bin/env python

import os
import pandas as pd
import sys
import click

OUTPUT_DIR = "/NL/stackexchange/work/matrix-completion/batch-out-SR"


@click.command()
@click.argument('in_perf_script_csv')
@click.argument('base_dir')
@click.option('--std-dir', 'output_dir', help='Where to save the stdout files.', default=OUTPUT_DIR)
@click.option('--incremental/--no-incremental', help='Run only for cases for which the output file does not exist.', default=False)
def cmd(in_perf_script_csv, base_dir, incremental, output_dir):
    """Read parameters from IN_PERF_SCRIPT_CSV and run SR on it while reading M_partial.mat from BASE_DIR."""
    df = pd.read_csv(in_perf_script_csv)
    M_file = 'M_partial.mat'

    seen = set()
    for ctx_id, seed, i_loo, j_loo in df[['context_id', 'seed', 'i_loo', 'j_loo']].values:
        if (ctx_id, seed) in seen:
            continue
        else:
            seen.add((ctx_id, seed))

        in_file = os.path.join(base_dir, ctx_id, M_file)
        stdout_file = f'{output_dir}/ctx{ctx_id}.s{seed}.i{i_loo}.j{j_loo}.out'

        op_file_tmpl = f'.s{seed}.i{i_loo}.j{j_loo}.SR.out'
        op_mat_file = os.path.join(base_dir, ctx_id, M_file + op_file_tmpl + '.mat')
        op_loo_file = os.path.join(base_dir, ctx_id, M_file + op_file_tmpl + '.loo')
        op_SC_file = os.path.join(base_dir, ctx_id, M_file + op_file_tmpl + '.SC')

        if incremental:
            if os.path.exists(op_mat_file) and os.path.exists(op_loo_file) and os.path.exists(op_SC_file):
                print('Not running for {}, output files exist.'.format((ctx_id, seed)))
                continue

        cmd = f'sbatch --mem=5000 --time=60 -o "{stdout_file}.%j" ./sbatch_SR_loo_job.sh {in_file} {seed} {i_loo} {j_loo} {op_mat_file} {op_loo_file} {op_SC_file}'
        # print(cmd)
        os.system(cmd)

if __name__ == '__main__':
    cmd()
