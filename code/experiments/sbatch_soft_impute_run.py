#!/usr/bin/env python

import os
import pandas as pd
import sys
import time
import click


OUTPUT_DIR = "/NL/crowdjudged/work/matrix-completion/batch-out-soft-impute-loo"


@click.command()
@click.argument('perf_script_csv')
@click.argument('base_dir')
@click.option('--output-dir', 'output_dir', default=OUTPUT_DIR, show_default=True)
@click.option('--dry-run/--no-dry-run', 'dry', default=False, show_default=True)
@click.option('--incremental/--no-incremental', 'incremental', default=False, show_default=True)
@click.option('--max-rank', 'max_rank', help='Only run experiments until this rank.', default=-1, show_default=True)
def cmd(perf_script_csv, base_dir, output_dir, dry, incremental, max_rank):
    """Read arguments from PERF_SCRIPT_CSV, run the LOO experiments with soft-impute and save the output/stdout to the respective folders."""

    os.makedirs(output_dir, exist_ok=True)
    df = pd.read_csv(perf_script_csv)

    # Hint: perf_script.csv
    for ctx_id, seed, rank, i_loo, j_loo in df[['context_id', 'seed', 'rank', 'i_loo', 'j_loo']].values:
        in_mat_file = os.path.join(base_dir, ctx_id, 'M_partial.mat')

        if max_rank > 0 and rank > max_rank:
            continue

        stdout_file = f'{output_dir}/ctx{ctx_id}.r{rank}.s{seed}.i{i_loo}.j{j_loo}.soft-impute.out'

        if incremental:
            file_tmpl = f'{in_mat_file}.r{rank}.s{seed}.i{i_loo}.j{j_loo}.soft-impute.out'
            op_mat_file = file_tmpl + '.mat'
            op_loo_file = file_tmpl + '.loo'

            if os.path.exists(op_mat_file) and os.path.exists(op_loo_file):
                print('Not processing {} because output exists.'.format(in_mat_file))
                continue

        # Setting a 5Gb memory limit
        # Setting a 120 minutes time limit
        cmd = f'sbatch --mem=5000 --time=120 -o "{stdout_file}" ./sbatch_soft_impute_job.sh {in_mat_file} {seed} {rank} {i_loo} {j_loo}'

        if dry:
            print(cmd)
        else:
            os.system(cmd)
            time.sleep(0.1)  # Throttling job submission


if __name__ == '__main__':
    cmd()
