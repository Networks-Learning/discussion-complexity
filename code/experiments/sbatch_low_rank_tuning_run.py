#!/usr/bin/env python

import click
import os
import pandas as pd


@click.command()
@click.argument('tuning_csv_file')
@click.argument('base_dir')
@click.option('--incremental/--no-incremental', 'incr', help='Whether to produce only the output files which do not exist or to reproduce all files.', default=False)
@click.option('--loo-op-dir', 'loo_op_dir', help='Where to save the output LOO files.', default='/NL/stackexchange/work/matrix-completion/1BMC-tuning')
@click.option('--std-op-dir', 'std_op_dir', help='Where to save the stdout files.', default='/NL/stackexchange/work/matrix-completion/1BMC-tuning-stdout')
def cmd(tuning_csv_file, base_dir, loo_op_dir, std_op_dir, incr):
    """Read parameters for the 1BMC-r from tuning_csv_file and start SLURM jobs."""
    os.makedirs(loo_op_dir, exist_ok=True)
    os.makedirs(std_op_dir, exist_ok=True)

    df = pd.read_csv(tuning_csv_file)

    for context_id, seed, rank, alpha, sigma, i_loo, j_loo in df[['context_id', 'seed', 'rank', 'alpha', 'sigma', 'i_loo', 'j_loo']].values:
        in_mat_file = os.path.join(base_dir, context_id, 'M_partial.mat')

        op_tmpl = f'ctx{context_id}.r{rank}.s{seed}.i{i_loo}.j{j_loo}.a{alpha}.sigma{sigma}.out'
        op_loo_file = os.path.join(loo_op_dir, 'loo.' + op_tmpl)
        op_std_file = os.path.join(std_op_dir, 'std.' + op_tmpl)

        if incr and os.path.exists(op_loo_file):
            print('Not starting for {} as output already exists.'.format(context_id))
            continue

        cmd = f'sbatch --mem=5000 --time=60 -o "{op_std_file}" ./sbatch_low_rank_tuning_job.sh {in_mat_file} {seed} {rank} {alpha} {sigma} {i_loo} {j_loo} {context_id} "{op_loo_file}"'
        # print(cmd)
        os.system(cmd)


if __name__ == '__main__':
    cmd()
