#!/usr/bin/env python

import os
import pandas as pd
import click


OUTPUT_DIR = '/NL/crowdjudged/work/demo/'


@click.command()
@click.argument('m', type=int)
@click.argument('n', type=int)
@click.argument('z3_num_csv', type=click.Path(exists=True))
@click.option('--base-dir', 'base_dir', help='Where to save the output.', type=click.Path(), default=OUTPUT_DIR)
@click.option('--timeout', 'timeout', help='Minutes to grant Z3 to look for a solution.', default=120)
@click.option('--dims', 'dims', help='Dims to update.', default=120 * 1000)
@click.option('--dry/--no-dry', help='Whether to just print the commands instead of running them.', default=True)
@click.option('--mem', help='How much memory usage to allow for each process.', default=5000)
@click.option('--incremental/--no-incremental', help='Run only for cases for which the output file does not exist.', default=False)
def cmd(m, n, z3_num_csv, timeout, base_dir, mem, dry, dims, incremental):
    """Run Z3 for M * N sized matrix and update results in Z3_NUM_CSV.
    Run z3_collate_num.py with an empty CSV to populate it initially."""

    df = pd.read_csv(z3_num_csv)

    assert 2 ** (n * m) == df.shape[0], "CSV and m, n do not match."

    std_dir = os.path.join(base_dir, '{}x{}-stdout'.format(m, n))
    op_dir = os.path.join(base_dir, '{}x{}'.format(m, n))
    os.makedirs(op_dir, exist_ok=True)
    os.makedirs(std_dir, exist_ok=True)

    z3_timeout = (timeout * 60 - 10) * 1000  # Z3 timeout in seconds, with 10 second buffer

    for i in df.i.values:
        stdout_file = os.path.join(std_dir, "{}.{}D.%j".format(i, dims))
        output_file = os.path.join(op_dir, '{}'.format(i), '{}D.out'.format(dims))

        if incremental and os.path.exists(output_file):
            print('Not running for {} because {} exists.'.format(i, output_file))
            continue

        cmd = f'sbatch --mem={mem} --time={timeout} -o "{stdout_file}" ./sbatch_Z3_num_job.sh {m} {n} {i} {dims} {z3_timeout}'

        if dry:
            print(cmd)
        else:
            os.system(cmd)


if __name__ == '__main__':
    cmd()
