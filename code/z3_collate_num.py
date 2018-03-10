#!/usr/bin/env python

import click
import os
import pandas as pd
import sys

# base_dir='/NL/crowdjudged/work/demo/4x4'


@click.command()
@click.argument('m', type=int)
@click.argument('n', type=int)
@click.argument('out_sat_csv')
@click.argument('base_dir', type=click.Path(exists=True))
@click.option('--max-dim', 'max_dim', help='Highest dimension to search for.', default=3)
@click.option('--force/--no-force', 'force', help='If true, overwrites the out_sat_csv.', default=False)
@click.option('--incremental/--no-incremental', 'incremental', help='If true, does not read the data which already exists in out_sat_csv.', default=False)
def cmd(m, n, out_sat_csv, base_dir, max_dim, force, incremental):
    """Read output created by Z3 for matrix of size NxM from BASE_DIR and collate it into out_sat_csv."""

    if os.path.exists(out_sat_csv) and not force:
        print('Cannot overwrite {}. Use --force.'.format(out_sat_csv))
        sys.exit(-1)

    if os.path.exists(out_sat_csv) and incremental:
        print('Continuing exploration from {}'.format(out_sat_csv))
        df = pd.read_csv(out_sat_csv)
    else:
        df = pd.DataFrame({'i': i} for i in range(2 ** (m * n)))

    filled = 0
    last_col = None

    for d in range(1, max_dim + 1):
        dim_col = '{}D_sat'.format(d)
        if dim_col not in df.columns:
            df[dim_col] = 'unknown'

        # If it was 'sat' in (n-1)D, it will be 'sat' in nD as well.
        if last_col is not None:
            df[dim_col][df[last_col] == 'sat'] = 'sat'

        last_col = dim_col

        dim_vals = df[dim_col].copy().values

        for idx, (i, sat_status) in enumerate(df[['i', dim_col]].values):
            if sat_status == 'unknown':
                Z3_sat_file = os.path.join(base_dir, f'{i}', f'{d}D.out')
                if os.path.exists(Z3_sat_file):
                    with open(Z3_sat_file) as f:
                        found_sat = f.readline().strip()
                        if found_sat != 'unknown':
                            # print('Found {} was {} in {}D'.format(i, found_sat, d))
                            filled += 1
                            dim_vals[idx] = found_sat

        df[dim_col] = dim_vals

    df.to_csv(out_sat_csv, index=False)
    print('Done. Filled {} entries.'.format(filled))


if __name__ == '__main__':
    cmd()
