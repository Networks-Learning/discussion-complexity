#!/usr/bin/env python

import click
import os
import pandas as pd
import sys


@click.command()
@click.argument('in_sat_csv', type=click.Path(exists=True))
@click.argument('out_sat_csv')
@click.argument('base_dir', type=click.Path(exists=True))
@click.option('--max-dim', 'max_dim', help='Highest dimension to search for.', default=3)
@click.option('--force/--no-force', 'force', help='If true, overwrites the out_sat_csv.', default=False)
def cmd(in_sat_csv, out_sat_csv, base_dir, max_dim, force):
    """Read IN_SAT_CSV, find whether there is any Z3 data for matrices of those sizes."""
    df = pd.read_csv(in_sat_csv)

    if os.path.exists(out_sat_csv) and not force:
        print('Cannot overwrite {}. Use --force.'.format(out_sat_csv))
        sys.exit(-1)

    filled = 0
    last_col = None

    for d in range(2, max_dim + 1):
        dim_col = '{}D_sat'.format(d)
        if dim_col not in df.columns:
            df[dim_col] = 'unknown'

        # If it was 'sat' in (n-1)D, it will be 'sat' in nD as well.
        if last_col is not None:
            df[dim_col][df[last_col] == 'sat'] = 'sat'

        last_col = dim_col

        dim_vals = df[dim_col].copy().values

        for idx, (ctx_id, sat_status) in enumerate(df[['context_id', dim_col]].values):
            if sat_status == 'unknown':
                M_sat_file = os.path.join(base_dir, ctx_id, 'M_partial.mat{}D_sat'.format(d))
                if os.path.exists(M_sat_file):
                    with open(M_sat_file) as f:
                        found_sat = f.readline().strip()
                        if found_sat != 'unknown':
                            print('Found {} was {} in {}D'.format(ctx_id, found_sat, d))
                            filled += 1
                            dim_vals[idx] = found_sat

        df[dim_col] = dim_vals

    df.to_csv(out_sat_csv, index=False)
    print('Done. Filled {} entries.'.format(filled))


if __name__ == '__main__':
    cmd()
