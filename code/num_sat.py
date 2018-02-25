#!/usr/bin/env python

import click
import os
import z3
import numpy as np
import cjr.models.dim as D

OUTPUT_DIR = '/NL/crowdjudged/work/demo/4x4/'


def i_to_M(n, m, i):
    """Convert a number 'i' to an array."""
    M = np.zeros(shape=(n, m), dtype=int)
    for k in range(n):
        for l in range(m):
            if i & (2 ** (n * k + l)):
                M[k, l] = 1
            else:
                M[k, l] = -1

    return M


@click.command()
@click.argument('m', type=int)
@click.argument('n', type=int)
@click.argument('i', type=int)
@click.option('--output-dir', 'output_dir', help='Where to save the output.', default=OUTPUT_DIR)
@click.option('--dims', help='How many dimensions to try to fit.', default=2)
@click.option('--timeout', help='What timeout to use (ms).', default=120 * 1000)
@click.option('--incremental', help='Do not run if an output-file with this dimension already exists.', default=False)
def cmd(n, m, i, output_dir, dims, timeout, incremental):
    """Assume that an NxM matrix is encoded as a binary number and is passed as I.
    Try to see whether the scoring pattern can be fit into 2D and save the output embeddings, if it can be."""

    output_dir = os.path.join(output_dir, f'{i}')
    os.makedirs(output_dir, exist_ok=True)

    output_file = os.path.join(output_dir, f'{dims}D.out')

    if os.path.exists(output_file) and incremental:
        print('Not running for m={}, n={}, i={} as the file {} already exists.'.format(m, n, i, output_file))
    else:
        M = i_to_M(n, m, i)
        voting_pats = D.sign_mat_to_voting_pats(M)

        ctx = z3.Context()

        prob, c_vars, v_vars = D.create_z3_prob(ctx=ctx, n_dim=dims, voting_patterns=voting_pats)
        prob.set('timeout', timeout)

        res = prob.check()

        if str(res) == 'sat':
            c_vec, v_vec = D.make_vectors_from_z3_soln(prob=prob, c_vars=c_vars, v_vars=v_vars)

            c_file = output_file + '.c_vecs'
            v_file = output_file + '.v_vecs'

            np.savetxt(c_file, c_vec)
            np.savetxt(v_file, v_vec)

        with open(output_file, 'wt') as f:
            f.write(str(res))


if __name__ == '__main__':
    cmd()
