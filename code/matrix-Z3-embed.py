#!/usr/bin/env python

import click
import scipy.io as io
import cjr.models.dim as D
import z3
import numpy as np


@click.command()
@click.argument('mat_file', type=click.Path(exists=True))
@click.option('--dim', help='What dimension to use while splitting matrix.', default=2)
@click.option('--timeout', help='What timeout to use (minutes).', default=20)
def cmd(mat_file, dim, timeout):
    """Read the partial matrix in MAT_FILE and save embeddings for the file to `mat_file.commenters` and `mat_file.voters` file."""

    M = io.loadmat(mat_file)['M_partial']

    prob, c_vars, v_vars = D.create_z3_prob(n_dim=dim, voting_patterns=D.sign_mat_to_voting_pats(M))
    prob.set('timeout', timeout * 60 * 1000)

    res = prob.check()

    if res == z3.sat:
        c_vecs, v_vecs = D.make_vectors_from_z3_soln(prob, c_vars, v_vars)
        np.savetxt(mat_file + '.c_vecs', c_vecs)
        np.savetxt(mat_file + '.v_vecs', v_vecs)
        print('Done.')
    else:
        print('Result for {} was {} instead of "sat"'.format(mat_file, str(res)))


if __name__ == '__main__':
    cmd()
