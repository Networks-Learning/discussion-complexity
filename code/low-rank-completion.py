#!/usr/bin/env python
"""Implements the Exact low rank matrix filling as discussed in:

Bhaskar, Sonia A., and Adel Javanmard. "1-bit matrix completion under exact low-rank constraint." Information Sciences and Systems (CISS), 2015 49th Annual Conference on. IEEE, 2015.
"""

import click
import scipy.io as io
import numpy as np
from cjr.models.low_rank import make_low_rank_params, optimize_low_rank, optimize_low_rank_lbfgs


@click.command()
@click.argument('in_mat_file', type=click.Path(exists=True))
@click.option('--dims', 'dims', help='The dimensionality of the embedding.', default=2)
@click.option('--seed', 'seed', help='The random seed to use for initializing matrices, in case initial values are not given.', default=42)
@click.option('--suffix', 'suffix', help='Suffix to add before saving the embeddings.', default='bfgs')
@click.option('--init-c-vecs', 'init_c_vecs', help='File which contains initial embedding of c_vecs.', default=None)
@click.option('--init-v-vecs', 'init_v_vecs', help='File which contains initial embedding of v_vecs.', default=None)
@click.option('-i', 'i_loo', help='Which i index to LOO.', default=-1)
@click.option('-j', 'j_loo', help='Which j index to LOO.', default=-1)
@click.option('--alpha', 'alpha', help='Bound on the spikiness of M.', default=1.0)
@click.option('--sigma', 'sigma', help='What is the variance of (logistic) noise to add.', default=1.0)
@click.option('--lbfgs/--no-lbfgs', 'lbfgs', help='Whether to use LBFGS instead of BFGS.', default=True)
@click.option('--loo-output', 'loo_output', help='Where to save the LOO output.', default=None)
@click.option('--loo-only/--no-loo-only', 'loo_only', help='Whether to only save the LOO output or whether to save the complete recovered matrix.', default=False)
def cmd(in_mat_file, init_c_vecs, init_v_vecs, dims, seed, suffix, i_loo, j_loo,
        alpha, sigma, lbfgs, loo_output, loo_only):
    """Read M_partial from IN_MAT_FILE and optimize the embeddings to maximize the likelihood under the logit model."""

    M = io.loadmat(in_mat_file)['M_partial']
    rank = dims

    LOO_mode = False
    if i_loo > -1 and j_loo > -1:
        LOO = M[i_loo, j_loo]
        M[i_loo, j_loo] = 0
        LOO_mode = True

    num_comments, num_voters = M.shape
    num_embed = dims

    RS = np.random.RandomState(seed)

    if init_c_vecs is not None:
        U = np.loadtxt(init_c_vecs, ndmin=2)
    else:
        U = RS.randn(num_comments, num_embed)

    if init_v_vecs is not None:
        V = np.loadtxt(init_v_vecs, ndmin=2)
    else:
        V = RS.randn(num_voters, num_embed)

    # Scaling the initial values such that all dot products are at most = alpha / 1.21
    M_max = np.max(np.abs(U.dot(V.T)))
    U /= np.sqrt(M_max / alpha) * 1.1
    V /= np.sqrt(M_max / alpha) * 1.1

    params = make_low_rank_params(M)
    n_neg = params['n_neg']
    n_pos = params['n_pos']
    omega = params['omega']

    opt_func = optimize_low_rank if not lbfgs else optimize_low_rank_lbfgs
    for i in range(15):
        print('Iter ', i)
        ret = opt_func(U, V, n_pos, n_neg, omega, reg_wt=1.0 / (2 ** i),
                       alpha=alpha, sigma=sigma, verbose=True)
        U, V = ret['U'], ret['V']

    if LOO_mode:
        file_tmpl = f'{in_mat_file}.r{rank}.s{seed}.i{i_loo}.j{j_loo}.low-rank.out'

        if not loo_only:
            op_mat_file = file_tmpl + '.mat'
            Mhat = U.dot(V.T)
            io.savemat(op_mat_file, {'Mhat': Mhat})

        op_loo_file = loo_output if loo_output is not None else file_tmpl + '.loo'
        loo_pred = U[i_loo, :].dot(V[j_loo, :])
        with open(op_loo_file, 'wt') as f:
            f.write('{}, {}'.format(LOO, loo_pred))
    else:
        np.savetxt(in_mat_file + '.' + suffix + '.c_vecs', U)
        np.savetxt(in_mat_file + '.' + suffix + '.v_vecs', V)

    print('Done.')


if __name__ == '__main__':
    cmd()
