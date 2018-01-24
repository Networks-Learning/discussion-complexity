#!/usr/bin/env python
"""Implements the Exact low rank matrix filling as discussed in:

Bhaskar, Sonia A., and Adel Javanmard. "1-bit matrix completion under exact low-rank constraint." Information Sciences and Systems (CISS), 2015 49th Annual Conference on. IEEE, 2015.

"""

import click
import scipy.io as io
import numpy as np
import scipy.optimize as OP
from datetime import datetime


# TODO: Define for probit model as well.
def f(x):
    return 1. / (1 + np.exp(-x))

def f_prime(x):
    return np.exp(-x) / ((1 + np.exp(-x)) ** 2)


# TODO: The objective can be split for each row of U/V matrix, leading
# to faster optimization problems.
def obj_u(u, V, n_pos, n_neg, omega, reg_wt, alpha=1.0, verbose=False):
    U = u.reshape(-1, V.shape[1])
    M = U.dot(V.T)
    m = M[omega]
    LL = np.sum(n_pos * (m - np.log(1 + np.exp(m))) - n_neg * np.log(1 + np.exp(m)))

    M_sq = np.square(M / alpha)
    if np.any(M_sq > 1.0):
        reg = -np.inf
    else:
        reg = reg_wt * np.sum(np.log1p(-M_sq))

    if verbose:
        print('LL = {}, reg = {}'.format(LL, reg))
    return -(LL + reg)

def obj_u_prime(u, V, n_pos, n_neg, omega, reg_wt, alpha=1.0):
    U = u.reshape(-1, V.shape[1])
    M = U.dot(V.T)
    m = M[omega]
    grad_barrier_denom =  2 * reg_wt * M / ((alpha ** 2) * (1 - np.square(M / alpha)))

    # TODO: This can probably be made faster using matrix multiplication.
    grad = np.zeros_like(U)
    for i in range(M.shape[0]):
        for j in range(M.shape[1]):
            grad[i, :] += grad_barrier_denom[i, j] * V[j, :]

    grad_m = -(n_pos * (1 - f(m)) - n_neg * f(m))
    for idx, (i, j) in enumerate(zip(*omega)):
        grad[i, :] +=  grad_m[idx] * V[j, :]
    return grad.reshape(-1)


def obj_v(v, U, n_pos, n_neg, omega, reg_wt, alpha=1.0):
    V = v.reshape(-1, U.shape[1])
    u = U.reshape(-1)
    return obj_u(u, V, n_pos, n_neg, omega, reg_wt, alpha=alpha)


def obj_v_prime(v, U, n_pos, n_neg, omega, reg_wt, alpha=1.0):
    V = v.reshape(-1, U.shape[1])
    M = U.dot(V.T)
    m = M[omega]

    grad_barrier_denom = 2 * reg_wt * M / ((alpha ** 2) * (1 - np.square(M / alpha)))

    grad = np.zeros_like(V)
    for i in range(M.shape[0]):
        for j in range(M.shape[1]):
            grad[j, :] += grad_barrier_denom[i, j] * U[i, :]

    grad_m = -(n_pos * (1 - f(m)) - n_neg * f(m))
    for idx, (i, j) in enumerate(zip(*omega)):
        grad[j, :] += grad_m[idx] * U[i, :]
    return grad.reshape(-1)



@click.command()
@click.argument('in_mat_file', type=click.Path(exists=True))
@click.option('--dims', 'dims', help='The dimensionality of the embedding.', default=2)
@click.option('--seed', 'seed', help='The random seed to use for initializing matrices, in case initial values are not given.', default=42)
@click.option('--suffix', 'suffix', help='Suffix to add before saving the embeddings.', default='bfgs')
@click.option('--init-c-vecs', 'init_c_vecs', help='File which contains initial embedding of c_vecs.', default=None)
@click.option('--init-v-vecs', 'init_v_vecs', help='File which contains initial embedding of v_vecs.', default=None)
def cmd(in_mat_file, init_c_vecs, init_v_vecs, dims, seed, suffix):
    """Read M_partial from IN_MAT_FILE and optimize the embeddings to maximize the likelihood."""

    # M = io.loadmat(os.path.join(base, ctx_id, 'M_partial.mat'))['M_partial']
    M = io.loadmat(in_mat_file)['M_partial']
    num_comments, num_voters = M.shape
    num_embeds = dims

    RS = np.random.RandomState(seed)

    if init_c_vecs is not None:
        U = np.loadtxt(init_c_vecs, ndmin=2)
    else:
        U = RS.randn(num_comments, num_embed)

    if init_v_vecs is not None:
        V = np.loadtxt(init_v_vecs, ndmin=2)
    else:
        V = RS.randn(num_comments, num_embed)

    # Scaling the initial values such that all dot products are less than alpha = 1.0
    M_max = np.max(U.dot(V.T))
    U /= np.sqrt(M_max) * 1.1
    V /= np.sqrt(M_max) * 1.1

    M_pos = np.zeros((num_comments, num_voters))
    M_neg = np.zeros((num_comments, num_voters))

    M_pos[M > 0] = 1.0
    M_neg[M < 0] = 1.0

    omega = (M_pos + M_neg).nonzero()
    n_neg = M_neg[omega]
    n_pos = M_pos[omega]

    U_bfgs = []
    V_bfgs = []

    for i in range(10):
        print('Iter ', i)

        u0 = U.reshape(-1)
        [u_next, f_opt, g_opt, Bopt, func_calls, grad_calls, warnflag] = OP.fmin_bfgs(obj_u, u0, fprime=obj_u_prime, args=(V, n_pos, n_neg, omega, 1.0 / (2 ** i), 1.0), disp=True, full_output=1)
        print('{} obj = {:0.3f}'.format(datetime.now(), f_opt))
        U = u_next.reshape(*U.shape)
        U_bfgs.append(U)

        v0 = V.reshape(-1)
        [v_next, f_opt, g_opt, Bopt, func_calls, grad_calls, warnflag] = OP.fmin_bfgs(obj_v, v0, fprime=obj_v_prime, args=(U, n_pos, n_neg, omega, 1.0 / (2 ** i), 1.0), disp=True, full_output=1)
        print('{} obj = {:0.3f}'.format(datetime.now(), f_opt))
        V = v_next.reshape(*V.shape)
        V_bfgs.append(V)

    np.savetxt(in_mat_file + '.' + suffix + '.c_vecs', U)
    np.savetxt(in_mat_file + '.' + suffix + '.v_vecs', V)
    print('Done.')


if __name__ == '__main__':
    cmd()
