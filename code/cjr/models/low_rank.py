"""This module contains the implementation of the objective functions needed for:

Bhaskar, Sonia A., and Adel Javanmard. "1-bit matrix completion under exact low-rank constraint." Information Sciences and Systems (CISS), 2015 49th Annual Conference on. IEEE, 2015.

The implementation assumes the logit model.
"""

import numpy as np
import scipy.optimize as OP
from scipy.special import xlog1py
from datetime import datetime


# TODO: Define for probit model as well.
def f(x, sigma):
    return 1. / (1 + np.exp(-x / sigma))

def f_prime(x, sigma):
    return np.exp(-x / sigma) / (sigma * ((1 + np.exp(-x / sigma)) ** 2))


# TODO: The objective can be split for each row of U/V matrix, leading
# to faster optimization problems.
def obj_u(u, V, n_pos, n_neg, omega, reg_wt, alpha=1.0, sigma=1.0, verbose=False):
    U = u.reshape(-1, V.shape[1])
    M = U.dot(V.T)
    m = M[omega]
    # LL = np.sum(n_pos * (m - np.log(1 + np.exp(m))) - n_neg * np.log(1 + np.exp(m)))
    LL = np.sum(n_pos * m / sigma
                - xlog1py(n_pos + n_neg, np.exp(m / sigma)))

    M_sq = np.square(M / alpha)
    if np.any(M_sq > 1.0):
        reg = -np.inf
    else:
        reg = reg_wt * np.sum(np.log1p(-M_sq))

    if verbose:
        print('LL = {}, reg = {}'.format(LL, reg))
    return -(LL + reg)

def obj_u_prime(u, V, n_pos, n_neg, omega, reg_wt, alpha=1.0, sigma=1.0):
    U = u.reshape(-1, V.shape[1])
    M = U.dot(V.T)
    m = M[omega]
    grad_barrier_denom =  2 * reg_wt * M / ((alpha ** 2) * (1 - np.square(M / alpha)))

    # TODO: This can probably be made faster using matrix multiplication.
    grad = np.zeros_like(U)
    for i in range(M.shape[0]):
        for j in range(M.shape[1]):
            grad[i, :] += grad_barrier_denom[i, j] * V[j, :]

    # Assumes the logist model.
    grad_m = -(n_pos * (1 - f(m, sigma)) - n_neg * f(m, sigma)) / sigma
    for idx, (i, j) in enumerate(zip(*omega)):
        grad[i, :] +=  grad_m[idx] * V[j, :]
    return grad.reshape(-1)


def obj_v(v, U, n_pos, n_neg, omega, reg_wt, alpha=1.0, sigma=1.0):
    V = v.reshape(-1, U.shape[1])
    u = U.reshape(-1)
    return obj_u(u, V, n_pos, n_neg, omega, reg_wt, alpha=alpha, sigma=sigma)


def obj_v_prime(v, U, n_pos, n_neg, omega, reg_wt, alpha=1.0, sigma=1.0):
    V = v.reshape(-1, U.shape[1])
    M = U.dot(V.T)
    m = M[omega]

    grad_barrier_denom = 2 * reg_wt * M / ((alpha ** 2) * (1 - np.square(M / alpha)))

    grad = np.zeros_like(V)
    for i in range(M.shape[0]):
        for j in range(M.shape[1]):
            grad[j, :] += grad_barrier_denom[i, j] * U[i, :]

    # Assumes the logistic noise model.
    grad_m = -(n_pos * (1 - f(m, sigma)) - n_neg * f(m, sigma)) / sigma
    for idx, (i, j) in enumerate(zip(*omega)):
        grad[j, :] += grad_m[idx] * U[i, :]
    return grad.reshape(-1)


def make_low_rank_params(M_orig):
    """Returns the arguments which the objective functions need."""
    num_comments, num_voters = M_orig.shape

    M_pos = np.zeros((num_comments, num_voters))
    M_neg = np.zeros((num_comments, num_voters))

    M_pos[M_orig > 0] = 1.0
    M_neg[M_orig < 0] = 1.0

    omega = (M_pos + M_neg).nonzero()
    n_neg = M_neg[omega]
    n_pos = M_pos[omega]

    return {
        'omega': omega,
        'n_neg': n_neg,
        'n_pos': n_pos
    }


def optimize_low_rank(U_init, V_init, n_pos, n_neg, omega,
                      reg_wt, alpha, sigma,
                      verbose=False):
    """Run one iteration of the alternating optimization."""

    u0 = U_init.reshape(-1)
    [u_next, f_opt, g_opt, Bopt, func_calls, grad_calls, warnflag] = OP.fmin_bfgs(obj_u, u0, fprime=obj_u_prime, args=(V_init, n_pos, n_neg, omega, reg_wt, alpha, sigma), disp=False, full_output=1)
    if verbose:
        print('{} obj = {:0.3f}'.format(datetime.now(), f_opt))
    U = u_next.reshape(*U_init.shape)

    v0 = V_init.reshape(-1)
    [v_next, f_opt, g_opt, Bopt, func_calls, grad_calls, warnflag] = OP.fmin_bfgs(obj_v, v0, fprime=obj_v_prime, args=(U, n_pos, n_neg, omega, reg_wt, alpha, sigma), disp=False, full_output=1)
    if verbose:
        print('{} obj = {:0.3f}'.format(datetime.now(), f_opt))
    V = v_next.reshape(*V_init.shape)

    return {
        'U': U,
        'V': V
    }


def optimize_low_rank_lbfgs(U_init, V_init, n_pos, n_neg, omega,
                            reg_wt, alpha, sigma,
                            verbose=False):
    """Run one iteration of the alternating optimization."""

    u0 = U_init.reshape(-1)
    [u_next, f_opt, d] = OP.fmin_l_bfgs_b(obj_u, u0, fprime=obj_u_prime, args=(V_init, n_pos, n_neg, omega, reg_wt, alpha, sigma), disp=verbose)
    if verbose:
        print('{} obj = {:0.3f}'.format(datetime.now(), f_opt))
    U = u_next.reshape(*U_init.shape)

    v0 = V_init.reshape(-1)
    [v_next, f_opt, d] = OP.fmin_l_bfgs_b(obj_v, v0, fprime=obj_v_prime, args=(U, n_pos, n_neg, omega, reg_wt, alpha, sigma), disp=verbose)
    if verbose:
        print('{} obj = {:0.3f}'.format(datetime.now(), f_opt))
    V = v_next.reshape(*V_init.shape)

    return {
        'U': U,
        'V': V
    }


def verify_low_rank_grads(U_init, V_init, reg_wt=1.0, alpha=1.0, sigma=1.0):
    """Verify that the gradients are correct. If the numbers returned are
    *small*, the gradients are correct."""

    M = np.random.randn(U_init.shape[0], V_init.shape[0])
    params = make_low_rank_params(M)
    omega, n_neg, n_pos = params['omega'], params['n_neg'], params['n_pos']

    return {
        'u_check': OP.check_grad(obj_u, obj_u_prime, U_init.reshape(-1),
                                 V_init, n_pos, n_neg, omega, reg_wt, alpha, sigma),
        'v_check': OP.check_grad(obj_v, obj_v_prime, V_init.reshape(-1),
                                 U_init, n_pos, n_neg, omega, reg_wt, alpha, sigma)
    }


