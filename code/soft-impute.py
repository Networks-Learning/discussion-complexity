#!/usr/bin/env python
"""Implements the Exact low rank matrix filling as discussed in:

Spectral Regularization Algorithms for Learning Large Incomplete Matrices by Mazumder et. al.
"""

import click
import scipy.io as io
import numpy as np
from datetime import datetime
from fancyimpute import SoftImpute


@click.command()
@click.argument('in_mat_file', type=click.Path(exists=True))
@click.option('--dims', 'dims', help='The dimensionality of the embedding.', default=2)
@click.option('--seed', 'seed', help='The random seed to use for initializing matrices, in case initial values are not given.', default=42)
@click.option('--suffix', 'suffix', help='Suffix to add before saving the embeddings.', default='svd')
@click.option('-i', 'i_loo', help='Which i index to LOO.', default=-1)
@click.option('-j', 'j_loo', help='Which j index to LOO.', default=-1)
@click.option('--loo-output', 'loo_output', help='Where to save the LOO output.', default=None)
@click.option('--loo-only/--no-loo-only', 'loo_only', help='Whether to only save the LOO output or whether to save the complete recovered matrix.', default=False)
@click.option('--verbose/--no-verbose', 'verbose', help='Verbose output.', default=False)
def cmd(in_mat_file, dims, suffix, i_loo, j_loo, loo_output, loo_only, verbose, seed):
    """Read M_partial from IN_MAT_FILE and complete the matrix using soft-impute method."""

    M = io.loadmat(in_mat_file)['M_partial']
    rank = dims

    LOO_mode = False
    if i_loo > -1 and j_loo > -1:
        LOO = M[i_loo, j_loo]
        M[i_loo, j_loo] = 0
        LOO_mode = True

    num_comments, num_voters = M.shape

    M[M == 0] = np.nan
    M_complete = SoftImpute(max_rank=dims).complete(M)

    if LOO_mode:
        file_tmpl = f'{in_mat_file}.r{rank}.s{seed}.i{i_loo}.j{j_loo}.soft-impute.out'

        if not loo_only:
            op_mat_file = file_tmpl + '.mat'
            io.savemat(op_mat_file, {'Mhat': M_complete})

        op_loo_file = loo_output if loo_output is not None else file_tmpl + '.loo'
        loo_pred = M_complete[i_loo, j_loo]
        with open(op_loo_file, 'wt') as f:
            f.write('{}, {}'.format(LOO, loo_pred))
    else:
        raise NotImplementedError('Use randomized_svd here.')
        # np.savetxt(in_mat_file + '.' + suffix + '.c_vecs', U)
        # np.savetxt(in_mat_file + '.' + suffix + '.v_vecs', V)

    print('Done at', datetime.now())


if __name__ == '__main__':
    cmd()
