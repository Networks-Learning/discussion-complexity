#!/usr/bin/env python
import cjr.models.dim as D
import click
import scipy.io as io
import numpy as np


@click.command()
@click.argument('in_mat_file', type=click.Path(exists=True))
@click.argument('op_mat_file', type=click.Path())
@click.argument('op_loo_file', type=click.Path())
@click.argument('op_SC_file', 'op_SC_file', type=click.Path())
@click.option('-i', 'i_loo', help='LOO i', default=-1)
@click.option('-j', 'j_loo', help='LOO j', default=-1)
@click.option('--seed', 'seed', help='Seed which was used to create this test-cast.', default=-1)
@click.option('--min-avg/--no-min-avg', 'min_avg', help='This flag will cause minimization of the average SC instead of worst case SC. Is much faster.', default=False)
def cmd(in_mat_file, op_mat_file, op_loo_file, op_SC_file, i_loo, j_loo, seed, min_avg):
    """Read M_partial from IN_MAT_FILE and fill in the matrix using SR.
    The vote at (i, j) will be removed before filling in the matrix.
    The resulting matrix will be saved at OP_MAT_FILE and the given LOO entry
    will be placed along with the original vote in OP_LOO_FILE.

    Additionally, the best guess for the Sign Rank will be placed in OP_SC_FILE along with the source node which results in that.
    """

    M_partial = io.loadmat(in_mat_file)['M_partial']

    # Leave i, j out.
    LOO = M_partial[i_loo, j_loo]
    M_partial[i_loo, j_loo] = 0

    spanning_tree, eq_sets, eq_signs = D.make_spanning_tree(sign_mat=M_partial, verbose=True, min_avg=min_avg)
    Mhat = D.get_M_full(M_partial, equiv_sets=eq_sets, equiv_signs=eq_signs)

    source_opt, SC_opt = 0, np.inf
    for source in range(Mhat.shape[0]):
        permut = D.make_one_permut(spanning_tree=spanning_tree, source=source)
        SC_, _ = D.SC(Mhat, permut)
        if SC_ < SC_opt:
            source_opt, SC_opt = source, SC_

    io.savemat(op_mat_file, {'Mhat': Mhat})

    with open(op_loo_file, 'wt') as f:
        f.write('{}, {}'.format(LOO, Mhat[i_loo, j_loo]))

    with open(op_SC_file, 'wt') as f:
        f.write('{}, {}'.format(source_opt, SC_opt))


if __name__ == '__main__':
    cmd()
