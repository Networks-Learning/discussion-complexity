#!/usr/bin/env python
import cjr.models.dim as D
import click
import scipy.io as io


@click.command()
@click.argument('in_mat_file', type=click.Path(exists=True))
@click.argument('op_mat_file', type=click.Path())
@click.argument('op_loo_file', type=click.Path())
@click.option('-i', 'i_loo', help='LOO i', default=-1)
@click.option('-j', 'j_loo', help='LOO j', default=-1)
@click.option('--sat_2d', 'sat_2d', help='Is the matrix 2D-SAT w/o LOO?', default=False)
@click.option('--sat_1d', 'sat_1d', help='Is the matrix 1D-SAT w/o LOO?', default=False)
@click.option('--seed', 'seed', help='Seed which was used to create this test-cast.', default=-1)
def cmd(in_mat_file, op_mat_file, op_loo_file, i_loo, j_loo, sat_2d, sat_1d, seed):
    """Read M_partial from IN_MAT_FILE and fill in the matrix using Z3.
    The vote at (i, j) will be removed before filling in the matrix.
    The resulting matrix will be saved at OP_MAT_FILE and the given LOO entry
    will be placed along with the original vote in OP_LOO_FILE."""

    M_partial = io.loadmat(in_mat_file)['M_partial']

    # Leave i, j out.
    LOO = M_partial[i_loo, j_loo]
    M_partial[i_loo, j_loo] = 0

    voting_pats = D.sign_mat_to_voting_pats(M_partial)

    if sat_2d == 'unsat' or sat_2d == 'unknown':
        # Do not even attempt this.
        with open(in_mat_file + '.Z3.give-up', 'wt') as f:
            f.write('Not solved.')
        return
    elif sat_1d == 'sat':
        n_dim = 1
    else:
        n_dim = 2

    prob, c_vars, v_vars = D.create_z3_prob(n_dim=n_dim, voting_patterns=voting_pats)
    prob.set('timeout', 10 * 60 * 1000)  # Set a 10 minute timeout.

    res = prob.check()

    if str(res) == 'sat':

        c_vec, v_vec = D.make_vectors_from_z3_soln(prob=prob, c_vars=c_vars, v_vars=v_vars)
        Mhat = c_vec.dot(v_vec.T)

        io.savemat(op_mat_file, {'Mhat': Mhat})

        with open(op_loo_file, 'wt') as f:
            f.write('{}, {}'.format(LOO, Mhat[i_loo, j_loo]))

    elif str(res) == 'unknown':
        # Do not even attempt this.
        with open(op_mat_file + '.Z3.timeout', 'wt') as f:
            f.write('Timed-out.')
    elif str(res) == 'unsat':
        # Do not even attempt this.
        with open(op_mat_file + '.Z3.error', 'wt') as f:
            f.write('Problem was unsat?')


if __name__ == '__main__':
    cmd()
