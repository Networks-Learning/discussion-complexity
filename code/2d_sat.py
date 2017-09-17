#!/usr/bin/env python
import cjr.models.dim as D
import z3
import click
import pandas as pd
import multiprocessing as MP

@click.command()
@click.argument('in_file', type=click.Path(exists=True))
@click.option('--cpus', help='How many CPUs to use.', type=int, default=-1)
def cmd(in_file, cpus):
    """Reads data from IN_FILE with the following format:

            comment_tree_id, commenter_id, voter_id, vote_type
            1, 200, 3000, 1
            1, 201, 3000, -1
            ...

    Outputs a csv which contains whether the comment-tree had a sign-rank of 2 or not.
    """
    df = pd.read_csv(in_file)
    comment_tree_ids = df.comment_tree_id.unique()

    if cpus == -1:
        cpus = None

    # Hack to make 'df' available in the environment of _worker without passing
    # it via pickling.
    global _worker
    def _worker(comment_tree_idx):
        comment_tree_id = comment_tree_ids[comment_tree_idx]
        M = D.make_M_from_df(df=df, comment_tree_id=comment_tree_id)
        voting_pats = D.sign_mat_to_voting_pats(M)

        ctx = z3.Context()
        prob, _, _ = D.create_z3_prob(ctx=ctx, n_dim=2,
                                      voting_patterns=voting_pats)
        return {'comment_tree_id': comment_tree_id,
                '2D_sat': prob.check().r == 1}

    with MP.Pool(processes=cpus) as pool:
        data = pool.map(_worker, range(len(comment_tree_ids)))

    out_df = pd.DataFrame.from_dict(data)
    print(out_df.to_csv(index=False))

if __name__ == '__main__':
    cmd()
