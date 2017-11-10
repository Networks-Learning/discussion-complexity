#!/usr/bin/env python
import cjr.models.dim as D
import z3
import click
import pandas as pd
import multiprocessing as MP

def make_canonical_df(csv_df):
    """Turn the csv df into a df with columns:

    'comment_tree_id', 'comment_id', 'voter_id', 'vote_type'.
    """
    canon_df = csv_df.rename(columns={'r.reply_to': 'comment_tree_id',
                                      'r.message_id': 'comment_id',
                                      'r.uid_alias': 'voter_id',
                                      'r.vote_type': 'vote_type'})

    canon_df['vote_type'].replace(['UP', 'DOWN'], [1, -1], inplace=True)
    canon_df['comment_tree_id'] = (canon_df['comment_tree_id']
                                   .where(~canon_df['comment_tree_id'].isnull(),
                                          canon_df['comment_id']))
    canon_df = canon_df[~canon_df['vote_type'].isnull()]
    return canon_df


@click.command()
@click.argument('in_file', type=click.Path(exists=True))
@click.option('--cpus', help='How many CPUs to use.', type=int, default=-1)
@click.option('--timeout', help='Time after which to give up (ms).', type=int, default=10 * 1000)
@click.option('--real/--no-real', help='Assume format of real-data.', default=False)
def cmd(in_file, cpus, timeout, real):
    """Reads data from IN_FILE with the following format:

       \b
       comment_tree_id, commenter_id, voter_id, vote_type
       1, 200, 3000, 1
       1, 201, 3000, -1
       ...

       or (in case of real data):

       \b
       r.reply_to\t r.message_id\t r.uid_alias\t r.vote_type
       1\t 200\t 3000\t UP
       1\t 201\t 3000\t DOWN
       ...

    Outputs a CSV which contains whether the comment-tree had a sign-rank of 2 or not.

    Redirect output to a file to save it.
    """

    if real:
        df = pd.read_csv(in_file, sep='\t')
        df = make_canonical_df(df)
    else:
        df = pd.read_csv(in_file)

    comment_tree_ids = df.comment_tree_id.dropna().unique()

    if cpus == -1:
        cpus = None

    # Hack to make 'df' available in the environment of _worker without passing
    # it via pickling.
    global _worker
    def _worker(comment_tree_idx):
        comment_tree_id = comment_tree_ids[comment_tree_idx]
        M = D.make_M_from_df(df=df, comment_tree_id=comment_tree_id)
        voting_pats = D.sign_mat_to_voting_pats(M)

        unique_voting_pats = list(set(tuple(x) for x in voting_pats))

        ctx = z3.Context()
        prob, _, _ = D.create_z3_prob(ctx=ctx, n_dim=2,
                                      voting_patterns=unique_voting_pats)
        prob.set('timeout', timeout)
        res = prob.check()
        return {'comment_tree_id': comment_tree_id,
                '2D_sat': str(res)}

    with MP.Pool(processes=cpus) as pool:
        data = pool.map(_worker, range(len(comment_tree_ids)))

    out_df = pd.DataFrame.from_dict(data)
    print(out_df.to_csv(index=False))

if __name__ == '__main__':
    cmd()
