#!/usr/bin/env python
import cjr.models.dim as D
import cjr.utils.utils as U
import z3
import click
import pandas as pd
import multiprocessing as MP


@click.command()
@click.argument('in_file', type=click.Path(exists=True))
@click.option('--dims', 'n_dims', help='What dimensional embedding to test for?', type=int, default=2)
@click.option('--cpus', help='How many CPUs to use.', type=int, default=-1)
@click.option('--timeout', help='Time after which to give up (ms).', type=int, default=10 * 1000)
@click.option('--real/--no-real', help='Assume format of real-data.', default=False)
@click.option('--improve', default=None, help='Improve the results from the provided file. Will only run for `unknown` ids in the file.', type=click.Path(exists=True))
@click.option('--context-id/--no-context-id', 'use_contextId', default=False, help='Use the context_id instead of comment_tree_id to group comments into matrices.')
@click.option('--nrows', default=-1, help='Number of rows from the CSV to read.')
def cmd(in_file, n_dims, cpus, timeout, real, improve, use_contextId, nrows):
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

       of (in case of merged data):

       \b
       comment_id,voter_id,comment_tree_id,vote_type,r.abuse_vote,m.uid_alias,m.created_at,context_id,message_id,namespace,description,url,comment,lang
       0,r0,U1,r0,1.0,,u217409,1502981871,UID1,r0,yahoo_content,{()},{()},Que lindo!!!!!,pt
       1,r0,U2,r0,1.0,,u217409,1502981871,UID2,r0,yahoo_content,{()},{()},Que lindo!!!!!,pt
       2,r0,U3,r0,1.0,,u217409,1502981871,UID3,r0,yahoo_content,{()},{()},Que lindo!!!!!,pt
       ...

    Outputs a CSV which contains whether the comment/article-tree had a sign-rank of 2 or not.

    Redirect output to a file to save it.
    """

    if nrows < 0:
        nrows = None

    if real:
        df = pd.read_csv(in_file, sep='\t', nrows=nrows)
        df = U.make_canonical_df(df)
        df = U.remove_dups(df)
    else:
        # Works for both merged data as well as synthetic data.
        df = pd.read_csv(in_file, nrows=nrows)

    key = 'context_id' if use_contextId else 'comment_tree_id'

    if improve is not None:
        old_results = pd.read_csv(improve)
        key_ids = old_results[key][old_results['{}D_sat'.format(n_dims)] == 'unknown'].values
    else:
        key_ids = df[key].dropna().unique()

    if cpus == -1:
        cpus = None

    # Hack to make 'df' available in the environment of _worker without passing
    # it via pickling.
    global _worker
    def _worker(key_idx):
        k_id = key_ids[key_idx]
        M = D.make_M_from_df_generic(df=df, key_name=key, key_value=k_id)
        voting_pats = D.sign_mat_to_voting_pats(M)

        unique_voting_pats = list(set(tuple(x) for x in voting_pats))

        ctx = z3.Context()
        prob, _, _ = D.create_z3_prob(ctx=ctx, n_dim=n_dims,
                                      voting_patterns=unique_voting_pats)
        prob.set('timeout', timeout)
        res = prob.check()
        return {key: k_id,
                '{}D_sat'.format(n_dims): str(res)}

    with MP.Pool(processes=cpus) as pool:
        data = pool.map(_worker, range(len(key_ids)))

    out_df = pd.DataFrame.from_dict(data)
    print(out_df.to_csv(index=False))


if __name__ == '__main__':
    cmd()
