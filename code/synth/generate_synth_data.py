#!/usr/bin/env python
import click
import pandas as pd
import numpy as np
import seqfile
import os
import decorated_options as Deco

COS_SIM = 'cossim'
THRES_FIXED = 'thres_fixed'
THRES_RAND = 'thres_rand'

VOTER = 'voter'
COMMENTER = 'commenter'

# Output headers:
#  - TraceId
#  - VoterId
#  - ParentCommentId   , ChildCommentId
#  - ParentVote        , ChildVote
#  - ArticleId         , ArticleTopic
#  - ParentCommenterId , ChildCommenterId
#  - ParentCommentTime , ChildCommentTime
#  - ParentVoteTime    , ChildVoteTime

DEF_OPTS = Deco.Options(
    N=10,
    M=100,
    A=100,
    K=1,
    P=1000,
    verbose=True,
    voting=COS_SIM,
    force=False,
    seed=42
)


def choose_idx_item(random_state, arr):
    """Returns a random item and an index."""
    idx = random_state.randint(low=0, high=len(arr))
    return idx, arr[idx]


@click.command()
@click.argument('output_path')
@click.argument('truth_path')
@click.option('-N', 'N', default=DEF_OPTS.N, type=int, help='Number of commenters.')
@click.option('-M', 'M', default=DEF_OPTS.M, type=int, help='Number of voters.')
@click.option('-A', 'A', default=DEF_OPTS.A, type=int, help='Number of articles.')
@click.option('-K', 'K', default=DEF_OPTS.K, type=int, help='Number of topics.')
@click.option('-P', 'P', default=DEF_OPTS.P, type=int, help='Number of patterns to generate.')
@click.option('--verbose/--no-verbose', default=DEF_OPTS.verbose, help='Verbose output.')
@click.option('--force/--no-force', default=DEF_OPTS.force, help='Force overwrite if the output file exists')
@click.option('--voting', default=DEF_OPTS.voting, help='Mechanism for voting', type=click.Choice([COS_SIM, THRES_RAND, THRES_FIXED]))
@click.option('--seed', default=DEF_OPTS.seed, help='Random seed.')
def cmd(output_path, truth_path, N, M, P, A, K, verbose, force, voting, seed):
    """Generate synthetic data and put it in OUTPUT_PATH while putting ground
    truth opinions in TRUTH_PATH."""
    df, truth = run(N=N, M=M, P=P, A=A, K=K, verbose=verbose, seed=seed,
                    voting=voting)
    save_data(df=df, truth=truth,
              truth_path=truth_path, output_path=output_path,
              verbose=verbose, force=force)


def save_data(df, truth, output_path, truth_path, verbose, force):
    """Save data to the OUTPUT_PATH."""
    if os.path.exists(output_path) or os.path.exists(truth_path):
        if not force:
            if verbose:
                print('Not overwriting "%s"/"%s" as --force is not passed.'
                      % (output_path, truth_path))
            output_path = seqfile.findNextFile(prefix=output_path, base=1)
            truth_path = seqfile.findNextFile(prefix=truth_path, base=1)

            if verbose:
                print('Instead, writing to "%s"/"%s"' %
                      (output_path, truth_path))
        else:
            if verbose:
                print('Overwriting "%s"/"%s"' % (output_path, truth_path))
    else:
        if verbose:
            print('Writing output to "%s"/"%s"' % (output_path, truth_path))

    df.to_csv(output_path, index=False)
    truth.to_csv(output_path, index=False)


@Deco.optioned()
def run(N, M, A, K, P, verbose, seed, voting):
    """Generate synthetic data."""

    # Set seed
    RS = np.random.RandomState(seed=seed)

    # Adding some values to them to avoid confusing the index with the ids.
    commenter_ids = np.arange(N) + 100
    article_ids = np.arange(A) + 1000
    voter_ids = np.arange(M) + 10000
    topic_ids = np.arange(K) + 100000

    # Assign topics to the articles
    map_article_to_topic = {a: choose_idx_item(RS, topic_ids)
                            for a in article_ids}

    # Generate random opinions
    if voting == COS_SIM:
        commenter_opinions = (RS.rand(N, K) - 0.5) * 2
        voter_opinions = (RS.rand(M, K) - 0.5) * 2
    elif voting == THRES_FIXED:
        commenter_opinions = RS.rand(N, K)
        voter_opinions = RS.rand(M, K)
        thres = RS.rand(1) / 2
    elif voting == THRES_RAND:
        commenter_opinions = RS.rand(N, K)
        voter_opinions = RS.rand(M, K)
        thres = RS.rand(M) / 2
    else:
        raise ValueError('Invalid voting system: %s' % voting)

    # Generate comments
    data = []
    traceId = 0

    while P > 0:
        P -= 1
        traceId += 1
        voter_idx, voterId = choose_idx_item(RS, voter_ids)
        article_idx, articleId = choose_idx_item(RS, article_ids)
        topic_idx, articleTopic = map_article_to_topic[articleId]
        voter_opinion = voter_opinions[voter_idx, topic_idx]
        parent_commenter_idx, parentCommenterId = choose_idx_item(RS, commenter_ids)
        # Child and the parent commenter should not be the same.
        while True:
            child_commenter_idx, childCommenterId = \
                choose_idx_item(RS, commenter_ids)
            if child_commenter_idx != parent_commenter_idx:
                break

        parentCommentId = 2 * P
        childCommentId = 2 * P + 1

        parent_opinion = commenter_opinions[parent_commenter_idx, topic_idx]
        child_opinion = commenter_opinions[child_commenter_idx, topic_idx]

        if voting == 'cossim':
            parentVote = -1 if np.cos((voter_opinion - parent_opinion) * np.pi) < 0 else 1
            childVote = -1 if np.cos((voter_opinion - child_opinion) * np.pi) < 0 else 1
        elif voting.startswith('thres_'):
            if thres.shape[0] > 1:
                voter_thres = thres[voter_idx]
            else:
                voter_thres = thres

            parentVote = -1 if np.abs(voter_opinion - parent_opinion) > voter_thres else 1
            childVote = -1 if np.abs(voter_opinion - child_opinion) > voter_thres else 1

        parentCommentTime = P
        childCommentTime = P + 1
        parentVoteTime, childVoteTime = P + 2, P + 3

        data.append({
            'TraceId': traceId,
            'VoterId': voterId,
            'ParentCommentId': parentCommentId,
            'ParentVote': parentVote,
            'ArticleId': articleId,
            'ParentCommenterId': parentCommenterId,
            'ParentCommentTime': parentCommentTime,
            'ParentVoteTime': parentVoteTime,
            'ChildVoteTime': childVoteTime,
            'ChildCommentId': childCommentId,
            'ChildVote': childVote,
            'ArticleTopic': articleTopic,
            'ChildCommenterId': childCommenterId,
            'ChildCommentTime': childCommentTime,
            'Disagree': parentVote != childVote
        })

    def get_thres(voter_idx):
        if voting == COS_SIM:
            return -1
        elif voting == THRES_FIXED:
            return thres
        elif voting == THRES_RAND:
            return thres[voter_idx]
        else:
            raise ValueError('Invalid user_idx: %s' % voter_idx)

    df = pd.DataFrame.from_dict(data)
    truth = pd.DataFrame.from_dict(
        [{'type': COMMENTER,
          'id': Id,
          'opinion': commenter_opinions[commenter_idx, t_idx],
          'thres': -1,
          'topic': topicId}
         for commenter_idx, Id in enumerate(commenter_ids)
         for t_idx, topicId in enumerate(topic_ids)] +
        [{'type': VOTER,
          'id': Id,
          'opinion': voter_opinions[v_idx, t_idx],
          'thres': get_thres(v_idx),
          'topic': topicId}
         for v_idx, Id in enumerate(voter_ids)
         for t_idx, topicId in enumerate(topic_ids)]
    )

    return df, truth


if __name__ == '__main__':
    cmd()
