import cvxpy as CVX
import decorated_options as Deco
import warnings
import numpy as np
from consts import COMMENTERS, COMMENTER, VOTERS, THETA


def make_thres_vars(N, M, K):
    """Create CVX model for the linear threshold model.

    N = number of commenters
    M = number of voters
    K = number of topics.
    """
    commenter_opinions = []
    for i in range(N):
        commenter_opinions.append([])
        for j in range(K):
            commenter_opinions[i].append(CVX.Variable(name='ComOp_%d_%d' % (i, j)))

    voter_opinions = []
    for i in range(M):
        voter_opinions.append([])
        for j in range(K):
            voter_opinions[i].append(CVX.Variable(name='VoOp_%d_%d' % (i, j)))

    theta = [CVX.Variable(name='theta_%d' % d) for d in range(K)]

    return {
        COMMENTERS: commenter_opinions,
        VOTERS: voter_opinions,
        THETA: theta
    }


def _upvote_cstr(x, v, thres):
    return [CVX.abs(x - v) <= thres]


# def _downvote_cstr(x, v, thres, b, M=100):
#     return [v - x >= thres - M * (1 - b),
#             x - v >= thres - M * b,
#             0 <= b, b <= 1]
def _downvote_cstr(x, v, thres):
    return [CVX.abs(x - v) > thres]


@Deco.optioned()
def make_constraints(topic_id_to_idx, commenter_id_to_idx, voter_id_to_idx,
                     df, model_vars, relax=True, fixed_thres=False):
    """Creates constraints and auxiliary variables for solving the
    problem with CVX.

    commenter_id_to_idx, voter_id_to_idx, topic_id_to_idx: Defines the ordering of model_vars.
    df: Dataframe with the complete dataset.
    model_vars: Model variables.
    relax: Whether to frame it as an integer linear program or regular linear program.
    fixed_thres: Determines whether all thresholds are the same or different.
    """
    # aux_vars = {}
    constraints = []

    # for voter in model_vars['voters']:
    #     for x in voter:
    #         constraints.extend([0 <= x, x <= 1])

    # for commenter in model_vars['voters']:
    #     for v in commenter:
    #         constraints.extend([0 <= v, v <= 1])

    # for thres in model_vars['theta']:
    #     constraints.extend([0 <= thres, thres <= 1])

    c_vars = model_vars['commenters']
    v_vars = model_vars['voters']
    theta = model_vars['theta']
    for trace_id, topic_id, parent_id, child_id, voter_id, parent_vote, child_vote \
            in df[['TraceId', 'ArticleTopic', 'ParentCommenterId',
                   'ChildCommenterId', 'VoterId', 'ParentVote', 'ChildVote']].values:
        topic_idx = topic_id_to_idx[topic_id]
        x_i = c_vars[commenter_id_to_idx[parent_id]][topic_idx]
        x_j = c_vars[commenter_id_to_idx[child_id]][topic_idx]
        v = v_vars[voter_id_to_idx[voter_id]][topic_idx]
        thres = theta[topic_idx]

        if parent_vote > 0:
            # Create convex constraints for upvotes on parent comment
            constraints.extend(_upvote_cstr(x_i, v, thres))
        else:
            # Create convex constraints for downvotes on child comment
            # if relax:
            #     b = CVX.Variable(name='Parent_%s_b' % trace_id)
            # else:
            #     b = CVX.Int(name='Parent_%s_b' % trace_id)
            # aux_vars[('Parent', trace_id)] = b

            # constraints.extend(_downvote_cstr(x_i, v, thres, b))
            constraints.extend(_downvote_cstr(x_i, v, thres))

        if child_vote > 0:
            constraints.extend(_upvote_cstr(x_j, v, thres))
        else:
            # Create convex constraints for downvotes on child comment
            # if relax:
            #     b = CVX.Variable(name='Child_%s_b' % trace_id)
            # else:
            #     b = CVX.Int(name='Child_%s_b' % trace_id)
            # aux_vars[('Child', trace_id)] = b

            # constraints.extend(_downvote_cstr(x_i, v, thres, b))
            constraints.extend(_downvote_cstr(x_i, v, thres))

    if fixed_thres:
        for k in range(1, len(topic_id_to_idx)):
            constraints.append(theta[k] == theta[k - 1])

    return constraints # , aux_vars


@Deco.optioned()
def make_downvote_objective(topic_id_to_idx, commenter_id_to_idx, voter_id_to_idx, df,
                            model_vars):
    """Create an objective function which aids the solution of the relaxed
    version of the problem.

    The objective is to maximize (x - v)^2 for all downvotes.
    """

    warnings.warn('This objective is will try to maximize a convex function.')

    obj = 0

    c_vars = model_vars['commenters']
    v_vars = model_vars['voters']
    for topic_id, p_id, c_id, v_id, p_v, c_v in \
            df[['ArticleTopic', 'ParentCommenterId', 'ChildCommenterId',
                'VoterId', 'ParentVote', 'ChildVote']].values:
        if p_v < 0:
            obj += CVX.square(c_vars[commenter_id_to_idx[p_id]][topic_id_to_idx[topic_id]] -
                              v_vars[voter_id_to_idx[v_id]][topic_id_to_idx[topic_id]])
        if c_v < 0:
            obj += CVX.square(c_vars[commenter_id_to_idx[c_id]][topic_id_to_idx[topic_id]] -
                              v_vars[voter_id_to_idx[v_id]][topic_id_to_idx[topic_id]])

    return CVX.Maximize(obj)


@Deco.optioned()
def make_satisfiable():
    """An objective function which just finds a satisfiable solution."""
    return CVX.Maximize(1)


@Deco.optioned()
def make_improvement(truth_df, topic_id_to_idx, commenter_id_to_idx,
                     noise_sigma, model_vars, seed):
    """Make an objective which says that the objective is to "correct" sentiment values."""
    Y = np.zeros((len(commenter_id_to_idx), len(topic_id_to_idx)), dtype=float)
    X = model_vars[COMMENTERS]
    rs = np.random.RandomState(seed=seed)
    obj = 0
    for c_id, opinion, topic in truth_df[truth_df.type == COMMENTER][['id', 'opinion', 'topic']].values:
        c_idx = commenter_id_to_idx[int(c_id)]
        t_idx = topic_id_to_idx[topic]
        Y[c_idx, t_idx] = opinion + rs.randn() * noise_sigma
        obj += CVX.square(X[c_idx][t_idx] - Y[c_idx, t_idx])

    # print(Y)
    return CVX.Minimize(obj)
