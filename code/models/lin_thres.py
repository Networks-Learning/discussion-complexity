import cvxpy as CVX

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
        commenter_opinions.append([])
        for j in range(K):
            voter_opinions[i].append(CVX.Variable(name='VoOp_%d_%d' % (i, j)))

    theta = CVX.Variable(rows=K, name='theta')

    return {
        'commenters': commenter_opinions,
        'voters': voter_opinions,
        'theta': theta
    }

def make_constraints(commenter_ids, voter_ids, df, model_vars):
    pass

