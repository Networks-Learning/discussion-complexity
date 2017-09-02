import z3
import numpy as np
import networkx as nx
import scipy as sp


def to_bin(n, length):
    """Convert a number into a vector of booleans corresponding to voting
    pattern of the given length."""
    s = bin(n)[2:]
    s = '0' * (length - len(s)) + s
    return [x == '1' for x in s]


def to_pattern(vote_str):
    """Convert votes expressed as a string to vector of booleans."""
    return [True if v == '+' else False if v == '-' else None
            for v in vote_str]


def to_str(pattern):
    """Converts a True/None/False array into a string."""
    return ''.join(['+' if p is True else '-' if p is False else 'o'
                    for p in pattern])


def mk_vars(num_dims, num_comments, num_voters):
    """Creates and returns variables for the comments and the voters."""
    com_vars = [[z3.Real('x_%d_%d' % (ii, jj))
                 for jj in range(num_dims)]
                for ii in range(num_comments)]

    voter_vars = [[z3.Real('v_%d_%d' % (ii, jj))
                   for jj in range(num_dims)]
                  for ii in range(num_voters)]

    return com_vars, voter_vars


def add_constr(voter_var, com_var, is_upvote):
    """Creates constraints for a given vote."""
    s = sum(x * v for x, v in zip(voter_var, com_var))
    return s > 0 if is_upvote else s < 0


def create_prob(n_dim, voting_patterns):
    """Given the (unique) votes, create a problem for Z3 to solve."""
    n_voters = len(voting_patterns)
    n_comments = len(voting_patterns[0])

    if isinstance(voting_patterns[0], str):
        voting_patterns = [to_pattern(x) for x in voting_patterns]

    com_vars, voter_vars = mk_vars(n_dim, n_comments, n_voters)
    s = z3.Solver()

    for v_idx, pat in enumerate(voting_patterns):
        for c_idx, vote in enumerate(pat):
            if vote is not None:
                s.add(add_constr(voter_vars[v_idx], com_vars[c_idx], vote))

    for c_idx in range(n_comments):
        s.add(sum([x * x for x in com_vars[c_idx]]) == 1)

    for v_idx in range(n_voters):
        s.add(sum([v * v for v in voter_vars[v_idx]]) == 1)

    return s, com_vars, voter_vars


def gen_voting_patterns(n_comments, n_votes, p_up, p_down, seed=360):
    """Generate random voting patterns with the given probability of
    up/down/no-votes. Note that the generated voting patterns are not forced to
    be unique.
    """
    rs = np.random.RandomState(seed=seed)
    return [to_str(rs.choice([True, False, None], size=n_comments,
                             p=[p_up, p_down, 1 - p_up - p_down]))
            for ii in range(n_votes)]


def opinions_differ(vote_1, vote_2):
    """Returns if the votes differ."""
    return vote_1 != vote_2 if vote_1 != 'o' and vote_2 != 'o' else False


def create_disgreement_graph(num_comments, rand_votes):
    """Create a graph of disagreements among comments as evinced by voting
    patterns."""
    vote_pats = set(rand_votes)
    edges = [(i, j)
             for pat in vote_pats
             for i in range(num_comments)
             for j in range(i + 1, num_comments)
             if opinions_differ(pat[i], pat[j])]
    return nx.Graph(list(set(edges)))


def generate_voting_matrix(num_comments, num_voters, dim, seed=100, voting_probs=None):
    """Creates voting patterns for users/voters.
    dim: The dimensionality of the opinion space.
    voting_probs: should sum up to 1 and gives the probability of picking each
    comment for voting. Default is Uniform.

    Returns a sparse matrix in LIL format.
    """
    RS = np.random.RandomState(seed)

    # Normalized latent representations.
    # TODO: These are not evenly distributed on a circle, the proper way to do
    # it would be to sample theta_1, theta_2, independently and then use
    # spherical coordinates?
    comment_vecs = RS.rand(num_comments, dim) - 0.5
    comment_vecs = comment_vecs / np.sqrt(np.square(comment_vecs).sum(axis=1))[:, np.newaxis]

    voter_vecs = RS.rand(num_voters, dim) - 0.5
    voter_vecs = voter_vecs / np.sqrt(np.square(voter_vecs).sum(axis=1))[:, np.newaxis]

    if voting_probs is None:
        # Assume that each voter votes on average 5 posts.
        voting_probs = [(1 / num_comments) * 5] * num_comments

    voting_probs = np.asarray(voting_probs)

    M = sp.sparse.lil_matrix((num_comments, num_voters), dtype=np.int8)
    for v_idx in range(num_voters):
        voted_on = np.nonzero(RS.rand(num_comments) < voting_probs)[0]
        voter_vec = voter_vecs[v_idx]

        for c_idx in voted_on:
            vote = comment_vecs[c_idx].dot(voter_vec)
            M[c_idx, v_idx] = np.sign(vote)

    return M, comment_vecs, voter_vecs


def similarity_matrix(comment_vecs, voter_vecs):
    """Creates a matrix with the similarity value between each pair of voter
    and commenter."""
    return comment_vecs.dot(voter_vecs.T)


def get_mat_R2(pred, truth):
    """Calculates the R^2 for the prediction matrix."""
    SS_res = np.square(pred - truth).sum()
    SS_tot = np.square(truth - truth.mean()).sum()
    return 1 - (SS_res / SS_tot)
