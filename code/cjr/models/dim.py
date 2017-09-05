import z3
import numpy as np
import networkx as nx
import networkx as nx
import scipy as sp
from UnionFind import UnionFind
from collections import defaultdict
import heapq as PQ


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


def generate_voting_matrix(num_comments, num_voters, dim, seed=100,
                           voting_probs=None, avg_comments_per_voter=5):
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
        # Assume that each voter votes on average `avg_comments_per_voter` posts.
        voting_probs = [avg_comments_per_voter / num_comments] * num_comments

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


# ======
# This is the O(n / log n) implementation check.

def same_sign(eq_signs, eq_sets, i, j, col):
    """Returns true if the rows i and j have the same sign on column col.
    Returns false if any of the signs are indeterminate."""
    return eq_signs.get(eq_sets[i, col], 0) * eq_signs.get(eq_sets[j, col], 0) == 1


def diff_signs(eq_signs, eq_sets, i, j, col):
    """Returns true if the rows i and j have different signs on column col.
    Returns false if any of the signs are indeterminate."""
    return eq_signs.get(eq_sets[i, col], 0) * eq_signs.get(eq_sets[j, col], 0) == -1


def mergeable(eq_signs, eq_sets, i, j, col):
    """Determines whether the equivalence classes corresponding to (i, col) and
    (j, col) can be merged."""
    return ((eq_sets[i, col] not in eq_signs) or
            (eq_sets[j, col] not in eq_signs) or
            (eq_signs[eq_sets[i, col]] == eq_signs[eq_sets[j, col]]))


def weight(i, j, P, eq_signs, eq_sets_uf):
    """Calculate the weight of edge (i, j) in sparse matrix M using probabilities vector P."""
    return sum(P[idx]
               for idx in range(len(P))
               if diff_signs(eq_signs, eq_sets_uf, i, j, idx))


def tiebreaker(i, j, len_P, eq_signs, eq_sets_uf):
    """Calculates how many entries row i and j have in common to help in breaking ties between them."""
    return -sum(1 for idx in range(len_P)
                if same_sign(eq_signs, eq_sets_uf, i, j, idx))


def _worker_spanning_tree(params):
    """Worker which does the spanning tree work."""
    ii, jj, probs, eq_sets_uf, eq_signs = params
    return (weight(i=ii, j=jj, P=probs, eq_sets_uf=eq_sets_uf, eq_signs=eq_signs),
            tiebreaker(i=ii, j=jj, len_P=len(probs), eq_sets_uf=eq_sets_uf, eq_signs=eq_signs),
            (ii, jj))


def make_spanning_tree(sign_mat, min_avg=False, pool=None):
    """Create a spanning tree."""
    uf = UnionFind()

    N = sign_mat.shape[0]
    for v in range(N):
        uf.union(v)

    equiv_sets = UnionFind()
    equiv_signs = {(i, j): sign_mat[i, j] for i, j in zip(*sign_mat.nonzero())}

    forest = defaultdict(lambda: set())
    probs = np.ones(sign_mat.shape[1]) / sign_mat.shape[1]

    params = [(ii, jj, probs, equiv_sets, equiv_signs)
              for ii in range(N) for jj in range(ii + 1, N)]

    if pool is None:
        edges_heap = [_worker_spanning_tree(x) for x in params]
    else:
        edges_heap = pool.map(_worker_spanning_tree, params)

    PQ.heapify(edges_heap)

    num_edges = 0
    while num_edges < N - 1:
        x, _, (i, j) = PQ.heappop(edges_heap)
        assert i < j

        if uf[i] != uf[j]:
            # Otherwise, this edge would have created a cycle
            edges_forest_i = forest[uf[i]]
            edges_forest_j = forest[uf[j]]

            new_root = uf.union(i, j)
            forest[new_root] = edges_forest_i.union(edges_forest_j)
            forest[new_root].add((i, j))
            num_edges += 1

            for col in range(sign_mat.shape[1]):
                if mergeable(equiv_signs, equiv_sets, i, j, col):
                    # If either of these is undecided, merge an equivalent sets.
                    old_i_root = equiv_sets[i, col]
                    old_j_root = equiv_sets[j, col]

                    root = equiv_sets.union((i, col), (j, col))

                    if old_i_root in equiv_signs:
                        equiv_signs[root] = equiv_signs[old_i_root]
                    elif old_j_root in equiv_signs:
                        equiv_signs[root] = equiv_signs[old_j_root]

            # if x > 0: # Merely an optimization
            if not min_avg:
                # Minimize the worst case.
                probs = [(2 * p / (1 + x))
                             if diff_signs(equiv_signs, equiv_sets, i, j, idx)
                             else p / (1 + x)
                         for idx, p in enumerate(probs)]
            else:
                # Minimize the average case.
                probs = [1
                         if diff_signs(equiv_signs, equiv_sets, i, j, idx)
                         else 0
                         for idx, p in enumerate(probs)]

            # This can probably be optimized further because update of the weights in
            # this manner was required for the proof.
            # It should be possible to reduce it since we are only doubling
            # some entries in probs since the division does not change the order of
            # any other entries in the matrices.
            #
            # As it stands, this is an O(M^2 * N) operation.
            params = [
                (ii, jj, probs, equiv_sets, equiv_signs)
                for ii in range(N)
                for jj in range(ii + 1, N)
                if uf[ii] != uf[jj]
                # Optimization to avoid cycles: from O(N^2) => O((N - num_edges)^2) per iteration
            ]

            if pool is None:
                edges_heap = [_worker_spanning_tree(y) for y in params]
            else:
                edges_heap = pool.map(_worker_spanning_tree, params)

            PQ.heapify(edges_heap)

    return forest[uf[0]], equiv_sets, equiv_signs


def get_M_full(M_sparse, equiv_sets, equiv_signs):
    """Creates a full matrix based on deductions done."""
    M_full = np.zeros(M_sparse.shape)
    for i in range(M_sparse.shape[0]):
        for j in range(M_sparse.shape[1]):
            M_full[i, j] = equiv_signs.get(equiv_sets[i, j], 0)
    return M_full


def make_one_permut(spanning_tree, source=None):
    """Creates a path from the spanning tree."""
    G = nx.MultiGraph()

    # Adding two copies of the edges to 'G' to make it Eulerian.
    G.add_edges_from(spanning_tree)
    G.add_edges_from(spanning_tree)

    permut = [None] * (len(spanning_tree) + 1)
    permut_idx = 0
    handled_nodes = set()

    for (u, v) in nx.eulerian_circuit(G, source=source):
        if u not in handled_nodes:
            handled_nodes.add(u)
            permut[permut_idx] = u
            permut_idx += 1

        if v not in handled_nodes:
            handled_nodes.add(v)
            permut[permut_idx] = v
            permut_idx += 1

    return permut


def SC(M_full, permut):
    """Calculates the maximum number of sign changes in M_full."""
    M_permut = M_full[permut, :]
    sc = 0
    max_cols = []

    for j in range(M_permut.shape[1]):
        col_sc = 0
        for i in range(1, M_permut.shape[0]):
            if M_permut[i, j] != M_permut[i - 1, j]:
                col_sc += 1

        if col_sc == sc:
            max_cols.append(j)
        elif col_sc > sc:
            max_cols = [j]
            sc = col_sc

    return sc, max_cols
