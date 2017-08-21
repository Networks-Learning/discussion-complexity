import click
import cjr.models.dim as D
import networkx as nx


@click.command()
@click.option('-N', '--comments', 'num_comments', help='Number of comments.', type=int)
@click.option('-M', '--voters', 'num_voters', help='Number of voters.', type=int)
@click.option('--seed', help='Seed for the voting patterns.', type=int)
@click.option('--up', help='Upvote probability.', type=float, default=0.2)
@click.option('--down', help='Downvote probability.', type=float, default=0.2)
@click.option('--verbose/--no-verbose', help='Verbose.', default=False)
def cmd(num_comments, num_voters, seed, up, down, verbose):
    rand_votes = D.gen_voting_patterns(num_comments, num_voters,
                                       p_up=up, p_down=down, seed=seed)

    unique_voting_pats = list(set(rand_votes))
    for dim in range(1, 2 ** num_voters):
        s, com_vars, voter_vars = D.create_prob(dim, unique_voting_pats)
        if s.check().r != 1:
            if verbose:
                print('Breaking for dim = ', dim)
            break

    max_sat_dim = dim - 1

    graph = D.create_disgreement_graph(num_comments, unique_voting_pats)
    clique = next(nx.algorithms.clique.find_cliques(graph))
    max_clique_dim = len(clique) - 1

    if max_sat_dim != max_clique_dim:
        print('seed = ', seed, 'N = ', N, 'M = ', M,
              'up = ', up, 'down = ', down)


if __name__ == '__main__':
    cmd()
