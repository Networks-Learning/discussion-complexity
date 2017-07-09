import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from collections import defaultdict


def get_all_commenter_ids(df):
    """Returns a (sorted) list of all commenter ids in the dataframe."""
    return sorted(set(df['ParentCommenterId']).union(set(df['ChildCommenterId'])))


def get_all_voter_ids(df):
    """Returns a (sorted) list of all voter ids in the dataframe."""
    return sorted(set(df['VoterId']))


def get_all_topic_ids(df):
    """Returns a (sorted) list of all voter ids in the dataframe."""
    return sorted(set(df['ArticleTopic']))


def get_unique_topic(topics):
    """Returns a unique topic if in df. Otherwise, raises an error."""
    unique_topics_ids = topics.unique()
    assert len(unique_topics_ids) == 1, ("If topic_id is not given, "
                                         "there must be only one unique "
                                         "topic_id in the dataset.")
    return unique_topics_ids[0]



def truth_df_to_matrix(truth_df, commenter_ids, topic_id=None):
    """Converts the truth_df which has absolute angles to a matrix."""
    if topic_id is None:
        topic_id = get_unique_topic(truth_df.topic)

    topic_df = truth_df[truth_df.topic == topic_id]
    commenter_df = topic_df[topic_df.type == 'commenter']
    # commenter_ids = sorted(commenter_df.id.values)
    N = len(commenter_ids)
    commenter_opinion_map = dict(commenter_df[['id', 'opinion']].values)
    mat = np.zeros((N, N))
    for i, c1_id in enumerate(commenter_ids):
        for j, c2_id in enumerate(commenter_ids):
            mat[i, j] = np.abs(commenter_opinion_map[c1_id] -
                               commenter_opinion_map[c2_id])

    return mat


def pred_df_to_matrix(pred_df, commenter_ids, topic_id=None):
    """Converts angles produced into a matrix."""
    if topic_id is None:
        topic_id = get_unique_topic(pred_df.ArticleTopic)

    topic_df = pred_df[pred_df.ArticleTopic == topic_id]
    N = len(commenter_ids)
    mat = np.zeros((N, N))
    predictions = defaultdict(lambda: 0)
    for c1_id, c2_id, pred in topic_df[['Commenter1Id',
                                        'Commenter2Id',
                                        'pred']].values:
        predictions[c1_id, c2_id] = pred

    for i, c1_id in enumerate(commenter_ids):
        for j, c2_id in enumerate(commenter_ids):
            mat[i, j] = predictions[c1_id, c2_id]
    return mat


def plot_angles(angles, variance=None):
    c1, = sns.color_palette(n_colors=1)
    ax = plt.subplot(111, projection='polar')

    # ax.plot([2 * theta, 2 * theta], [0, 1])
    if variance is None:
        variance = [1.0] * len(angles)

    for theta, var in zip(angles, variance):
        ax.arrow(x=theta, y=0, dx=0, dy=var,
                 head_width=0.05, head_length=0.1, alpha=0.2,
                 ec=c1, fc=c1, linewidth=1, length_includes_head=True)
    ax.set_yticklabels([])
    ax.set_xticklabels([])
    ax.set_ylim([0, np.max(variance)])
    return ax
    # plt.hist();


def opinion_RMSE(pred_matrix, truth_matrix, with_zeros=False):
    """Calculates the RMSE for a given topic_id given the commenters and the
    predictions and ground truth.

    The predictions should be in form of a matrix with rows and columns
    standing for the given commenter_ids and the (i, j)th entry giving the
    absolute value of the difference between them for the given topic_id.
    If it is None, it is assumed that there is only one topic.

    """
    if with_zeros:
        N = pred_matrix.shape[0]
        return np.sqrt(np.sum(np.square(pred_matrix - truth_matrix))) / N
    else:
        mask = pred_matrix > 0
        N = np.sum(mask)
        return np.sqrt(np.sum(np.square(pred_matrix[mask] - truth_matrix[mask]))) / N

