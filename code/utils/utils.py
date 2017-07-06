import numpy as np

def opinion_RMSE(commenter_ids, pred_matrix, truth_df, topic_id=None):
    """Calculates the RMSE for a given topic_id given the commenters and the
    predictions and ground truth.

    The predictions should be in form of a matrix with rows and columns
    standing for the given commenter_ids and the (i, j)th entry giving the
    absolute value of the difference between them for the given topic_id.
    If it is None, it is assumed that there is only one topic.

    The truth_df contains the absolute angles for the users."""

    raise NotImplemented('This should be weighed by something?')


def truth_df_to_matrix(truth_df):
    """Converts the truth_df which has absolute angles to a matrix."""
    commenter_df = truth_df[truth_df.type == 'commenter']
    commenter_ids = sorted(commenter_df.id.values)
    N = len(commenter_ids)
    commenter_angle_map = dict(commenter_df[['id', 'opinion']].values)
    mat = np.zeros((N, N))
    for i, c1_id in enumerate(commenter_ids):
        for j, c2_id in enumerate(commenter_ids):
            mat[i, j] = np.abs(commenter_angle_map[c1_id] -
                               commenter_angle_map[c2_id])

    return mat


def pred_df_to_matrix(pred_df):
    """Converts angles produced into a matrix."""
    pass

