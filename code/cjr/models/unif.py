import numpy as np
from collections import defaultdict
import pandas as pd

try:
    'get_unique_topic' in globals()
except NameError:
    from ..utils import get_unique_topic

# Steps:
#
#  - For each `topic_id`:
#    - For each pair (`parent_commenter_id`, `child_commenter_id`) in `topic_id`:
#      - `pred = #(disagreements) / #(total)`

def get_pred_unif_polar(df, topic_id=None):
    """Assume that the opinions of the voters and commenters lie on uniformly
    on a unit circle."""
    if topic_id is None:
        topic_id = get_unique_topic(df.ArticleTopic)

    grouped = (df.groupby(['ArticleTopic',
                           'ParentCommenterId',
                           'ChildCommenterId'])
               .agg({ 'TraceId': lambda x: x.size, 'Disagree': np.sum})
               .rename(columns={'TraceId': 'NumPatterns',
                                'Disagree': 'NumDisagreements'})
               .reset_index())
    # grouped['pred'] = grouped['NumDisagreements'] / grouped['NumPatterns']
    # TODO: We have not grouped the P/C and C/P relations together.
    # In real data, it seems very rare that the same commenters converse more
    # than once. However, we ought to just in case.
    total_pats = defaultdict(lambda: 0)
    total_disagree = defaultdict(lambda: 0)
    all_tups = set()

    for aId, pId, cId, pat, dis in grouped[['ArticleTopic',
                                            'ParentCommenterId',
                                            'ChildCommenterId',
                                            'NumPatterns',
                                            'NumDisagreements']].values:
        k = min(pId, cId)
        l = max(pId, cId)
        total_pats[aId, k,l] += pat
        total_disagree[aId, k, l] += dis
        all_tups.add((aId, k, l))

    data = [{'ArticleTopic': aId, 'Commenter1Id': k, 'Commenter2Id': l,
             'NumPatterns': total_pats[aId, k, l],
             'NumDisagreements': total_disagree[aId, k, l],
             'pred': total_disagree[aId, k, l] / total_pats[aId, k, l]}
            for aId, k, l in all_tups]

    return pd.DataFrame.from_dict(data)
