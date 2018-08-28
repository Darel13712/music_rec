import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from sklearn.externals import joblib

from config import path
from collections import namedtuple

Triple = namedtuple('Triple', ['user', 'known', 'unknown'])

def get_item_attention(play_count: int) -> float:
    """
    Convert play count to attention value [0..1]

    Parameters
    ----------
    play_count: int > 0
        number of times the song was played

    Returns
    -------
    attention: float
        Float attention value: [0..1].
        0 for 0 play_count and shifted logarithm for positive values.
        Clips to 1 at 10 play_counts.
    """
    if play_count > 0:
        return np.log10(play_count + 0.3).round(2) if play_count < 10 else 1.0
    elif play_count == 0:
        return 0.0
    else:
        raise ValueError('play_count cannot be negative (got {})'.format(play_count))

def draw_triple(play_counts: pd.DataFrame, index: pd.DataFrame) -> Triple:
    """
    Get random triple (user, played song, not played song)

    Parameters
    ----------
    play_counts: DataFrame
        DataFrame with play history for users
    index: DataFrame
        DataFrame with info for every song in dataset

    Returns
    -------
    Triple: NamedTuple
        Named tuple with user id, played song id, not played song id
    """

    sample = play_counts.sample(n=1)
    user, known_song = sample.user.iloc[0], sample.song.iloc[0]
    all_played_songs = set(play_counts.loc[play_counts.user == user, 'song'])

    unknown_song = known_song
    while unknown_song in all_played_songs:
        unknown_song = index.sample(n=1).song_id.iloc[0]

    return Triple(user, known_song, unknown_song)


#%%
def read_play_history(path):
    return pd.read_csv(path, sep='\t', header=None, names=['user', 'song', 'plays'])
#%%
df = read_play_history(path.data + 'EvalDataYear1MSDWebsite/year1_test_triplets_hidden.txt')
index = joblib.load(path.out + 'index.jbl')
#%%
draw_triple(df, index)
#%%
