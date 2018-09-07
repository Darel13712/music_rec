import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from sklearn.externals import joblib

from config import path
from collections import namedtuple

Triple = namedtuple('Triple', ['user', 'song', 'known', 'unknown'])
#%%
def get_sample(df):
    return df.sample(n=1).iloc[0]

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
    Get random triple (user, current song, played song, not played song)

    Parameters
    ----------
    play_counts: DataFrame
        DataFrame with play history for users
    index: DataFrame
        DataFrame with info for every song in dataset

    Returns
    -------
    Triple: NamedTuple
        Named tuple with user id, current played song id, other played song id, not played song id
    """

    sample = get_sample(play_counts)
    user, current_song = sample.user, sample.song
    all_played_songs = play_counts.loc[play_counts.user == user, 'song']

    known_song = current_song
    while known_song == current_song: # whatif user listened to 1 song?
        known_song = get_sample(all_played_songs)

    all_played_songs = set(all_played_songs)

    unknown_song = current_song
    while unknown_song in all_played_songs: # whatif user listened all the songs?
        unknown_song = get_sample(index).song_id

    return Triple(user, current_song, known_song, unknown_song)

def get_played_songs(play_counts: pd.DataFrame, user: str) -> pd.Series:
    return play_counts.loc[play_counts.user == user, 'song'].values

def get_components(features: pd.DataFrame, song: str) -> np.array:
    return features[song]


#%%
def read_play_history(path):
    return pd.read_csv(path, sep='\t', header=None, names=['user', 'song', 'plays'])
#%%
df = read_play_history(path.data + 'EvalDataYear1MSDWebsite/year1_test_triplets_hidden.txt')
index = joblib.load(path.out + 'index.jbl')
features = joblib.load(path.out + 'song_features.jbl')
#%%
current = draw_triple(df, index)
played_songs = get_played_songs(df, current.user)


#%%
c = get_components(features, current.unknown)


