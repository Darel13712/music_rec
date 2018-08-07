import numpy as np
import hdf5_getters

# My wrapping of hdf5 wrapper to make code look cleaner
get_chroma = hdf5_getters.get_segments_pitches
get_timbre = hdf5_getters.get_segments_timbre
get_max_loudness = hdf5_getters.get_segments_loudness_max
get_artist = hdf5_getters.get_artist_name
get_title = hdf5_getters.get_title
get_num_songs = hdf5_getters.get_num_songs
get_mbid = hdf5_getters.get_artist_mbid
get_soid = hdf5_getters.get_song_id


def get_features(file, song_index=0):
    """
    Get a set of selected features for a given hdf5 file

    Parameters
    ----------
    file:
        opened with tables hdf5 file
    song_index: int
        hdf5 can store multiple songs at once

    Returns
    -------
    ndarray
        np.array with a shape (number of beats, 25)
        where 25 stays for 12 timbre features, 12 chroma components
        and a max loudness value for current beat.
    """

    chroma = get_chroma(file, song_index)
    timbre = get_timbre(file, song_index)
    max_loudness = get_max_loudness(file, song_index)

    # normalize to get ~ 0-1
    timbre = (timbre + 1000) / 1200
    max_loudness = (max_loudness + 70) / 70
    max_loudness = max_loudness.reshape(-1, 1)
    features = np.hstack([timbre, chroma, max_loudness])
    return features
