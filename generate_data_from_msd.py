import hdf5_getters
import tables
import glob
import pickle
import numpy as np

# My wrapping of hdf5 wrapper to make code look cleaner
get_chroma = hdf5_getters.get_segments_pitches
get_timbre = hdf5_getters.get_segments_timbre
get_max_loudness = hdf5_getters.get_segments_loudness_max
get_artist = hdf5_getters.get_artist_name
get_title = hdf5_getters.get_title
get_num_songs = hdf5_getters.get_num_songs
get_mbid = hdf5_getters.get_artist_mbid


def get_features(file, song_index=0):
    chroma = get_chroma(file, song_index)
    timbre = get_timbre(file, song_index)
    max_loudness = get_max_loudness(file, song_index)

    # normalize to get ~ 0-1
    timbre = (timbre + 1000) / 1200
    max_loudness = (max_loudness + 70) / 70
    max_loudness = max_loudness.reshape(-1, 1)
    features = np.hstack([timbre, chroma, max_loudness])
    return features


data = []
for filename in glob.iglob('data/**/*.h5', recursive=True):
    try:
        file = tables.File(filename)
    except:
        print('Error reading file {}, skipping'.format(filename))
        continue
    for song_index in range(get_num_songs(file)):
        song_vector = get_features(file, song_index)
        data.append(song_vector)
    file.close()

pickle.dump(data, open('song_features.p', 'wb'))
