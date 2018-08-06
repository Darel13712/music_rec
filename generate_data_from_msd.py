from hdf5 import *
import tables
import glob
import pickle



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
