from sklearn.externals import joblib
from config import path
from tqdm import tqdm
from hdf5 import *
import tables
import glob
import pandas as pd

#%%
df = pd.DataFrame(columns=['song_id', 'artist', 'title', 'album'])
h5 = path.mss + 'data/**/*.h5'
files = list(glob.iglob(h5, recursive=True))
print(len(files))

for filename in tqdm(files):
    try:
        file = tables.File(filename)
    except:
        print('Error reading file {}, skipping'.format(filename))
        continue
    for song_index in range(get_num_songs(file)):
        id     = get_soid  (file, song_index)
        title  = get_title (file, song_index)
        artist = get_artist(file, song_index)
        album  = get_album (file, song_index)
        mbid   = get_mbid  (file, song_index)
        id7    = get_7did  (file, song_index)
        df = df.append({'song_id': id,
                        'artist' : artist,
                        'title'  : title,
                        'album'  : album,
                        'mb_id'  : mbid,
                        'd7_id'   : id7},
                       ignore_index=True)

    file.close()

df = df.apply(lambda x: x.str.decode('utf-8'), axis=0)
#%%
df.to_csv(path.out + 'msd_index.csv', index=False)


#%%
mismatches = []
with open(path.data + 'sid_mismatches.txt') as f:
    for line in f.readlines():
        line = line[line.find('<') + 1:]
        line = line[:line.find(' ')]
        mismatches += [line.encode()]

#%%

features = pd.Series()
for filename in tqdm(files):
    try:
        file = tables.File(filename)
    except:
        print('Error reading file {}, skipping'.format(filename))
        continue
    for song_index in range(get_num_songs(file)):
        id = get_soid(file, song_index)
        song_vector = get_features(file, song_index).astype(np.float16)
        features[id] = song_vector
    file.close()

#%%
filtered = features[~features.index.isin(mismatches)]
filtered.index = filtered.index.map(lambda x: x.decode('utf-8'))

#%%
filtered.to_csv(path.out + 'msd_features.csv')

#%%

