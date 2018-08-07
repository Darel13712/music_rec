from sklearn.externals import joblib
from config import path
from tqdm import tqdm
from hdf5 import *
import tables
import glob
import pandas as pd

#%%
df = pd.DataFrame(columns=['song_id', 'artist', 'title'])
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
        df = df.append({'song_id': id,
                        'artist': artist,
                        'title': title}, ignore_index=True)

    file.close()
#%%
joblib.dump(df, path.out + 'index.jbl')


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

#%%
joblib.dump(filtered, path.out + 'song_features.jbl')