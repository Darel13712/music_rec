from config import path
from tqdm import tqdm
from hdf5 import *
import tables
import glob
import pandas as pd
from multiprocessing import Pool, cpu_count


def read_h5(filename):
    try:
        file = tables.File(filename)
        return file
    except:
        print('Error reading file {}, skipping'.format(filename))
        return None

def list_files(folder):
    h5 = folder + 'data/**/*.h5'
    files = list(glob.iglob(h5, recursive=True))
    print(len(files))
    return files

def get_mismatches(p):
    mismatches = []
    with open(p) as f:
        for line in f.readlines():
            line = line[line.find('<') + 1:]
            line = line[:line.find(' ')]
            mismatches += [line.encode()]
    mismatches = list(map(lambda x: x.decode('utf-8'), mismatches))
    return mismatches

def parallel(func, files):
    res = []
    with Pool(processes=cpu_count()) as pool:
        gen = pool.imap(func, files)
        for stat in tqdm(gen, total=len(files)):
            res.append(stat)
    return res

def get_info(filename):
    file = read_h5(filename)

    if file is None:
        return {'song_id': None,
                'artist': None,
                'title': None,
                'album': None,
                'mb_id': None,
                'd7_id': None}

    # We should've used
    # >>> for song_index in range(get_num_songs(file)):
    # but all the files have only one song in it

    song_index = 0

    id = get_soid(file, song_index)
    title = get_title(file, song_index)
    artist = get_artist(file, song_index)
    album = get_album(file, song_index)
    mbid = get_mbid(file, song_index)
    id7 = get_7did(file, song_index)

    file.close()

    return {'song_id': id,
            'artist': artist,
            'title': title,
            'album': album,
            'mb_id': mbid,
            'd7_id': int(id7)}

def collect_features(filename):

    file = read_h5(filename)

    if file is None:
        return (None, None)

    song_index = 0

    id = get_soid(file, song_index)
    song_vector = get_features(file, song_index).astype(np.float16)

    file.close()

    return (id, song_vector)

def read_index(files, mismatches=None):
    df = parallel(get_info, files)
    df = pd.DataFrame(df)
    obj_cols = df.columns[df.dtypes == 'object']
    df.loc[:, obj_cols] = df.loc[:, obj_cols].apply(lambda x: x.str.decode('utf-8'), axis=0)
    if mismatches is not None:
        df = df.loc[~df.song_id.isin(mismatches)]
    return df


def read_features(files, mismatches=None):
    features = parallel(collect_features, files)
    index, features = zip(*features)
    features = pd.Series(features, index=index)
    features.index = features.index.map(lambda x: x.decode('utf-8'))
    if mismatches is not None:
        features = features[~features.index.isin(mismatches)]
    return features

#%%  ================================================================

files = list_files(path.msd)

mismatches = get_mismatches(path.data + 'sid_mismatches.txt')

df = read_index(files, mismatches)
df.to_csv(path.out + 'msd_index.csv', index=False)

features = read_features(files, mismatches)
features.to_csv(path.out + 'msd_features.csv')
