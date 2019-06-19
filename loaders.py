import pandas as pd


def read_features(path):
    return pd.read_csv(path,
                       index_col=0,
                       header=None,
                       squeeze=True,
                       names=['song_index', 'features'])


def read_train(path):
    return pd.read_csv(path, dtype={'user': 'category',
                                    'song': 'category'})


def filter_mss(df, ind):
    return df.loc[df.song.isin(ind.song_id)]


def load(path=None, subset=False):
    if path is None:
        path = 'data/'
    train = read_train(f'{path}train_playcounts.csv')
    query = pd.read_csv(f'{path}test_playcounts_visible.csv')
    test = pd.read_csv(f'{path}test_playcounts_hidden.csv')
    if subset:
        dataset = 'mss'
    else:
        dataset = 'msd'
    index = pd.read_csv(f'{path}{dataset}_index.csv')
    features = read_features(f'{path}{dataset}_features.csv')

    if dataset == 'mss':
        train = filter_mss(train, index)

    return train, query, test, index, features
