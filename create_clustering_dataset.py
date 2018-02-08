import numpy as np


def get_random_samples(data, num=10):
    return np.random.choice(data.shape[0], num)


def choose_words_for_clustering(data, sample_num=10):
    '''
    Choose <sample> random descriptors from each track in <data>
    '''
    shape_value = data[0].shape[1]  # shape is supposed to be (n, 25)
    samples = np.array([]).reshape(0, shape_value)
    for track in data:
        samples = np.concatenate(
            (samples, track[get_random_samples(track, sample_num)]))

    return samples
