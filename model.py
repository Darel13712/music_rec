import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

def get_item_attention(play_count: int) -> float:
    """
    Convert play count to attention value [0..1]

    Parameters
    ----------
    play_count: int > 0
        number of times the song was played

    Returns
    -------
    Float attention value: [0..1]
    """
    if play_count > 0:
        return np.log10(play_count + 0.3) if play_count < 10 else 1.0
    elif play_count == 0:
        return 0.0
    else:
        raise ValueError('play_count cannot be negative (got {})'.format(play_count))
#%%

#%%
