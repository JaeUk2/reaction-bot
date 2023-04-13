import numpy as np
import random
import torch
from transformers import is_torch_available

def set_seed(seed):
    """
    seed 고정하는 함수 (random, numpy, torch)

    Args:
        seed (:obj:`int`): The seed to set.
    """
    random.seed(seed)
    np.random.seed(seed)
    if is_torch_available():
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)  # if use multi-GPU
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    print('lock_all_seed')


def num_to_label(label):
    num_label = []
    sentiment = {1:"분노", 2:"슬픔", 3:"불안", 4:"상처", 5:"당황", 6:"기쁨"}
    for v in label:
        num_label.append(sentiment[v])
    
    return num_label
