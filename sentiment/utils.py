import numpy as np
import random
import torch
from transformers import is_torch_available
from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score
import sklearn
from torch import nn

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
    # sentiment = {1:"분노", 2:"슬픔", 3:"불안", 4:"상처", 5:"당황", 6:"기쁨"}
    sentiment = {0:"분노", 1:"슬픔", 2:"불안", 3:"상처", 4:"당황", 5:"기쁨", 6:"중립"}
    for v in label:
        num_label.append(sentiment[v])
    
    return num_label

def sentiment_micro_f1(preds, labels):
    label_list = ['분노', '슬픔', '불안', '상처', '당황', '기쁨', '중립']
    
    label_indices = list(range(len(label_list)))
    return sklearn.metrics.f1_score(labels, preds, average="micro", labels=label_indices) * 100.0

def sentiment_auprc(probs, labels):
    labels = np.eye(7)[labels]

    score = np.zeros((7,))
    for c in range(7):
        targets_c = labels.take([c], axis=1).ravel()
        preds_c = probs.take([c], axis=1).ravel()
        precision, recall, _ = sklearn.metrics.precision_recall_curve(targets_c, preds_c)
        score[c] = sklearn.metrics.auc(recall, precision)
    return np.average(score) * 100.0

def compute_metrics(pred):
    """ validation을 위한 metrics function """
    labels = pred.label_ids
    preds = pred.predictions.argmax(-1)
    probs = pred.predictions

    # calculate accuracy using sklearn's function
    f1 = sentiment_micro_f1(preds, labels)
    auprc = sentiment_auprc(probs, labels)
    acc = accuracy_score(labels, preds) # 리더보드 평가에는 포함되지 않습니다.

    return {
      'micro_f1_score': f1,
      'auprc' : auprc,
      'accuracy': acc,
    }