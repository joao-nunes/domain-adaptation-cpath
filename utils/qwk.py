import torch
import numpy as np


def qwk(logits, labels, num_classes=5, epsilon=1e-10):
    
    device = logits.device
    probas = torch.nn.functional.softmax(logits.float(), dim=1).float()
    #print(probas.shape)
    #print(torch.sum(probas,dim=1))
    #print(torch.argmax(probas,dim=1))
    #print(labels)
    labels = torch.nn.functional.one_hot(labels, num_classes).float()
    repeat_op = torch.arange(0, num_classes).view(num_classes, 1).repeat(1, num_classes).float()
    repeat_op_sq = torch.pow((repeat_op - repeat_op.transpose(0, 1)), 2)
    weights = repeat_op_sq / (num_classes - 1) ** 2
    weights = weights.to(device)
    pred_ = probas ** 2
    pred_norm = pred_ / (epsilon + pred_.sum(1).view(-1, 1))

    hist_rater_a = pred_norm.sum(0)
    hist_rater_b = labels.sum(0)
    conf_mat = torch.matmul(pred_norm.transpose(0, 1), labels)

    nom = (weights * conf_mat).sum()
    denom = (weights * torch.matmul(hist_rater_a.view(num_classes, 1), hist_rater_b.view(1, num_classes)) / labels.shape[0]).sum()
    return nom / (denom + epsilon)

def histogram(ratings, min_rating=None, max_rating=None):
    """https://github.com/benhamner/Metrics/blob/master/Python/ml_metrics/quadratic_weighted_kappa.py"""

    if min_rating is None:
        min_rating = min(ratings)
    if max_rating is None:
        max_rating = max(ratings)
    num_ratings = int(max_rating - min_rating + 1)
    hist_ratings = [0 for x in range(num_ratings)]
    for r in ratings:
        hist_ratings[r - min_rating] += 1
    return hist_ratings

def confusion_matrix(rater_a, rater_b, min_rating=None, max_rating=None):
    """https://github.com/benhamner/Metrics/blob/master/Python/ml_metrics/quadratic_weighted_kappa.py"""
    assert(len(rater_a) == len(rater_b))
    if min_rating is None:
        min_rating = min(rater_a + rater_b)
    if max_rating is None:
        max_rating = max(rater_a + rater_b)
    num_ratings = int(max_rating - min_rating + 1)
    conf_mat = [[0 for i in range(num_ratings)]
                for j in range(num_ratings)]
    #pdb.set_trace()
    for a, b in zip(rater_a, rater_b):
        conf_mat[a - min_rating][b - min_rating] += 1
    return conf_mat

def quadratic_weighted_kappa(rater_a, rater_b, min_rating=None, max_rating=None, eps=1e-10):
    """https://github.com/benhamner/Metrics/blob/master/Python/ml_metrics/quadratic_weighted_kappa.py"""

    rater_a = np.array(rater_a, dtype=int)
    rater_b = np.array(rater_b, dtype=int)
    assert(len(rater_a) == len(rater_b))
    #if min_rating is None:
    #    min_rating = min(min(rater_a), min(rater_b))
    #if max_rating is None:
    #    max_rating = max(max(rater_a), max(rater_b))
    conf_mat = confusion_matrix(rater_a, rater_b,
                                min_rating, max_rating)
    num_ratings = len(conf_mat)
    num_scored_items = float(len(rater_a))

    hist_rater_a = histogram(rater_a, min_rating, max_rating)
    hist_rater_b = histogram(rater_b, min_rating, max_rating)

    numerator = 0.0
    denominator = 0.0

    for i in range(num_ratings):
        for j in range(num_ratings):
            expected_count = (hist_rater_a[i] * hist_rater_b[j]
                              / num_scored_items)
            d = pow(i - j, 2.0) / pow(num_ratings - 1, 2.0)
              
            numerator += d * conf_mat[i][j] / num_scored_items
            denominator += d * expected_count / num_scored_items

    return 1.0 - numerator / (denominator + eps)


def find_nearest(array, value):
    array = np.asarray(array)
    idx = (np.abs(array - value)).argmin()
    return idx, array[idx]

