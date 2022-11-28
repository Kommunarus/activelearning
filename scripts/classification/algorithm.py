import torch
import math

def least_confidence(prob_dist, sorted=False, num_labels=2):
    if sorted:
        simple_least_conf = prob_dist.data[0]
    else:
        simple_least_conf = torch.max(prob_dist, 1)[0]
    # num_labels = 2 # number of labels
    normalized_least_conf = (1 - simple_least_conf) * (num_labels / (num_labels - 1))
    return normalized_least_conf.tolist()

def entropy_based(prob_dist, num_labels):
    log_probs = prob_dist * torch.log2(prob_dist)
    raw_entropy = 0 - torch.sum(log_probs, dim=1)
    normalized_entropy = raw_entropy / math.log2(num_labels)
    return normalized_entropy.tolist()

def margin_confidence(prob_dist, sorted=False):
    if not sorted:
        prob_dist, _ = torch.sort(prob_dist, descending=True, dim=1)
    difference = (prob_dist.data[:, 0] - prob_dist.data[:, 1])
    margin_conf = 1 - difference
    return margin_conf.tolist()

def ratio_confidence(prob_dist, sorted=False):
    if not sorted:
        prob_dist, _ = torch.sort(prob_dist, descending=True, dim=1)
    ratio_conf = prob_dist.data[:, 1] / prob_dist.data[:, 0]
    return ratio_conf.tolist()

# https://towardsdatascience.com/how-to-cluster-images-based-on-visual-similarity-cd6e7209fe34