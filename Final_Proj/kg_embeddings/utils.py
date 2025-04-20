import numpy as np
import scipy.sparse as sp
from sklearn.metrics import roc_auc_score

from models import *

import torch
import torch.nn.functional as F

def split_graph_mask(graph, train_prop=0.6, test_prop=0.2, val_prop=0.2):
    if len(graph.etypes) > 1:
        train_mask = {}
        for etype in graph.canonical_etypes:
            train_mask[etype] = torch.zeros(graph.num_edges(etype=etype))
            eids = np.arange(graph.num_edges(etype=etype))
            eids = np.random.permutation(eids)
            test_size = int(len(eids) * test_prop)
            val_size = int(len(eids) * val_prop)
            train_size = graph.num_edges(etype=etype) - val_size - test_size
            train_mask[etype][eids[:train_size]] = 0
            train_mask[etype][eids[train_size:train_size+val_size]] = 1
            train_mask[etype][eids[train_size+val_size:]] = 2
    else:
        train_mask = torch.zeros(graph.num_edges())
        eids = np.arange(graph.num_edges())
        eids = np.random.permutation(eids)
        test_size = int(len(eids) * test_prop)
        val_size = int(len(eids) * val_prop)
        train_size = graph.num_edges() - val_size - test_size
        train_mask[eids[:train_size]] = 0
        train_mask[eids[train_size:train_size+val_size]] = 1
        train_mask[eids[train_size+val_size:]] = 2
    return train_mask

def split_node_mask(n_nodes, train_prop=0.6, test_prop=0.2, val_prop=0.2):
    train_size = int(n_nodes*train_prop)
    test_size = int(n_nodes*test_prop)
    val_size = n_nodes - train_size - test_size
    shuffled = torch.randperm(n_nodes)
    
    train_mask = torch.zeros(n_nodes)
    train_mask[shuffled[:train_size]] = 0
    train_mask[shuffled[train_size:train_size+val_size]] = 1
    train_mask[shuffled[train_size+val_size:]] = 2
    return train_mask

def compute_loss_multilabel(pos_score, neg_score, pos_labels):
    scores = torch.cat([pos_score, neg_score])
    labels = torch.cat(
        [pos_labels, torch.tensor(pos_labels.max()+1).repeat(len(neg_score))]
    )
    return F.cross_entropy(scores, labels)

def compute_loss(pos_score, neg_score):
    n_edges = pos_score.shape[0]
    return (1 - pos_score + neg_score.view(n_edges, -1)).clamp(min=0).mean()

def calc_acc(pred, label):
    return (pred == label).float().mean()

def compute_loss(pos_score, neg_score, device):
    scores = torch.cat([pos_score, neg_score])
    labels = torch.cat(
        [torch.ones(pos_score.shape[0]), torch.zeros(neg_score.shape[0])]
    ).to(device)
    return F.binary_cross_entropy_with_logits(scores, labels)