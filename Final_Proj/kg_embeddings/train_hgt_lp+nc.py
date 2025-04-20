import itertools
import argparse
import math
import urllib.request

import numpy as np
import scipy.io
import scipy.sparse as sp
from sklearn.metrics import roc_auc_score
from tqdm import tqdm

from models import *
from utils import *
import dgl
import dgl.function as fn
from dgl.nn.pytorch.linear import TypedLinear
from dgl.nn.pytorch.softmax import edge_softmax

import torch
import torch.nn as nn
import torch.nn.functional as F

def train(dataloader, model):
    optimizer = torch.optim.AdamW(model.parameters())
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer, total_steps=total_steps, max_lr=max_lr
    )
    best_train_acc = torch.tensor(0)
    best_val_acc = torch.tensor(0)
    best_test_acc = torch.tensor(0)

    best_train_acc2 = torch.tensor(0)
    best_val_acc2 = torch.tensor(0)
    best_test_acc2 = torch.tensor(0)

    for epoch in range(n_epoch + 1):
        for input_nodes, positive_graph, negative_graph, blocks in dataloader:
            blocks = [b.to(device) for b in blocks]
            positive_graph = positive_graph.to(torch.device(device))
            negative_graph = negative_graph.to(torch.device(device))

            train_idx = positive_graph.edata['train_mask']==0
            val_idx = positive_graph.edata['train_mask']==1
            test_idx = positive_graph.edata['train_mask']==2

            train_idx2 = positive_graph.ndata['train_mask']==0
            val_idx2 = positive_graph.ndata['train_mask']==1
            test_idx2 = positive_graph.ndata['train_mask']==2

            model.train()
            h, _, _ = model(blocks, 'emb')

            # link prediction
            pos_score = edge_pred(positive_graph, h)
            neg_score = edge_pred(negative_graph, h)
            lp_loss = compute_loss_multilabel(pos_score[train_idx], neg_score, positive_graph.edata['_TYPE'][train_idx])

            # node classification
            score = node_pred(h)
            nc_loss = F.binary_cross_entropy(score[train_idx2], positive_graph.ndata['labels'][train_idx2])
            
            loss = lp_loss + nc_loss

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
            optimizer.step()
            scheduler.step()

            if epoch % 5 == 0:
                model.eval()
                pos_score = pos_score.argmax(1)
                neg_score = neg_score.argmax(1)
                pos_label = positive_graph.edata['_TYPE']
                neg_label = torch.tensor(positive_graph.edata['_TYPE'].max()+1).repeat(neg_score.shape[0])

                train_acc = calc_acc(torch.cat([pos_score[train_idx], neg_score]), torch.cat([pos_label[train_idx], neg_label]))
                if len(val_idx)>0:
                    val_acc = calc_acc(torch.cat([pos_score[val_idx], neg_score]), torch.cat([pos_label[val_idx], neg_label]))
                if len(test_idx)>0:
                    test_acc = calc_acc(torch.cat([pos_score[test_idx], neg_score]), torch.cat([pos_label[test_idx], neg_label]))
                if best_val_acc < val_acc:
                    best_val_acc = val_acc
                if best_test_acc < test_acc:
                    best_test_acc = test_acc

                node_preds = torch.where(score>thres, 1, 0)
                nc_train_acc = calc_acc(node_preds[train_idx2], positive_graph.ndata['labels'][train_idx2])
                if len(val_idx2)>0:
                    nc_val_acc = calc_acc(node_preds[val_idx2], positive_graph.ndata['labels'][val_idx2])
                if len(test_idx2)>0:
                    nc_test_acc = calc_acc(node_preds[test_idx2], positive_graph.ndata['labels'][test_idx2])
                if best_val_acc2 < nc_val_acc:
                    best_val_acc2 = nc_val_acc
                if best_test_acc2 < nc_test_acc:
                    best_test_acc2 = nc_test_acc

                print(
                        "Epoch: %d LR: %.5f LP Loss %.4f, NC Loss %.4f, Loss %.4f, Train LP Acc %.4f, Val LP Acc %.4f (Best %.4f), Test LP Acc %.4f (Best %.4f), Train NC Acc %.4f, Val NC Acc %.4f (Best %.4f), Test NC Acc %.4f (Best %.4f)"
                        % (
                            epoch,
                            optimizer.param_groups[0]["lr"],
                            lp_loss.item(),
                            nc_loss.item(),
                            loss.item(),
                            train_acc.item(),
                            val_acc.item(),
                            best_val_acc.item(),
                            test_acc.item(),
                            best_test_acc.item(),
                            nc_train_acc.item(),
                            nc_val_acc.item(),
                            best_val_acc2.item(),
                            nc_test_acc.item(),
                            best_test_acc2.item(),
                        )
                    )
    torch.save({
            'epoch': epoch+1,
            'model_state_dict': model.state_dict(),
            'edge_pred_state_dict': edge_pred.state_dict(),
            'node_pred_state_dict': node_pred.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': loss,
            }, '/nfs/turbo/umms-drjieliu/proj/prompt_re/kg_embeddings/outputs/lp+nc_final_model.pth')

# args
## model args
n_epoch = 200
n_hid = 256
n_inp = 768
clip = 1.0
max_lr =1e-3
n_out = 30
n_layers = 2
n_heads = 4
head_size = 10
if torch.cuda.is_available():
    device = torch.device("cuda:0")
else:
    device = torch.device("cpu")
# print(torch.cuda.is_available())
# device = torch.device("cuda:0")
print(device)
## dataloader args
num_workers = 0
batch_size = 1000000
train_seeds = 42
neg_sample_rate = 1
total_steps = 1000000
## other parameters
thres = 0.5
FILE = '/nfs/turbo/umms-drjieliu/proj/prompt_re/kg_embeddings/data/semantic_graph_nodetypes.bin'

data, _ = dgl.load_graphs(FILE)
G = data[-1]
G.edata['train_mask'] = split_graph_mask(G)
G.ndata['train_mask'] = split_node_mask(G.num_nodes())
g = dgl.to_homogeneous(G, ndata=['emb', 'labels', 'train_mask'], edata=['train_mask'])

# generate dataloader
train_eid_dict = {
    etype: g.edges(etype=etype, form='eid')
    for etype in g.canonical_etypes}
sampler = dgl.dataloading.MultiLayerFullNeighborSampler(n_layers)
sampler = dgl.dataloading.as_edge_prediction_sampler(
    sampler, negative_sampler=dgl.dataloading.negative_sampler.Uniform(neg_sample_rate))
dataloader = dgl.dataloading.DataLoader(
    g,
    train_eid_dict,
    sampler,
    batch_size=batch_size,
    shuffle=True,
    drop_last=False,
    # pin_memory=True,
    num_workers=num_workers)

# define model
node_dict = {}
edge_dict = {}
for ntype in G.ntypes:
    node_dict[ntype] = len(node_dict)
for etype in G.etypes:
    edge_dict[etype] = len(edge_dict)
    G.edges[etype].data["id"] = (
        torch.ones(G.num_edges(etype), dtype=torch.long) * edge_dict[etype]
    )
model = HGT_built_in_sample(
    node_dict,
    edge_dict,
    n_inp,
    n_hid,
    n_layers,
    n_heads,
    head_size,
    n_out
    ).to(device)
edge_pred = MLPPredictor(n_out, len(G.etypes) + 1)
node_pred = Multilabel_Node_Classifier(n_out, G.ndata['labels'].shape[1])

train(dataloader, model)