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


print(dgl.__version__)

def partition_to_train(g, nfeat, efeat):
    g.ndata['inp'] = torch.zeros(g.num_nodes(), nfeat['_N/inp'].shape[1], dtype=torch.float64)
    g.ndata['_TYPE'] = torch.zeros(g.num_nodes(), dtype=torch.int64)
    g.edata['train_mask'] = torch.zeros(g.num_edges(), dtype=torch.float32)
    g.edata['_TYPE'] = torch.zeros(g.num_edges(), dtype=torch.int64)
    
    node_idx = g.ndata['inner_node'].bool()
    g.ndata['inp'][node_idx] = nfeat['_N/inp']
    g.ndata['_TYPE'][node_idx] = nfeat['_N/_TYPE']
    edge_idx = g.edata['inner_edge'].bool()
    g.edata['train_mask'][edge_idx] = efeat['_N:_E:_N/train_mask']
    g.edata['_TYPE'][edge_idx] = efeat['_N:_E:_N/_TYPE']

    return dgl.node_subgraph(g, node_idx)

# args
## model args
n_epoch = 500
n_hid = 256
n_inp = 1536 # 768
clip = 1.0
max_lr =1e-3
n_out = 400 # 768
n_layers = 2
n_heads = 4
head_size = 10
if torch.cuda.is_available():
    device = torch.device("cuda:0")
else:
    device = torch.device("cpu")
print(torch.cuda.is_available())
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
# FILE = '/nfs/turbo/umms-drjieliu/proj/prompt_re/kg_embeddings/data/semantic_graph_10part.bin'
FILE = '/nfs/turbo/umms-drjieliu/proj/medlineKG/data/glkb_processed_data/openai_embeddings/dgl_graphs/glkb_hier.bin'

data, _ = dgl.load_graphs(FILE)
G = data[-1]

G.edata['train_mask'] = split_graph_mask(G)
g = dgl.to_homogeneous(G, ndata=['emb'], edata=['train_mask'])

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
pred = MLPPredictor(n_out, len(G.etypes) + 1)

# state_dict = torch.load('/nfs/turbo/umms-drjieliu/proj/prompt_re/kg_embeddings/outputs/l2_binary_final_model.pth')
# model.load_state_dict(state_dict['model_state_dict'])

optimizer = torch.optim.AdamW(model.parameters())
scheduler = torch.optim.lr_scheduler.OneCycleLR(
    optimizer, total_steps=total_steps, max_lr=max_lr
)

# iter = 0
# train
best_train_acc = torch.tensor(0)
best_val_acc = torch.tensor(0)
best_test_acc = torch.tensor(0)

thres = 0.5

best_train_acc = torch.tensor(0)
best_val_acc = torch.tensor(0)
best_test_acc = torch.tensor(0)
for epoch in range(n_epoch + 1):
    for input_nodes, positive_graph, negative_graph, blocks in dataloader:
        blocks = [b.to(device) for b in blocks]
        positive_graph = positive_graph.to(torch.device(device))
        negative_graph = negative_graph.to(torch.device(device))

        train_idx = positive_graph.edata['train_mask']==0
        val_idx = positive_graph.edata['train_mask']==1
        test_idx = positive_graph.edata['train_mask']==2
        # train_idx2 = negative_graph.edata['train_mask']==0
        # val_idx2 = negative_graph.edata['train_mask']==1
        # test_idx2 = negative_graph.edata['train_mask']==2

        model.train()
        pred.train()
        # forward
        # h, ntype, etype = model(blocks, 'inp')
        h, ntype, etype = model(blocks, 'emb')
        ## filter target edge type
        pos_score = pred(positive_graph, h)
        neg_score = pred(negative_graph, h)
        
        loss = compute_loss_multilabel(pos_score[train_idx], neg_score[:10000], positive_graph.edata['_TYPE'][train_idx])
        # backward
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

        train_acc = calc_acc(torch.cat([pos_score[train_idx], neg_score[test_idx]]), torch.cat([pos_label[train_idx], neg_label[test_idx]]))
        if len(val_idx)>0:
            val_acc = calc_acc(torch.cat([pos_score[val_idx], neg_score[val_idx]]), torch.cat([pos_label[val_idx], neg_label[val_idx]]))
        if len(test_idx)>0:
            test_acc = calc_acc(torch.cat([pos_score[test_idx], neg_score[test_idx]]), torch.cat([pos_label[test_idx], neg_label[test_idx]]))
            
            print('total num of pos samples:', len(pos_score))
            print('total num of neg samples:', len(neg_score))
            print('false negative:', len(torch.where(pos_score==positive_graph.edata['_TYPE'].max()+1)[0]))
            print('false positive:', len(torch.where(neg_score!=positive_graph.edata['_TYPE'].max()+1)[0]))

        if best_val_acc < val_acc:
            best_val_acc = val_acc
        if best_test_acc < test_acc:
            best_test_acc = test_acc

        print(
                "Epoch: %d LR: %.5f Loss %.4f, Train Acc %.4f, Val Acc %.4f (Best %.4f), Test Acc %.4f (Best %.4f)"
                % (
                    epoch,
                    optimizer.param_groups[0]["lr"],
                    loss.item(),
                    train_acc.item(),
                    val_acc.item(),
                    best_val_acc.item(),
                    test_acc.item(),
                    best_test_acc.item(),
                )
            )

torch.save({
            'epoch': epoch+1,
            'model_state_dict': model.state_dict(),
            'pred_state_dict': pred.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': loss,
            }, '/nfs/turbo/umms-drjieliu/proj/medlineKG/data/glkb_processed_data/openai_embeddings/dgl_graphs/glkb_hier.l2_final_model.pth') # '/nfs/turbo/umms-drjieliu/proj/prompt_re/kg_embeddings/outputs/l2_final_model.pth'

# torch.save({
#             'epoch': best_epoch,
#             'model_state_dict': best_model,
#             'pred_state_dict': best_epoch,
#             'optimizer_state_dict': best_opt,
#             'loss': best_loss,
#             }, 'outputs/l2_best_model.pth')


# infer
model.eval()
h, ntype, etype = model(g, 'emb')
# torch.save(h, '/nfs/turbo/umms-drjieliu/proj/prompt_re/kg_embeddings/outputs/term_embeddings_multilabel.pk') # 

node_emb = h.detach().numpy()
node_index_mappings = json.load(open('/nfs/turbo/umms-drjieliu/proj/medlineKG/data/glkb_processed_data/openai_embeddings/dgl_graphs/glkb_hier.node_index_mappings.json')) # '/nfs/turbo/umms-drjieliu/proj/prompt_re/kg_embeddings/data/node_index_mappings.json'
dic = {k:node_emb[v] for k, v in node_index_mappings.items()}
pickle.dump(dic, open('/nfs/turbo/umms-drjieliu/proj/medlineKG/data/glkb_processed_data/openai_embeddings/dgl_graphs/glkb_hier.term_embeddings_multilabel.pk', 'wb')) # '/nfs/turbo/umms-drjieliu/proj/medlineKG/data/glkb_processed_data/hgt_semantic_embeddings.pk'