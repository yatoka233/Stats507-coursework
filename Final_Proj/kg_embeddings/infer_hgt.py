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
import numpy as np
import pickle
import json

# args
## model args
n_epoch = 100
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
print(torch.cuda.is_available())
# device = torch.device("cpu")
print(device)
## dataloader args
num_workers = 0
batch_size = 1000000
train_seeds = 42
neg_sample_rate = 20
total_steps = 1000000
## other parameters
thres = 0.5
# FILE = '/nfs/turbo/umms-drjieliu/proj/prompt_re/kg_embeddings/data/semantic_graph_10part.bin'
# FILE = '/nfs/turbo/umms-drjieliu/proj/prompt_re/kg_embeddings/data/semantic_graph_nodetypes.bin'
FILE = '/nfs/turbo/umms-drjieliu/proj/medlineKG/data/glkb_processed_data/openai_embeddings/dgl_graphs/glkb_hier.bin'

data, _ = dgl.load_graphs(FILE)
G = data[-1]
# args
## model args
n_epoch = 200
n_hid = 256
# n_inp = 256
n_inp = 1536 # 768
clip = 1.0
max_lr =1e-3
n_out = 400
n_layers = 2 # 4
n_heads = 4 # 7
head_size = 10
## dataloader args
num_workers = 0
batch_size = 100000
train_seeds = 42
neg_sample_rate = 1
g = dgl.to_homogeneous(G, ndata=['emb'])

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
# pred = MLPPredictor(n_out, len(G.etypes) + 1)

optimizer = torch.optim.AdamW(model.parameters())
scheduler = torch.optim.lr_scheduler.OneCycleLR(
    optimizer, total_steps=100000, max_lr=max_lr
)

# load from pretrained
# state_dict = torch.load('/nfs/turbo/umms-drjieliu/proj/prompt_re/kg_embeddings/outputs/l2_final_model.pth')
# model.load_state_dict(state_dict['model_state_dict'])
# pred.load_state_dict(state_dict['pred_state_dict'])
# state_dict = torch.load('/nfs/turbo/umms-drjieliu/proj/prompt_re/kg_embeddings/outputs/lp+nc_final_model.pth')
# model.load_state_dict(state_dict['model_state_dict'])
# state_dict = torch.load('/nfs/turbo/umms-drjieliu/proj/prompt_re/kg_embeddings/outputs/l2_binary_final_model.pth')
# model.load_state_dict(state_dict['model_state_dict'])
# state_dict = torch.load('/nfs/turbo/umms-drjieliu/proj/prompt_re/kg_embeddings/outputs/multilabel_model1010.pth', map_location=device)
# model.load_state_dict(state_dict['model_state_dict'])
state_dict = torch.load('/nfs/turbo/umms-drjieliu/proj/medlineKG/data/glkb_processed_data/openai_embeddings/dgl_graphs/glkb_hier.l2_binary_final_model.pth', map_location=device)
model.load_state_dict(state_dict['model_state_dict'])

# inference
model.eval()
h, ntype, etype = model(g, 'emb')
# torch.save(h, '/nfs/turbo/umms-drjieliu/proj/prompt_re/kg_embeddings/outputs/term_embeddings_multilabel.pk') # 

node_emb = h.detach().numpy()
node_index_mappings = json.load(open('/nfs/turbo/umms-drjieliu/proj/medlineKG/data/glkb_processed_data/openai_embeddings/dgl_graphs/glkb_hier.node_index_mappings.json')) # '/nfs/turbo/umms-drjieliu/proj/prompt_re/kg_embeddings/data/node_index_mappings.json'
dic = {k:node_emb[v] for k, v in node_index_mappings.items()}
pickle.dump(dic, open('/nfs/turbo/umms-drjieliu/proj/medlineKG/data/glkb_processed_data/openai_embeddings/dgl_graphs/glkb_hier.term_embeddings.pk', 'wb')) # '/nfs/turbo/umms-drjieliu/proj/medlineKG/data/glkb_processed_data/hgt_semantic_embeddings.pk'