import pytorch_lightning as L
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

class LitHGT(L.LightningModule):
    def __init__(self, FILE):
        super().__init__()

        ## model args
        self.n_hid = 256
        self.n_inp = 768
        self.clip = 1.0
        self.max_lr =1e-3
        self.n_out = 30
        self.n_layers = 2
        self.n_heads = 4
        self.head_size = 10
        if torch.cuda.is_available():
            # self.device = torch.device("cuda")
            device = torch.device("cuda")
        else:
            # self.device = torch.device("cpu")
            device = torch.device("cpu")
        self.neg_sample_rate = 20

        # load data
        data, _ = dgl.load_graphs(FILE)
        G = data[-1]
        G.edata['train_mask'] = split_graph_mask(G)
        G.ndata['train_mask'] = split_node_mask(G.num_nodes())
        g = dgl.to_homogeneous(G, ndata=['emb', 'labels', 'train_mask'], edata=['train_mask'])

        node_dict = {}
        edge_dict = {}
        for ntype in G.ntypes:
            node_dict[ntype] = len(node_dict)
        for etype in G.etypes:
            edge_dict[etype] = len(edge_dict)
            G.edges[etype].data["id"] = (
                torch.ones(G.num_edges(etype), dtype=torch.long) * edge_dict[etype]
            )


        self.model = HGT_built_in_sample(
        node_dict,
        edge_dict,
        self.n_inp,
        self.n_hid,
        self.n_layers,
        self.n_heads,
        self.head_size,
        self.n_out
        ).to(device)
        self.edge_pred = MLPPredictor(self.n_out, len(G.etypes) + 1, device=device)

        # generate dataloader
        train_eid_dict = {
            etype: g.edges(etype=etype, form='eid')
            for etype in g.canonical_etypes}
        sampler = dgl.dataloading.MultiLayerFullNeighborSampler(self.n_layers)
        sampler = dgl.dataloading.as_edge_prediction_sampler(
            sampler, negative_sampler=dgl.dataloading.negative_sampler.Uniform(self.neg_sample_rate))
        self.dataloader = dgl.dataloading.DataLoader(
            g,
            train_eid_dict,
            sampler,
            batch_size=1000000,
            shuffle=True,
            drop_last=False,
            # pin_memory=True,
            num_workers=0)
    
    def forward(self, g):
        h, _, _ = self.model(g, 'emb')
        return h
    
    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters())
        return optimizer
    
    def training_step(self):
        input_nodes, positive_graph, negative_graph, blocks = next(self.dataloader)
        h, _, _ = self.model(blocks, 'emb')
        pos_score = self.edge_pred(positive_graph, h)
        neg_score = self.edge_pred(negative_graph, h)
        loss = compute_loss_multilabel(pos_score, neg_score, positive_graph.edata['_TYPE'])
        return loss

    def validation_step(self):
        input_nodes, positive_graph, negative_graph, blocks = next(self.dataloader)
        h, _, _ = self.model(blocks, 'emb')
        pos_score = self.edge_pred(positive_graph, h)
        neg_score = self.edge_pred(negative_graph, h)
        score = self.node_pred(h)
        loss = compute_loss_multilabel(pos_score, neg_score, positive_graph.edata['_TYPE'])
        return loss
    

model = LitHGT(FILE='/nfs/turbo/umms-drjieliu/proj/prompt_re/kg_embeddings/data/semantic_graph_nodetypes.bin')
model.train()
