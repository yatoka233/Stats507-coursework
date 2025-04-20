from utils import *
import dgl
import dgl.function as fn

import torch
import torch.nn as nn
import torch.nn.functional as F


class HGT_built_in_sample(nn.Module):
    def __init__(
        self,
        node_dict,
        edge_dict,
        n_inp,
        n_hid,
        n_layers,
        n_heads,
        head_size,
        n_out,
        use_norm=True,
        device=torch.device("cpu")
    ):
        super(HGT_built_in_sample, self).__init__()
        self.node_dict = node_dict
        self.edge_dict = edge_dict
        self.gcs = nn.ModuleList()
        self.n_inp = n_inp
        self.n_hid = head_size * n_heads
        self.head_size = head_size
        self.n_layers = n_layers
        self.n_out = n_out
        self.adapt_ws = nn.ModuleList()
        self.device = device
        for t in range(len(node_dict)):
            self.adapt_ws.append(nn.Linear(n_inp, self.n_hid))
        for _ in range(n_layers):
            self.gcs.append(
                dgl.nn.pytorch.conv.HGTConv(
                self.n_hid,
                head_size,
                n_heads,
                len(node_dict.keys()),
                len(edge_dict.keys()),
                use_norm = use_norm)
            )
        self.out = nn.Linear(head_size * n_heads, n_out)

    def prepare_block_feature(self, block, ndata_key):
        # dst_x = block.dstdata[ndata_key]
        # # dst_type = torch.tensor(node_dict.get(block.dsttypes[0])).repeat(dst_x.shape[0])
        # dst_type = block.dstdata['_TYPE']
        src_x = block.srcdata[ndata_key]
        # src_type = torch.tensor(node_dict.get(block.srctypes[0])).repeat(src_x.shape[0])
        src_type = block.srcdata['_TYPE']
        etype = block.edata['_TYPE']
        # return torch.cat([dst_x, src_x]), torch.cat([dst_type, src_type]), etype
        return src_x, src_type, etype

    def forward(self, blocks, ndata_key):
        if isinstance(blocks, list):
            blocks[0].srcdata['input'] = torch.zeros(blocks[0].num_src_nodes(), self.n_hid).to(self.device)
            types = blocks[0].srcdata['_TYPE']
            for n_id in torch.unique(types):
                blocks[0].srcdata['input'][types==n_id] = F.gelu(self.adapt_ws[n_id](blocks[0].srcdata[ndata_key][types==n_id].float()))
            h, ntype, etype = self.prepare_block_feature(blocks[0], 'input')

            for i in range(self.n_layers):
                if i>0:
                    blocks[i].srcdata['input'] = torch.zeros(blocks[i].num_src_nodes(), self.n_hid).to(self.device)
                    types = blocks[i].srcdata['_TYPE']
                    for n_id in torch.unique(types):
                        blocks[i].srcdata['input'][types==n_id] = F.gelu(self.adapt_ws[n_id](blocks[i].srcdata[ndata_key][types==n_id].float()))
                    _, ntype, etype = self.prepare_block_feature(blocks[i], 'input')
                h = self.gcs[i](blocks[i], x=h, ntype=ntype, etype=etype)
            return self.out(h), blocks[-1].dstdata['_TYPE'], blocks[-1].edata['_TYPE']
        
        elif isinstance(blocks, dgl.DGLGraph):
            blocks.srcdata['input'] = torch.zeros(blocks.num_src_nodes(), self.n_hid).to(self.device)
            types = blocks.srcdata['_TYPE']
            for n_id in torch.unique(types):
                blocks.srcdata['input'][types==n_id] = F.gelu(self.adapt_ws[n_id](blocks.srcdata[ndata_key][types==n_id].float()))
            h, ntype, etype = self.prepare_block_feature(blocks, 'input')

            for i in range(self.n_layers):
                h = self.gcs[i](blocks, x=h, ntype=ntype, etype=etype)
            return self.out(h), blocks.dstdata['_TYPE'], blocks.edata['_TYPE']

class Multilabel_Node_Classifier(nn.Module):
    def __init__(self, h_feats, num_node_types, device=torch.device("cpu")):
        super().__init__()
        self.W1 = nn.Linear(h_feats, int(h_feats/2))
        self.W2 = nn.Linear(int(h_feats/2), num_node_types)
        self.device = device
    
    def forward(self, h):
        return torch.sigmoid(self.W2(F.relu(self.W1(h))))

class MLPPredictor(nn.Module):
    def __init__(self, h_feats, num_edge_types, device=torch.device("cpu")):
        super().__init__()
        self.W1 = nn.Linear(h_feats * 2, h_feats)
        self.W2 = nn.Linear(h_feats, num_edge_types)
        self.device = device

    def apply_edges(self, edges):
        """
        Computes a scalar score for each edge of the given graph.

        Parameters
        ----------
        edges :
            Has three members ``src``, ``dst`` and ``data``, each of
            which is a dictionary representing the features of the
            source nodes, the destination nodes, and the edges
            themselves.

        Returns
        -------
        dict
            A dictionary of new edge features.
        """
        h = torch.cat([edges.src["h"], edges.dst["h"]], 1).to(self.device)
        return {"score": F.softmax(self.W2(F.relu(self.W1(h))), dim=1).squeeze(1)}

    def forward(self, g, h):
        g = g.to(self.device)
        with g.local_scope():
            g.ndata["h"] = h.to(self.device)
            g.apply_edges(self.apply_edges)
            return g.edata["score"]
        
class DotPredictor(nn.Module):
    def forward(self, g, h):
        with g.local_scope():
            g.ndata["h"] = h
            # g.ndata["h"] = h / h.norm(dim=1)[:, None]
            # Compute a new edge feature named 'score' by a dot-product between the
            # source node feature 'h' and destination node feature 'h'.
            g.apply_edges(fn.u_dot_v("h", "h", "score"))
            # u_dot_v returns a 1-element vector for each edge so you need to squeeze it.
            return g.edata["score"][:, 0]

class RotatEPredictor(nn.Module):
    def __init__(self, h_feats):
        super().__init__()
        self.W = nn.Linear(h_feats, h_feats)

    def apply_edges(self, edges):
        """
        Computes a scalar score for each edge of the given graph.

        Parameters
        ----------
        edges :
            Has three members ``src``, ``dst`` and ``data``, each of
            which is a dictionary representing the features of the
            source nodes, the destination nodes, and the edges
            themselves.

        Returns
        -------
        dict
            A dictionary of new edge features.
        """
        scores = torch.sum(self.W(edges.src["h"]) * edges.dst["h"], dim=-1)
        return {"score": scores.squeeze()}

    def forward(self, g, h):
        with g.local_scope():
            g.ndata["h"] = h
            g.apply_edges(self.apply_edges)
            return g.edata["score"]