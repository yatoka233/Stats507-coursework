from argparse import ArgumentParser
from json import decoder
from logging import debug
import pytorch_lightning as pl
import json
import torch
import torch.nn as nn
import torch.nn.functional as F
# Hide lines below until Lab 5
import numpy as np
# Hide lines above until Lab 5


class kg_model(nn.Module):
    # hgt is a trainable model
    def __init__(self, hidden_size, num_relation, hgt=None):
        super().__init__()

        self.hgt = hgt
        self.hidden_size = hidden_size
        self.num_relation = num_relation
        self.node2idx = json.load(open("/nfs/turbo/umms-drjieliu/proj/prompt_re/kg_embeddings/data/node_index_mappings.json"))
        
        if hgt is None:
            # /nfs/turbo/umms-drjieliu/proj/prompt_re/kg_embeddings/outputs/term_embeddings.pk
            self.node_embeddings = torch.load("/nfs/turbo/umms-drjieliu/proj/prompt_re/kg_embeddings/outputs/term_embeddings.pk")
            self.emb_size = self.node_embeddings.shape[1]    
        else:
            pass
        
        self.MLP = nn.Sequential(
                nn.Linear(hidden_size * 3 + self.emb_size, hidden_size * 4),
                # nn.LayerNorm(self.model.config.hidden_size),
                nn.ReLU(),
                nn.Linear(hidden_size * 4 + self.emb_size, num_relation),
            )
        
        self.align = nn.Sequential(
                nn.Linear(hidden_size, self.emb_size),
            )
        
    

