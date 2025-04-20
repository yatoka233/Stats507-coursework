from argparse import ArgumentParser
from json import decoder
from logging import debug
import pytorch_lightning as pl
import json
import torch
import torch.nn as nn
import torch.nn.functional as F
# Hide lines below until Lab 5
import wandb
import numpy as np
# Hide lines above until Lab 5

from .base import BaseLitModel
from .util import f1_eval, compute_f1, acc, f1_score
from transformers.optimization import get_linear_schedule_with_warmup

from functools import partial

import random
import sys
# sys.path.append("/nfs/turbo/umms-drjieliu/proj/prompt_re/kg_embeddings")

from models.models import HGT_built_in_sample
from models.models import MLPPredictor
from utils import split_graph_mask
from utils import split_node_mask
from utils import compute_loss_multilabel
import dgl

device = torch.device("cuda")

def mask_hook(grad_input, st, ed):
    mask = torch.zeros((grad_input.shape[0], 1)).type_as(grad_input)
    mask[st: ed] += 1.0  # 只优化id为1～8的token
    # for the speaker unused token12
    mask[1:3] += 1.0
    return grad_input * mask

def multilabel_categorical_crossentropy(y_pred, y_true):
    y_pred = (1 - 2 * y_true) * y_pred
    y_pred_neg = y_pred - y_true * 1e12
    y_pred_pos = y_pred - (1 - y_true) * 1e12
    zeros = torch.zeros_like(y_pred[..., :1])
    y_pred_neg = torch.cat([y_pred_neg, zeros], dim=-1)
    y_pred_pos = torch.cat([y_pred_pos, zeros], dim=-1)
    neg_loss = torch.logsumexp(y_pred_neg, dim=-1)
    pos_loss = torch.logsumexp(y_pred_pos, dim=-1)
    return (neg_loss + pos_loss).mean()

# class AMSoftmax(nn.Module):
#     def __init__(self,
#                  in_feats,
#                  n_classes=10,
#                  m=0.35,
#                  s=30):
#         super(AMSoftmax, self).__init__()
#         self.m = m
#         self.s = s
#         self.in_feats = in_feats
#         self.W = torch.nn.Linear(in_feats, n_classes)
#         # self.W = torch.nn.Parameter(torch.randn(in_feats, n_classes), requires_grad=True)
#         self.ce = nn.CrossEntropyLoss()
#         # nn.init.xavier_normal_(self.W, gain=1)

#     def forward(self, x, lb):
#         assert x.size()[0] == lb.size()[0]
#         assert x.size()[1] == self.in_feats
#         x_norm = torch.norm(x, p=2, dim=1, keepdim=True).clamp(min=1e-12)
#         x_norm = torch.div(x, x_norm)
#         w_norm = torch.norm(self.W, p=2, dim=0, keepdim=True).clamp(min=1e-12)
#         w_norm = torch.div(self.W, w_norm)
#         costh = torch.mm(x_norm, w_norm)
#         # print(x_norm.shape, w_norm.shape, costh.shape)
#         lb_view = lb.view(-1, 1)
#         if lb_view.is_cuda: lb_view = lb_view.cpu()
#         delt_costh = torch.zeros(costh.size()).scatter_(1, lb_view, self.m)
#         if x.is_cuda: delt_costh = delt_costh.cuda()
#         costh_m = costh - delt_costh
#         costh_m_s = self.s * costh_m
#         loss = self.ce(costh_m_s, lb)
#         return loss, costh_m_s

# class AMSoftmax(nn.Module):

#     def __init__(self, in_features, out_features, s=30.0, m=0.35):
#         '''
#         AM Softmax Loss
#         '''
#         super().__init__()
#         self.s = s
#         self.m = m
#         self.in_features = in_features
#         self.out_features = out_features
#         self.fc = nn.Linear(in_features, out_features, bias=False)

#     def forward(self, x, labels):
#         '''
#         input shape (N, in_features)
#         '''
#         assert len(x) == len(labels)
#         assert torch.min(labels) >= 0
#         assert torch.max(labels) < self.out_features
#         for W in self.fc.parameters():
#             W = F.normalize(W, dim=1)

#         x = F.normalize(x, dim=1)

#         wf = self.fc(x)
#         numerator = self.s * (torch.diagonal(wf.transpose(0, 1)[labels]) - self.m)
#         excl = torch.cat([torch.cat((wf[i, :y], wf[i, y+1:])).unsqueeze(0) for i, y in enumerate(labels)], dim=0)
#         denominator = torch.exp(numerator) + torch.sum(torch.exp(self.s * excl), dim=1)
#         L = numerator - torch.log(denominator)
#         return -torch.mean(L)


class BertLitModel(BaseLitModel):
    """
    use AutoModelForMaskedLM, and select the output by another layer in the lit model
    """
    def __init__(self, model, args, tokenizer):
        super().__init__(model, args)
        self.tokenizer = tokenizer
        
        with open(f"{args.data_dir}/rel2id.json","r") as file:
            rel2id = json.load(file)
        
        Na_num = 0
        for k, v in rel2id.items():
            if k == "NA" or k == "no_relation" or k == "Other":
                Na_num = v
                break
        num_relation = len(rel2id)
        # init loss function
        self.loss_fn = multilabel_categorical_crossentropy if "dialogue" in args.data_dir else nn.CrossEntropyLoss()
        # self.loss_fn = AMSoftmax(self.model.config.hidden_size, num_relation)
        # ignore the no_relation class to compute the f1 score
        self.eval_fn = f1_eval if "dialogue" in args.data_dir else partial(f1_score, rel_num=num_relation, na_num=Na_num)
        self.best_f1 = 0
        self.t_lambda = args.t_lambda
        
        self.label_st_id = tokenizer("[class1]", add_special_tokens=False)['input_ids'][0]  # label start id
        self.tokenizer = tokenizer

        self.biored = args.biored
        self.use_entity_loss = False
        if args.use_entity_loss:
            self.use_entity_loss = True
            self.ei_list = ["GeneOrGeneProduct", "DiseaseOrPhenotypicFeature", "OrganismTaxon", "SequenceVariant", "ChemicalEntity", "CellLine"]
            self.ei_linear = nn.Linear(self.model.config.hidden_size, len(self.ei_list))
            self.e_lambda = args.e_lambda

            torch.nn.init.xavier_uniform(self.ei_linear.weight)
            torch.nn.init.zeros_(self.ei_linear.bias)
        

        self.use_head = False
        if args.use_head:
            self.use_head = True
            # MLP
            self.MLP = nn.Sequential(
                nn.Linear(self.model.config.hidden_size * 3, self.model.config.hidden_size * 4),
                # nn.LayerNorm(self.model.config.hidden_size),
                nn.ReLU(),
                nn.Linear(self.model.config.hidden_size * 4, num_relation),
            )
            
            for layer in self.MLP:
                if isinstance(layer, nn.Linear):
                    torch.nn.init.xavier_uniform(layer.weight)
                    torch.nn.init.zeros_(layer.bias)

            # transformer head
            encoder_layer = nn.TransformerEncoderLayer(d_model=self.model.config.hidden_size, nhead=8)
            self.head = nn.TransformerEncoder(encoder_layer, num_layers=1)

            for name, param in self.head.named_parameters():
                if 'weight' in name and param.data.dim() == 2:
                    nn.init.kaiming_uniform_(param)


        self.use_kg = False
        if args.use_kg:
            self.use_kg = True
            self.use_kg_mlp = args.use_kg_mlp

            # load data
            FILE='/nfs/turbo/umms-drjieliu/proj/prompt_re/kg_embeddings/data/semantic_graph_nodetypes.bin'
            data, _ = dgl.load_graphs(FILE)
            G = data[-1]
            G.edata['train_mask'] = split_graph_mask(G)
            G.ndata['train_mask'] = split_node_mask(G.num_nodes())
            g = dgl.to_homogeneous(G, ndata=['emb', 'labels', 'train_mask'], edata=['train_mask'])
            self.g = g.to(device)

            node_dict = {}
            edge_dict = {}
            for ntype in G.ntypes:
                node_dict[ntype] = len(node_dict)
            for etype in G.etypes:
                edge_dict[etype] = len(edge_dict)
                G.edges[etype].data["id"] = (
                    torch.ones(G.num_edges(etype), dtype=torch.long) * edge_dict[etype]
                )

            ## load model
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
            self.hgt = HGT_built_in_sample(
                node_dict,
                edge_dict,
                n_inp,
                n_hid,
                n_layers,
                n_heads,
                head_size,
                n_out,
                device=device
                ).to(device)
            ## load weights
            state_dict = torch.load('/nfs/turbo/umms-drjieliu/proj/prompt_re/kg_embeddings/outputs/l2_final_model.pth')
            self.hgt.load_state_dict(state_dict['model_state_dict'])

            ## link prediction
            ## dataloader args
            num_workers = 0
            batch_size = 1000000
            train_seeds = 42
            neg_sample_rate = 20
            total_steps = 1000000
            train_eid_dict = {
                etype: g.edges(etype=etype, form='eid')
                for etype in g.canonical_etypes}
            sampler = dgl.dataloading.MultiLayerFullNeighborSampler(n_layers)
            sampler = dgl.dataloading.as_edge_prediction_sampler(
                sampler, negative_sampler=dgl.dataloading.negative_sampler.Uniform(neg_sample_rate))
            self.dataloader = dgl.dataloading.DataLoader(
                g,
                train_eid_dict,
                sampler,
                batch_size=batch_size,
                shuffle=True,
                drop_last=False,
                # pin_memory=True,
                num_workers=num_workers,
                device=device)
            ## load link prediction model
            self.edge_pred = MLPPredictor(n_out, len(G.etypes) + 1, device=device)
            self.edge_pred.load_state_dict(state_dict['pred_state_dict'])



            self.node2idx = json.load(open("/nfs/turbo/umms-drjieliu/proj/prompt_re/kg_embeddings/data/node_index_mappings.json"))
            

            # /nfs/turbo/umms-drjieliu/proj/prompt_re/kg_embeddings/outputs/term_embeddings.pk
            self.node_embeddings = torch.load("/nfs/turbo/umms-drjieliu/proj/prompt_re/kg_embeddings/outputs/term_embeddings_lp+nc.pk").cuda()
            self.kg_emb_size = self.node_embeddings.shape[1] 

            
            # for classification
            self.kg_MLP = nn.Sequential(
                nn.Linear(self.model.config.hidden_size * 3 + self.kg_emb_size * 2, self.model.config.hidden_size * 4),
                # nn.LayerNorm(self.model.config.hidden_size),
                nn.ReLU(),
                nn.Linear(self.model.config.hidden_size * 4, num_relation),
            )
            for layer in self.kg_MLP:
                if isinstance(layer, nn.Linear):
                    torch.nn.init.xavier_uniform(layer.weight)
                    torch.nn.init.zeros_(layer.bias)

            # for alignment loss
            self.align = nn.Sequential(
                nn.Linear(self.model.config.hidden_size, self.kg_emb_size),
            )
            for layer in self.align:
                if isinstance(layer, nn.Linear):
                    torch.nn.init.xavier_uniform(layer.weight)
                    torch.nn.init.zeros_(layer.bias)
            
    
        self._init_label_word()

        
        # with torch.no_grad():
        #     self.loss_fn.fc.weight = nn.Parameter(self.model.get_output_embeddings().weight[self.label_st_id:self.label_st_id+num_relation])
            # self.loss_fn.fc.bias = nn.Parameter(self.model.get_output_embeddings().bias[self.label_st_id:self.label_st_id+num_relation])

    def _init_label_word(self, ):
        args = self.args
        # ./dataset/dataset_name
        dataset_name = args.data_dir.split("/")[1]
        model_name_or_path = args.model_name_or_path.split("/")[-1]
        label_path = f"./dataset/{model_name_or_path}_{dataset_name}.pt" # splited label word idx
        # [num_labels, num_tokens], ignore the unanswerable
        if "dialogue" in args.data_dir:
            label_word_idx = torch.load(label_path)[:-1]
        else:
            label_word_idx = torch.load(label_path)
        
        num_labels = len(label_word_idx)    # number of labels
        
        self.model.resize_token_embeddings(len(self.tokenizer))
        
        with torch.no_grad():
            word_embeddings = self.model.get_input_embeddings()
            # label index
            continous_label_word = [a[0] for a in self.tokenizer([f"[class{i}]" for i in range(1, num_labels+1)], add_special_tokens=False)['input_ids']]
            
            # for abaltion study
            if self.args.init_answer_words:
                print("init answer words")
                if self.args.init_answer_words_by_one_token:
                    for i, idx in enumerate(label_word_idx):
                        word_embeddings.weight[continous_label_word[i]] = word_embeddings.weight[idx][-1]
                else:
                    for i, idx in enumerate(label_word_idx):
                        word_embeddings.weight[continous_label_word[i]] = torch.mean(word_embeddings.weight[idx], dim=0)
                # word_embeddings.weight[continous_label_word[i]] = self.relation_embedding[i]
            
            if self.args.init_type_words:
                print("init type words")
                so_word = [a[0] for a in self.tokenizer(["[obj]","[sub]"], add_special_tokens=False)['input_ids']]
                type_list = ["GeneOrGeneProduct", "DiseaseOrPhenotypicFeature", "OrganismTaxon", "SequenceVariant", "ChemicalEntity", "CellLine"]
                e_word = [a[0] for a in self.tokenizer([f"[e{i}]" for i in range(1, 7)], add_special_tokens=False)['input_ids']]
                if self.biored:
                    meaning_word = [a[0] for a in self.tokenizer(["GeneOrGeneProduct", "DiseaseOrPhenotypicFeature", "OrganismTaxon", "SequenceVariant", "ChemicalEntity", "CellLine"], add_special_tokens=False)['input_ids']]
                    # meaning_word = [a[0] for a in self.tokenizer(["Gene Or Gene Product", "Disease Or Phenotypic Feature", "Organism Taxon", "Sequence Variant", "Chemical Entity", "CellLine"], add_special_tokens=False)['input_ids']]
                else:
                    meaning_word = [a[0] for a in self.tokenizer(["person","organization", "location", "date", "country"], add_special_tokens=False)['input_ids']]
            
                for i, idx in enumerate(so_word):
                    word_embeddings.weight[so_word[i]] = torch.mean(word_embeddings.weight[meaning_word], dim=0)
                    # # random init
                    # word_embeddings.weight[so_word[i]] = torch.randn(word_embeddings.weight[meaning_word].shape[1])
                for i, idx in enumerate(e_word):
                    word_embeddings.weight[e_word[i]] = torch.mean(word_embeddings.weight[self.tokenizer(type_list[i], add_special_tokens=False)['input_ids']], dim=0)

            assert torch.equal(self.model.get_input_embeddings().weight, word_embeddings.weight)
            assert torch.equal(self.model.get_input_embeddings().weight, self.model.get_output_embeddings().weight)
        
        self.word2label = continous_label_word # a continous list

            
                
    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):  # pylint: disable=unused-argument
        if len(batch) == 5:
            input_ids, attention_mask, labels, so, ei = batch
            result = self.model(input_ids, attention_mask, return_dict=True, output_hidden_states=True)
        elif len(batch) == 6:
            input_ids, attention_mask, token_type_ids, labels, so, ei = batch
            result = self.model(input_ids, attention_mask, token_type_ids, return_dict=True, output_hidden_states=True)
        elif self.use_kg:
            input_ids, attention_mask, token_type_ids, labels, so, ei, nodes1, nodes2 = batch
            result = self.model(input_ids, attention_mask, token_type_ids, return_dict=True, output_hidden_states=True)
            if self.hgt is not None:
                # self.hgt.train()
                node_embeddings, ntype, etype = self.hgt(self.g, "emb")
                # node_embeddings = node_embeddings.detach()
                kg_emb_size = node_embeddings.shape[1]
            else:
                node_embeddings = self.node_embeddings
                kg_emb_size = self.kg_emb_size
            
            ## link prediction loss
            for input_nodes, positive_graph, negative_graph, blocks in self.dataloader:
                blocks = [b.to(device) for b in blocks]
                positive_graph = positive_graph.to(torch.device(device))
                negative_graph = negative_graph.to(torch.device(device))

                train_idx = positive_graph.edata['train_mask']==0
                val_idx = positive_graph.edata['train_mask']==1
                test_idx = positive_graph.edata['train_mask']==2

                h, ntype, etype = self.hgt(blocks, 'emb')
                ## filter target edge type
                pos_score = self.edge_pred(positive_graph, h)
                neg_score = self.edge_pred(negative_graph, h)
                
                lp_loss = compute_loss_multilabel(pos_score[train_idx], neg_score, positive_graph.edata['_TYPE'][train_idx])

        logits = result.logits  # [bs, seq_len, vocab_size]
        output_embedding = result.hidden_states[-1]

        if self.use_head:
            logits = self.mlp_pvp(output_embedding, so, input_ids)    # [bs, num_labels]
        elif self.use_kg:
            if self.use_kg_mlp:
                logits = self.mlp_kg_pvp(node_embeddings, kg_emb_size, output_embedding, so, input_ids, nodes1, nodes2)    # [bs, num_labels]
            else:
                logits = self.pvp(logits, input_ids)
            al_loss = self.align_loss(node_embeddings, output_embedding, so, input_ids, nodes1, nodes2)
        else:
            logits = self.pvp(logits, input_ids)    # [bs, num_labels]
        # logits = self.model.roberta(input_ids, attention_mask).last_hidden_state
        # loss = self.get_loss(logits, input_ids, labels)

        ke_loss = self.ke_loss(output_embedding, labels, so, input_ids)

        if self.args.use_entity_loss:
            ei_loss = self.en_loss(output_embedding, ei, so, input_ids)
            loss = self.loss_fn(logits, labels) + self.t_lambda * ke_loss + self.e_lambda * ei_loss
            self.log("Train/loss", loss)
            self.log("Train/ke_loss", ke_loss)
            self.log("Train/ei_loss", ei_loss)
        elif self.use_kg:
            loss = self.loss_fn(logits, labels) + \
                    self.t_lambda * ke_loss + \
                    self.args.align_lambda * al_loss +\
                    self.args.lp_lambda * lp_loss
            self.log("Train/loss", loss)
            self.log("Train/ke_loss", ke_loss)
            self.log("Train/align_loss", al_loss)
        else:
            loss = self.loss_fn(logits, labels) + self.t_lambda * ke_loss
            # loss = self.loss_fn(logits, labels)
            self.log("Train/loss", loss)
            self.log("Train/ke_loss", ke_loss)
        return loss
    
    def get_loss(self, logits, input_ids, labels):
        _, mask_idx = (input_ids == self.tokenizer.mask_token_id).nonzero(as_tuple=True)
        bs = input_ids.shape[0]
        mask_output = logits[torch.arange(bs), mask_idx]
        
        loss = self.loss_fn(mask_output, labels)
        return loss


    def validation_step(self, batch, batch_idx):  # pylint: disable=unused-argument
        if len(batch) == 5:
            input_ids, attention_mask, labels, so, ei = batch
            result = self.model(input_ids, attention_mask, return_dict=True, output_hidden_states=True)
        elif len(batch) == 6:
            input_ids, attention_mask, token_type_ids, labels, so, ei = batch
            result = self.model(input_ids, attention_mask, token_type_ids, return_dict=True, output_hidden_states=True)
        elif self.use_kg:
            input_ids, attention_mask, token_type_ids, labels, so, ei, nodes1, nodes2 = batch
            result = self.model(input_ids, attention_mask, token_type_ids, return_dict=True, output_hidden_states=True)
            if self.hgt is not None:
                node_embeddings, ntype, etype = self.hgt(self.g, "emb")
                # node_embeddings = node_embeddings.detach()
                kg_emb_size = node_embeddings.shape[1]
            else:
                node_embeddings = self.node_embeddings
                kg_emb_size = self.kg_emb_size

        logits = result.logits
        # logits = self.model.roberta(input_ids, attention_mask).last_hidden_state
        # loss = self.loss_fn(logits, labels)
        output_embedding = result.hidden_states[-1]

        if self.use_head:
            logits = self.mlp_pvp(output_embedding, so, input_ids)    # [bs, num_labels]
        elif self.use_kg:
            if self.use_kg_mlp:
                logits = self.mlp_kg_pvp(node_embeddings, kg_emb_size, output_embedding, so, input_ids, nodes1, nodes2)    # [bs, num_labels]
            else:
                logits = self.pvp(logits, input_ids)
            al_loss = self.align_loss(node_embeddings, output_embedding, so, input_ids, nodes1, nodes2)
        else:
            logits = self.pvp(logits, input_ids)    # [bs, num_labels]
        loss = self.loss_fn(logits, labels)
        self.log("Eval/loss", loss)
        return {"eval_logits": logits.detach().cpu().numpy(), "eval_labels": labels.detach().cpu().numpy()}
    
    def validation_epoch_end(self, outputs) -> None:
        logits = np.concatenate([o["eval_logits"] for o in outputs])
        labels = np.concatenate([o["eval_labels"] for o in outputs])

        f1 = self.eval_fn(logits, labels)['f1']
        f1_by_relation = self.eval_fn(logits, labels)['f1_by_relation']
        # print("f1:", f1)
        print("f1_by_relation:", f1_by_relation)
        # write f1 to txt
        with open("output" + "/f1.txt", "a+") as f:
            f.write(str(f1) + "\n")

        self.log("Eval/f1", f1)
        self.log("Eval/f1_by_relation", f1_by_relation)
        if f1 > self.best_f1:
            self.best_f1 = f1
            print("best f1:", self.best_f1)
        self.log("Eval/best_f1", self.best_f1, prog_bar=True, on_epoch=True)

    def test_step(self, batch, batch_idx):  # pylint: disable=unused-argument
        if len(batch) == 5:
            input_ids, attention_mask, labels, so, ei = batch
            result = self.model(input_ids, attention_mask, return_dict=True, output_hidden_states=True)
        elif len(batch) == 6:
            input_ids, attention_mask, token_type_ids, labels, so, ei = batch
            result = self.model(input_ids, attention_mask, token_type_ids, return_dict=True, output_hidden_states=True)
        elif self.use_kg:
            input_ids, attention_mask, token_type_ids, labels, so, ei, nodes1, nodes2 = batch
            result = self.model(input_ids, attention_mask, token_type_ids, return_dict=True, output_hidden_states=True)
            if self.hgt is not None:
                node_embeddings, ntype, etype = self.hgt(self.g, "emb")
                # node_embeddings = node_embeddings.detach()
                kg_emb_size = node_embeddings.shape[1]
            else:
                node_embeddings = self.node_embeddings
                kg_emb_size = self.kg_emb_size

        logits = result.logits
        output_embedding = result.hidden_states[-1]

        if self.use_head:
            logits = self.mlp_pvp(output_embedding, so, input_ids)    # [bs, num_labels]
        elif self.use_kg:
            if self.use_kg_mlp:
                logits = self.mlp_kg_pvp(node_embeddings, kg_emb_size, output_embedding, so, input_ids, nodes1, nodes2)    # [bs, num_labels]
            else:
                logits = self.pvp(logits, input_ids)
            al_loss = self.align_loss(node_embeddings, output_embedding, so, input_ids, nodes1, nodes2)
        else:
            logits = self.pvp(logits, input_ids)    # [bs, num_labels]
        return {"test_logits": logits.detach().cpu().numpy(), "test_labels": labels.detach().cpu().numpy()}

    def test_epoch_end(self, outputs) -> None:
        logits = np.concatenate([o["test_logits"] for o in outputs])
        labels = np.concatenate([o["test_labels"] for o in outputs])

        f1 = self.eval_fn(logits, labels)['f1']
        f1_by_relation = self.eval_fn(logits, labels)['f1_by_relation']
        self.log("Test/f1", f1)
        self.log("Test/f1_by_relation", f1_by_relation)
        print("f1_by_relation:", f1_by_relation)
        


    @staticmethod
    def add_to_argparse(parser):
        BaseLitModel.add_to_argparse(parser)
        parser.add_argument("--t_lambda", type=float, default=0.01, help="")
        parser.add_argument("--t_gamma", type=float, default=0.3, help="")
        return parser
        
    def pvp(self, logits, input_ids):
        # convert the [batch_size, seq_len, vocab_size] => [batch_size, num_labels]
        #! hard coded
        _, mask_idx = (input_ids == self.tokenizer.mask_token_id).nonzero(as_tuple=True)
        bs = input_ids.shape[0] # batch size
        mask_output = logits[torch.arange(bs), mask_idx]
        assert mask_idx.shape[0] == bs, "only one mask in sequence!"
        final_output = mask_output[:,self.word2label]
        
        return final_output
        
    def ke_loss(self, logits, labels, so, input_ids):
        subject_embedding = []
        object_embedding = []
        neg_subject_embedding = []
        neg_object_embedding = []
        bsz = logits.shape[0]
        for i in range(bsz):
            # x['so'] =[sub_st, sub_ed, obj_st, obj_ed]
            subject_embedding.append(torch.mean(logits[i, so[i][0]:so[i][1]], dim=0))
            object_embedding.append(torch.mean(logits[i, so[i][2]:so[i][3]], dim=0))

            # virtual token position
            if False:
                subject_embedding.append(torch.mean(logits[i, [so[i][0]-1, so[i][1]+1]], dim=0))
                object_embedding.append(torch.mean(logits[i, [so[i][2]-1, so[i][3]+1]], dim=0))

            # random select the neg samples
            st_sub = random.randint(1, logits[i].shape[0] - 6)
            span_sub = random.randint(1, 5)
            st_obj = random.randint(1, logits[i].shape[0] - 6)
            span_obj = random.randint(1, 5)
            neg_subject_embedding.append(torch.mean(logits[i, st_sub:st_sub+span_sub], dim=0))
            neg_object_embedding.append(torch.mean(logits[i, st_obj:st_obj+span_obj], dim=0))
            
        subject_embedding = torch.stack(subject_embedding)
        object_embedding = torch.stack(object_embedding)
        neg_subject_embedding = torch.stack(neg_subject_embedding)
        neg_object_embedding = torch.stack(neg_object_embedding)
        # trick , the relation ids is concated, 


        _, mask_idx = (input_ids == self.tokenizer.mask_token_id).nonzero(as_tuple=True)
        mask_output = logits[torch.arange(bsz), mask_idx]
        mask_relation_embedding = mask_output
        real_relation_embedding = self.model.get_output_embeddings().weight[labels+self.label_st_id]
        
        d_1 = torch.norm(subject_embedding + mask_relation_embedding - object_embedding, p=2) / bsz
        d_2 = torch.norm(neg_subject_embedding + real_relation_embedding - neg_object_embedding, p=2) / bsz
        f = torch.nn.LogSigmoid()
        loss = -1.*f(self.args.t_gamma - d_1) - f(d_2 - self.args.t_gamma)
        
        return loss
    
    def mlp_pvp(self, hidden_states, so, input_ids):
        # use another transformer layer or not
        # hidden_states = self.head(hidden_states)
        
        # convert the [batch_size, seq_len, hidden] => [batch_size, num_labels]
        _, mask_idx = (input_ids == self.tokenizer.mask_token_id).nonzero(as_tuple=True)
        bs = input_ids.shape[0] # batch size
        mask_output = hidden_states[torch.arange(bs), mask_idx]
        assert mask_idx.shape[0] == bs, "only one mask in sequence!"

        subject_embeddings = []
        object_embeddings = []
        for i in range(bs):
            # x['so'] =[sub_st, sub_ed, obj_st, obj_ed]
            subject_embeddings.append(torch.mean(hidden_states[i, so[i][0]:so[i][1]], dim=0))
            object_embeddings.append(torch.mean(hidden_states[i, so[i][2]:so[i][3]], dim=0))
        subject_embeddings = torch.stack(subject_embeddings)
        object_embeddings = torch.stack(object_embeddings)

        cat_output = torch.cat([mask_output, subject_embeddings, object_embeddings], dim=1)
        # cat_output = mask_output
        final_output = self.MLP(cat_output)
        
        return final_output
    
    def mlp_kg_pvp(self, node_embeddings, kg_emb_size, hidden_states, so, input_ids, nodes1, nodes2):
        _, mask_idx = (input_ids == self.tokenizer.mask_token_id).nonzero(as_tuple=True)
        bs = input_ids.shape[0] # batch size
        mask_output = hidden_states[torch.arange(bs), mask_idx]
        assert mask_idx.shape[0] == bs, "only one mask in sequence!"

        subject_embeddings = []
        object_embeddings = []
        for i in range(bs):
            if nodes1[i] == [0]:
                node1_embedding = torch.zeros(kg_emb_size)
            else:
                idx1 = (nodes1[i]-1).cuda()
                node1_embedding = torch.mean(node_embeddings[idx1], dim=0)
            
            if nodes2[i] == [0]:
                node2_embedding = torch.zeros(kg_emb_size)
            else:
                idx2 = (nodes2[i]-1).cuda()
                node2_embedding = torch.mean(node_embeddings[idx2], dim=0)
            
            # x['so'] =[sub_st, sub_ed, obj_st, obj_ed]
            # concat the node embedding
            subject_embeddings.append(torch.cat([torch.mean(hidden_states[i, so[i][0]:so[i][1]], dim=0), node1_embedding], dim=0))
            object_embeddings.append(torch.cat([torch.mean(hidden_states[i, so[i][2]:so[i][3]], dim=0), node2_embedding], dim=0))
        
        subject_embeddings = torch.stack(subject_embeddings)
        object_embeddings = torch.stack(object_embeddings)

        cat_output = torch.cat([mask_output, subject_embeddings, object_embeddings], dim=1)
        # cat_output = mask_output
        final_output = self.kg_MLP(cat_output)
        
        return final_output

    def align_loss(self, node_embeddings, hidden_states, so, input_ids, nodes1, nodes2):
        _, mask_idx = (input_ids == self.tokenizer.mask_token_id).nonzero(as_tuple=True)
        bs = input_ids.shape[0] # batch size
        mask_output = hidden_states[torch.arange(bs), mask_idx]
        assert mask_idx.shape[0] == bs, "only one mask in sequence!"

        subject_embeddings = []
        object_embeddings = []
        node1_embeddings = []
        node2_embeddings = []
        for i in range(bs):
            if nodes1[i] != [0]:
                idx1 = (nodes1[i]-1).cuda()
                node1_embeddings.append(torch.mean(node_embeddings[idx1], dim=0))
                subject_embeddings.append(torch.mean(hidden_states[i, so[i][0]:so[i][1]], dim=0))
            
            if nodes2[i] != [0]:
                idx2 = (nodes2[i]-1).cuda()
                node2_embeddings.append(torch.mean(node_embeddings[idx2], dim=0))
                object_embeddings.append(torch.mean(hidden_states[i, so[i][2]:so[i][3]], dim=0))
        
        subject_embeddings = torch.stack(subject_embeddings)
        object_embeddings = torch.stack(object_embeddings)
        node1_embeddings = torch.stack(node1_embeddings)
        node2_embeddings = torch.stack(node2_embeddings)

        subject_embeddings = self.align(subject_embeddings)
        object_embeddings = self.align(object_embeddings)

        # L2 loss
        loss = torch.norm(subject_embeddings - node1_embeddings, p=2) + torch.norm(object_embeddings - node2_embeddings, p=2)

        return loss

            
    
    def en_loss(self, hidden_states, ei, so, input_ids):
        entity_embeddings = []
        entity_labels = []
        bsz = hidden_states.shape[0]

        for i in range(bsz):
            # x['so'] =[sub_st, sub_ed, obj_st, obj_ed]
            subject_embedding = torch.mean(hidden_states[i, so[i][0]:so[i][1]], dim=0)
            object_embedding = torch.mean(hidden_states[i, so[i][2]:so[i][3]], dim=0)
            entity_embeddings.append(subject_embedding)
            entity_embeddings.append(object_embedding)

            entity_labels.append(ei[i])
        
        entity_embeddings = torch.stack(entity_embeddings)
        entity_labels = torch.cat(entity_labels)

        # print("en shape: ", entity_embeddings.shape, entity_labels.shape)

        entity_feature = self.ei_linear(entity_embeddings)
        loss = self.loss_fn(entity_feature, entity_labels)
        return loss

    def configure_optimizers(self):
        no_decay_param = ["bias", "LayerNorm.weight"]

        if not self.args.two_steps: 
            if self.use_entity_loss:
                parameters = [self.model.named_parameters(), self.ei_linear.named_parameters()]
            elif self.use_head:
                parameters = [self.model.named_parameters(), self.MLP.named_parameters(), self.head.named_parameters()]
            elif self.use_kg:
                parameters = [self.model.named_parameters(), self.hgt.named_parameters(), self.edge_pred.named_parameters(),
                              self.kg_MLP.named_parameters(), self.align.named_parameters()]
            else:
                parameters = self.model.named_parameters()
        else:
            print("only optimize the embedding parameters")
            # model.bert.embeddings.weight
            parameters = [next(self.model.named_parameters())]
        # only optimize the embedding parameters
        if self.use_entity_loss :
            optimizer_group_parameters = [
                {"params": [p for n, p in parameters[0] if not any(nd in n for nd in no_decay_param)], "weight_decay": self.args.weight_decay},
                {"params": [p for n, p in parameters[0] if any(nd in n for nd in no_decay_param)], "weight_decay": 0},
                {"params": [p for n, p in parameters[1] if not any(nd in n for nd in no_decay_param)], "weight_decay": self.args.weight_decay},
                {"params": [p for n, p in parameters[1] if any(nd in n for nd in no_decay_param)], "weight_decay": 0}
            ]
        elif self.use_head :
            optimizer_group_parameters = [
                {"params": [p for n, p in parameters[0] if not any(nd in n for nd in no_decay_param)], "weight_decay": self.args.weight_decay},
                {"params": [p for n, p in parameters[0] if any(nd in n for nd in no_decay_param)], "weight_decay": 0},
                {"params": [p for n, p in parameters[1] if not any(nd in n for nd in no_decay_param)], "weight_decay": self.args.weight_decay},
                {"params": [p for n, p in parameters[1] if any(nd in n for nd in no_decay_param)], "weight_decay": 0},
                {"params": [p for n, p in parameters[2] if not any(nd in n for nd in no_decay_param)], "weight_decay": self.args.weight_decay},
                {"params": [p for n, p in parameters[2] if any(nd in n for nd in no_decay_param)], "weight_decay": 0}
            ]
        elif self.use_kg:
            optimizer_group_parameters = [
                {"params": [p for n, p in parameters[0] if not any(nd in n for nd in no_decay_param)], "weight_decay": self.args.weight_decay},
                {"params": [p for n, p in parameters[0] if any(nd in n for nd in no_decay_param)], "weight_decay": 0},
                {"params": [p for n, p in parameters[1] if not any(nd in n for nd in no_decay_param)], "weight_decay": self.args.weight_decay},
                {"params": [p for n, p in parameters[1] if any(nd in n for nd in no_decay_param)], "weight_decay": 0},
                {"params": [p for n, p in parameters[2] if not any(nd in n for nd in no_decay_param)], "weight_decay": self.args.weight_decay},
                {"params": [p for n, p in parameters[2] if any(nd in n for nd in no_decay_param)], "weight_decay": 0},
                {"params": [p for n, p in parameters[3] if not any(nd in n for nd in no_decay_param)], "weight_decay": self.args.weight_decay},
                {"params": [p for n, p in parameters[3] if any(nd in n for nd in no_decay_param)], "weight_decay": 0},
                {"params": [p for n, p in parameters[4] if not any(nd in n for nd in no_decay_param)], "weight_decay": self.args.weight_decay},
                {"params": [p for n, p in parameters[4] if any(nd in n for nd in no_decay_param)], "weight_decay": 0}
            ]
        else:
            optimizer_group_parameters = [
                {"params": [p for n, p in parameters if not any(nd in n for nd in no_decay_param)], "weight_decay": self.args.weight_decay},
                {"params": [p for n, p in parameters if any(nd in n for nd in no_decay_param)], "weight_decay": 0}
            ]

        
        optimizer = self.optimizer_class(optimizer_group_parameters, lr=self.lr, eps=1e-8)
        scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=self.num_training_steps * 0.1, num_training_steps=self.num_training_steps)
        return {
            "optimizer": optimizer, 
            "lr_scheduler":{
                'scheduler': scheduler,
                'interval': 'step',  # or 'epoch'
                'frequency': 1,
            }
        }

class TransformerLitModelTwoSteps(BertLitModel):
    def configure_optimizers(self):
        no_decay_param = ["bais", "LayerNorm.weight"]
        optimizer_group_parameters = [
            {"params": [p for n, p in self.model.named_parameters() if not any(nd in n for nd in no_decay_param)], "weight_decay": self.args.weight_decay},
            {"params": [p for n, p in self.model.named_parameters() if any(nd in n for nd in no_decay_param)], "weight_decay": 0}
        ]
        optimizer = self.optimizer_class(optimizer_group_parameters, lr=self.args.lr_2, eps=1e-8)
        scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=self.num_training_steps * 0.1, num_training_steps=self.num_training_steps)
        return {
            "optimizer": optimizer, 
            "lr_scheduler":{
                'scheduler': scheduler,
                'interval': 'step',  # or 'epoch'
                'frequency': 1,
            }
        }



class DialogueLitModel(BertLitModel):

    def training_step(self, batch, batch_idx):  # pylint: disable=unused-argument
        input_ids, attention_mask, token_type_ids , labels = batch
        result = self.model(input_ids, attention_mask, token_type_ids, return_dict=True, output_hidden_states=True)
        logits = result.logits
        logits = self.pvp(logits, input_ids)
        loss = self.loss_fn(logits, labels) 
        self.log("Train/loss", loss)
        return loss

    def validation_step(self, batch, batch_idx):  # pylint: disable=unused-argument
        input_ids, attention_mask, token_type_ids , labels = batch
        logits = self.model(input_ids, attention_mask, token_type_ids, return_dict=True).logits
        logits = self.pvp(logits, input_ids)
        loss = self.loss_fn(logits, labels)
        self.log("Eval/loss", loss)
        return {"eval_logits": logits.detach().cpu().numpy(), "eval_labels": labels.detach().cpu().numpy()}
    
    def validation_epoch_end(self, outputs) -> None:
        logits = np.concatenate([o["eval_logits"] for o in outputs])
        labels = np.concatenate([o["eval_labels"] for o in outputs])

        f1 = self.eval_fn(logits, labels)['f1']
        self.log("Eval/f1", f1)
        if f1 > self.best_f1:
            self.best_f1 = f1
        self.log("Eval/best_f1", self.best_f1, prog_bar=True, on_epoch=True)

    def test_step(self, batch, batch_idx):  # pylint: disable=unused-argument
        input_ids, attention_mask, token_type_ids , labels = batch
        logits = self.model(input_ids, attention_mask, token_type_ids, return_dict=True).logits
        logits = self.pvp(logits, input_ids)
        return {"test_logits": logits.detach().cpu().numpy(), "test_labels": labels.detach().cpu().numpy()}

    def test_epoch_end(self, outputs) -> None:
        logits = np.concatenate([o["test_logits"] for o in outputs])
        labels = np.concatenate([o["test_labels"] for o in outputs])

        f1 = self.eval_fn(logits, labels)['f1']
        self.log("Test/f1", f1)
        


    @staticmethod
    def add_to_argparse(parser):
        BaseLitModel.add_to_argparse(parser)
        parser.add_argument("--t_lambda", type=float, default=0.01, help="")
        return parser
        
    def pvp(self, logits, input_ids):
        # convert the [batch_size, seq_len, vocab_size] => [batch_size, num_labels]
        #! hard coded
        _, mask_idx = (input_ids == 103).nonzero(as_tuple=True)
        bs = input_ids.shape[0]
        mask_output = logits[torch.arange(bs), mask_idx]
        assert mask_idx.shape[0] == bs, "only one mask in sequence!"
        final_output = mask_output[:,self.word2label]
        
        return final_output
        
def decode(tokenizer, output_ids):
    return [tokenizer.decode(g, skip_special_tokens=True, clean_up_tokenization_spaces=False) for g in output_ids]

class GPTLitModel(BaseLitModel):
    def __init__(self, model, args , tokenizer):
        super().__init__(model, args)
        # self.num_training_steps = data_config["num_training_steps"]
        self.loss_fn = nn.CrossEntropyLoss()
        # self.loss_fn = multilabel_categorical_crossentropy
        self.best_f1 = 0

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):  # pylint: disable=unused-argument
        input_ids, attention_mask, cls_idx, labels = batch
        logits = self.model(input_ids, attention_mask=attention_mask, mc_token_ids=cls_idx)
        if not isinstance(logits, torch.Tensor):
            logits = logits.mc_logits

        loss = self.loss_fn(logits, labels)
        self.log("Train/loss", loss)
        return loss

    def validation_step(self, batch, batch_idx):  # pylint: disable=unused-argument
        input_ids, attention_mask, cls_idx , labels = batch
        logits = self.model(input_ids, attention_mask=attention_mask, mc_token_ids=cls_idx)
        if not isinstance(logits, torch.Tensor):
            logits = logits.mc_logits
        loss = self.loss_fn(logits, labels)
        self.log("Eval/loss", loss)
        return {"eval_logits": logits.detach().cpu().numpy(), "eval_labels": labels.detach().cpu().numpy()}
    
    def validation_epoch_end(self, outputs) -> None:
        logits = np.concatenate([o["eval_logits"] for o in outputs])
        labels = np.concatenate([o["eval_labels"] for o in outputs])

        # f1 = compute_f1(logits, labels)["f1"]
        f1 = f1_score(logits, labels)['f1']
        self.log("Eval/f1", f1)
        if f1 > self.best_f1:
            self.best_f1 = f1
        self.log("Eval/best_f1", self.best_f1, prog_bar=True, on_epoch=True)

    def test_step(self, batch, batch_idx):  # pylint: disable=unused-argumenT
        input_ids, attention_mask, cls_idx , labels = batch
        logits = self.model(input_ids, attention_mask=attention_mask, mc_token_ids=cls_idx)
        if not isinstance(logits, torch.Tensor):
            logits = logits.mc_logits
        return {"test_logits": logits.detach().cpu().numpy(), "test_labels": labels.detach().cpu().numpy()}

    def test_epoch_end(self, outputs) -> None:
        logits = np.concatenate([o["test_logits"] for o in outputs])
        labels = np.concatenate([o["test_labels"] for o in outputs])

        f1 = f1_score(logits, labels)['f1']
        # f1 = acc(logits, labels)
        self.log("Test/f1", f1)





## new added GPT for ke-loss
class GPTLitModel2(BaseLitModel):
    def __init__(self, model, args , tokenizer):
        super().__init__(model, args)
        self.tokenizer = tokenizer
        
        with open(f"{args.data_dir}/rel2id.json","r") as file:
            rel2id = json.load(file)
        
        Na_num = 0
        for k, v in rel2id.items():
            if k == "NA" or k == "no_relation" or k == "Other":
                Na_num = v
                break
        num_relation = len(rel2id)

        # self.num_training_steps = data_config["num_training_steps"]
        self.loss_fn = nn.CrossEntropyLoss()
        # self.loss_fn = multilabel_categorical_crossentropy
        self.eval_fn = f1_eval if "dialogue" in args.data_dir else partial(f1_score, rel_num=num_relation, na_num=Na_num)
        self.best_f1 = 0
        self.t_lambda = args.t_lambda
        
        self.label_st_id = tokenizer("[class1]", add_special_tokens=False)['input_ids'][0]  # label start id
        self.tokenizer = tokenizer

        self._init_label_word()
    
    def _init_label_word(self, ):
        args = self.args
        # ./dataset/dataset_name
        dataset_name = args.data_dir.split("/")[1]
        model_name_or_path = args.model_name_or_path.split("/")[-1]
        label_path = f"./dataset/{model_name_or_path}_{dataset_name}.pt" # splited label word idx
        # [num_labels, num_tokens], ignore the unanswerable
        if "dialogue" in args.data_dir:
            label_word_idx = torch.load(label_path)[:-1]
        else:
            label_word_idx = torch.load(label_path)
        
        num_labels = len(label_word_idx)    # number of labels
        
        self.model.resize_token_embeddings(len(self.tokenizer))
        
        with torch.no_grad():
            word_embeddings = self.model.get_input_embeddings()
            # label index
            continous_label_word = [a[0] for a in self.tokenizer([f"[class{i}]" for i in range(1, num_labels+1)], add_special_tokens=False)['input_ids']]
            
            # for abaltion study
            if self.args.init_answer_words:
                print("init answer words")
                if self.args.init_answer_words_by_one_token:
                    for i, idx in enumerate(label_word_idx):
                        word_embeddings.weight[continous_label_word[i]] = word_embeddings.weight[idx][-1]
                else:
                    for i, idx in enumerate(label_word_idx):
                        word_embeddings.weight[continous_label_word[i]] = torch.mean(word_embeddings.weight[idx], dim=0)
                # word_embeddings.weight[continous_label_word[i]] = self.relation_embedding[i]
            
            if self.args.init_type_words:
                print("init type words")
                so_word = [a[0] for a in self.tokenizer(["[obj]","[sub]"], add_special_tokens=False)['input_ids']]
                meaning_word = [a[0] for a in self.tokenizer(["person","organization", "location", "date", "country"], add_special_tokens=False)['input_ids']]
            
                for i, idx in enumerate(so_word):
                    word_embeddings.weight[so_word[i]] = torch.mean(word_embeddings.weight[meaning_word], dim=0)
            assert torch.equal(self.model.get_input_embeddings().weight, word_embeddings.weight)
            assert torch.equal(self.model.get_input_embeddings().weight, self.model.get_output_embeddings().weight)
        
        self.word2label = continous_label_word # a continous list  

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):  # pylint: disable=unused-argument
        input_ids, attention_mask, cls_idx, labels, so = batch
        result = self.model(input_ids, attention_mask=attention_mask, mc_token_ids=cls_idx, return_dict=True, output_hidden_states=True)
        logits = result.logits
        output_embedding = result.hidden_states[-1]
        logits = self.pvp(logits, input_ids) 

        ke_loss = self.ke_loss(output_embedding, labels, so, input_ids)
        loss = self.loss_fn(logits, labels) + self.t_lambda * ke_loss
        self.log("Train/loss", loss)
        self.log("Train/ke_loss", loss)
        return loss

    def validation_step(self, batch, batch_idx):  # pylint: disable=unused-argument
        input_ids, attention_mask, cls_idx , labels, so = batch
        logits = self.model(input_ids, attention_mask=attention_mask, mc_token_ids=cls_idx, return_dict=True).logits
        logits = self.pvp(logits, input_ids)
        loss = self.loss_fn(logits, labels)
        self.log("Eval/loss", loss)
        return {"eval_logits": logits.detach().cpu().numpy(), "eval_labels": labels.detach().cpu().numpy()}
    
    def validation_epoch_end(self, outputs) -> None:
        logits = np.concatenate([o["eval_logits"] for o in outputs])
        labels = np.concatenate([o["eval_labels"] for o in outputs])

        # f1 = compute_f1(logits, labels)["f1"]
        f1 = f1_score(logits, labels)['f1']
        self.log("Eval/f1", f1)
        if f1 > self.best_f1:
            self.best_f1 = f1
        self.log("Eval/best_f1", self.best_f1, prog_bar=True, on_epoch=True)

    def test_step(self, batch, batch_idx):  # pylint: disable=unused-argumenT
        input_ids, attention_mask, cls_idx , labels, so = batch
        logits = self.model(input_ids, attention_mask=attention_mask, mc_token_ids=cls_idx, return_dict=True).logits
        logits = self.pvp(logits, input_ids)
        return {"test_logits": logits.detach().cpu().numpy(), "test_labels": labels.detach().cpu().numpy()}

    def test_epoch_end(self, outputs) -> None:
        logits = np.concatenate([o["test_logits"] for o in outputs])
        labels = np.concatenate([o["test_labels"] for o in outputs])

        f1 = f1_score(logits, labels)['f1']
        # f1 = acc(logits, labels)
        self.log("Test/f1", f1)  

    @staticmethod
    def add_to_argparse(parser):
        BaseLitModel.add_to_argparse(parser)
        parser.add_argument("--t_lambda", type=float, default=0.01, help="")
        parser.add_argument("--t_gamma", type=float, default=0.3, help="")
        return parser
        
    def pvp(self, logits, input_ids):
        # convert the [batch_size, seq_len, vocab_size] => [batch_size, num_labels]
        #! hard coded
        _, cls_idx = (input_ids == self.tokenizer.cls_token_id).nonzero(as_tuple=True)
        bs = input_ids.shape[0] # batch size
        cls_output = logits[torch.arange(bs), cls_idx]
        assert cls_idx.shape[0] == bs, "only one cls in sequence!"
        final_output = cls_output[:,self.word2label]
        
        return final_output

    def ke_loss(self, logits, labels, so, input_ids):
        subject_embedding = []
        object_embedding = []
        neg_subject_embedding = []
        neg_object_embedding = []
        bsz = logits.shape[0]
        for i in range(bsz):
            # x['so'] =[sub_st, sub_ed, obj_st, obj_ed]
            subject_embedding.append(torch.mean(logits[i, so[i][0]:so[i][1]], dim=0))
            object_embedding.append(torch.mean(logits[i, so[i][2]:so[i][3]], dim=0))

            # random select the neg samples
            st_sub = random.randint(1, logits[i].shape[0] - 6)
            span_sub = random.randint(1, 5)
            st_obj = random.randint(1, logits[i].shape[0] - 6)
            span_obj = random.randint(1, 5)
            neg_subject_embedding.append(torch.mean(logits[i, st_sub:st_sub+span_sub], dim=0))
            neg_object_embedding.append(torch.mean(logits[i, st_obj:st_obj+span_obj], dim=0))
            
        subject_embedding = torch.stack(subject_embedding)
        object_embedding = torch.stack(object_embedding)
        neg_subject_embedding = torch.stack(neg_subject_embedding)
        neg_object_embedding = torch.stack(neg_object_embedding)
        # trick , the relation ids is concated, 


        _, cls_idx = (input_ids == self.tokenizer.cls_token_id).nonzero(as_tuple=True)
        cls_output = logits[torch.arange(bsz), cls_idx]
        cls_relation_embedding = cls_output
        real_relation_embedding = self.model.get_output_embeddings().weight[labels+self.label_st_id]
        
        d_1 = torch.norm(subject_embedding + cls_relation_embedding - object_embedding, p=2) / bsz
        d_2 = torch.norm(neg_subject_embedding + real_relation_embedding - neg_object_embedding, p=2) / bsz
        f = torch.nn.LogSigmoid()
        loss = -1.*f(self.args.t_gamma - d_1) - f(d_2 - self.args.t_gamma)
        
        return loss

    def configure_optimizers(self):
        no_decay_param = ["bias", "LayerNorm.weight"]

        if not self.args.two_steps: 
            parameters = self.model.named_parameters()
        else:
            print("only optimize the embedding parameters")
            # model.bert.embeddings.weight
            parameters = [next(self.model.named_parameters())]
        # only optimize the embedding parameters
        optimizer_group_parameters = [
            {"params": [p for n, p in parameters if not any(nd in n for nd in no_decay_param)], "weight_decay": self.args.weight_decay},
            {"params": [p for n, p in parameters if any(nd in n for nd in no_decay_param)], "weight_decay": 0}
        ]

        
        optimizer = self.optimizer_class(optimizer_group_parameters, lr=self.lr, eps=1e-8)
        scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=self.num_training_steps * 0.1, num_training_steps=self.num_training_steps)
        return {
            "optimizer": optimizer, 
            "lr_scheduler":{
                'scheduler': scheduler,
                'interval': 'step',  # or 'epoch'
                'frequency': 1,
            }
        }
















from models.trie import get_trie
class BartRELitModel(BaseLitModel):
    def __init__(self, model, args, tokenizer=None):
        super().__init__(model, args)
        self.best_f1 = 0
        self.first = True

        with open(f"{args.data_dir}/rel2id.json","r") as file:
            rel2id = json.load(file)

        Na_num = 0
        for k, v in rel2id.items():
            if k == "NA" or k == "no_relation" or k == "Other":
                Na_num = v
                break
        num_relation = len(rel2id)
        # init loss function
        self.loss_fn = multilabel_categorical_crossentropy if "dialogue" in args.data_dir else nn.CrossEntropyLoss()
        # ignore the no_relation class to compute the f1 score
        self.eval_fn = f1_eval if "dialogue" in args.data_dir else partial(f1_score, rel_num=num_relation, na_num=Na_num)
        
        self.tokenizer = tokenizer
        self.trie, self.rel2id = get_trie(args, tokenizer=tokenizer)
        
        self.decode = partial(decode, tokenizer=self.tokenizer)

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):  # pylint: disable=unused-argument
        real_label  = batch.pop("label")
        loss = self.model(**batch).loss
        self.log("Train/loss", loss)
        return loss
        
        

    def validation_step(self, batch, batch_idx):
        real_label = batch.pop("label")
        labels = batch.pop("labels")
        batch.pop("decoder_input_ids")
        topk = 1
        outputs = self.model.generate(**batch, 
            prefix_allowed_tokens_fn=lambda batch_id, sent: self.trie.get(sent.tolist()),
            num_beams=topk, num_return_sequences=topk,
            output_scores=True,
            min_length=0,
            max_length=32,
        ).cpu()
        # calculate the rank in the decoder output 

        pad_id = self.tokenizer.pad_token_id
        outputs = self.decode(output_ids=outputs)
        labels = self.decode(output_ids=labels)
        
        preds = torch.tensor([self.rel2id[o] for o in outputs])
        true = real_label


        return {"eval_logits": preds.detach().cpu().numpy(), "eval_labels": true.detach().cpu().numpy()}


    
    def validation_epoch_end(self, outputs) -> None:
        logits = np.concatenate([o["eval_logits"] for o in outputs])
        labels = np.concatenate([o["eval_labels"] for o in outputs])

        f1 = self.eval_fn(logits, labels)['f1']
        self.log("Eval/f1", f1)
        if f1 > self.best_f1 and not self.first:
            self.best_f1 = f1
        self.first = False
        self.log("Eval/best_f1", self.best_f1, prog_bar=True, on_epoch=True)
       

    def test_step(self, batch, batch_idx):  # pylint: disable=unused-argument
        real_label = batch.pop("label")
        labels = batch.pop("labels")
        batch.pop("decoder_input_ids")
        topk = 1
        outputs = self.model.generate(**batch, 
            prefix_allowed_tokens_fn=lambda batch_id, sent: self.trie.get(sent.tolist()),
            num_beams=topk, num_return_sequences=topk,
            output_scores=True,
            min_length=0,
            max_length=32,
        ).cpu()
        # calculate the rank in the decoder output 

        pad_id = self.tokenizer.pad_token_id
        outputs = self.decode(output_ids=outputs)
        labels = self.decode(output_ids=labels)
        
        preds = torch.tensor([self.rel2id[o] for o in outputs])
        true = real_label


        return {"test_logits": preds.detach().cpu().numpy(), "test_labels": true.detach().cpu().numpy()}

    def test_epoch_end(self, outputs) -> None:
        logits = np.concatenate([o["test_logits"] for o in outputs])
        labels = np.concatenate([o["test_labels"] for o in outputs])

        f1 = self.eval_fn(logits, labels)['f1']
        self.log("Test/f1", f1)
        

    def configure_optimizers(self):
        no_decay_param = ["bias", "LayerNorm.weight"]

        optimizer_group_parameters = [
            {"params": [p for n, p in self.model.named_parameters() if not any(nd in n for nd in no_decay_param)], "weight_decay": self.args.weight_decay},
            {"params": [p for n, p in self.model.named_parameters() if any(nd in n for nd in no_decay_param)], "weight_decay": 0}
        ]

        optimizer = self.optimizer_class(optimizer_group_parameters, lr=self.lr, eps=1e-8)
        scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=self.num_training_steps * 0.1, num_training_steps=self.num_training_steps)
        return {
            "optimizer": optimizer, 
            "lr_scheduler":{
                'scheduler': scheduler,
                'interval': 'step',  # or 'epoch'
                'frequency': 1,
            }
        }
