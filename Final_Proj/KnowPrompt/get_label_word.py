from transformers import AutoTokenizer
from torch.nn.utils.rnn import pad_sequence
import re
import torch
import json

# roberta-large
# bert-base-uncased
# allenai/scibert_scivocab_uncased
# allenai/scibert_scivocab_cased
# microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract-fulltext

# microsoft/biogpt
# microsoft/BioGPT-Large
# suarkadipa/GPT-2-finetuned-medical-domain

# emilyalsentzer/Bio_ClinicalBERT 
# dmis-lab/biobert-base-cased-v1.1
# dmis-lab/biobert-base-cased-v1.2
# allenai/biomed_roberta_base
# microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract
# cambridgeltl/SapBERT-from-PubMedBERT-fulltext
# microsoft/BiomedNLP-BiomedBERT-base-uncased-abstract-fulltext

model_name_or_path = "dmis-lab/biobert-base-cased-v1.2"
# dataset_name = "biored_mapped"


dataset_name = "biored_mapped/k-shot/4-1"




tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
def split_label_words(tokenizer, label_list):
    label_word_list = []
    for label in label_list:
        if label == 'no_relation' or label == "NA":
            label_word_id = tokenizer.encode('no relation', add_special_tokens=False)
            label_word_list.append(torch.tensor(label_word_id))
        else:
            tmps = label
            label = label.lower()
            label = label.split("(")[0]
            label = label.replace(":"," ").replace("_"," ").replace("per","person").replace("org","organization")
            label_word_id = tokenizer(label, add_special_tokens=False)['input_ids']
            print(label, label_word_id)
            label_word_list.append(torch.tensor(label_word_id))
    padded_label_word_list = pad_sequence([x for x in label_word_list], batch_first=True, padding_value=0)
    return padded_label_word_list

with open(f"dataset/{dataset_name}/rel2id.json", "r") as file:
    t = json.load(file)
    label_list = list(t)

t = split_label_words(tokenizer, label_list)

# name string after /
model_name_or_path = model_name_or_path.split('/')[-1]
dataset_name = dataset_name.split('/')[0]

with open(f"./dataset/{model_name_or_path}_{dataset_name}.pt", "wb") as file:
    torch.save(t, file)