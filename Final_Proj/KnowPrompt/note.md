# Hyper parameters needs to be set
self.args.init_answer_words \
self.args.init_answer_words_by_one_token \
self.args.init_type_words \
args.two_steps \


# Environment settings
pip install torchmetrics==0.5 \
pip install wandb \
多卡训练DDP不需要加 CUDA_VISIBLE_DEVICES=0 \
使用多卡训练DDP，并使用wandb进行log的话，需要先设置成tensorboard，run起来初始化好了之后再切断重新换成wandb


# Differences from paper

1. 若不用two_steps (default)，可训练参数是全部的embeddings.weight (transformer.py line 307)

2. input = prompt + tokens (processor.py line 793)

3. virtual tokens 的初始化是直接取子token的mean (transformer.py line 162)