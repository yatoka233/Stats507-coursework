#!/bin/bash

#SBATCH --job-name=knowprompt
#SBATCH --nodes=1
#SBATCH --cpus-per-task=4
#SBATCH --gpus=a40:1
#SBATCH --mem=50g
#SBATCH --time=1-00:00:00
#SBATCH --account=drjieliu99
#SBATCH --partition=spgpu
#SBATCH --output=runlog/biored_PubMedBERT.log
#SBATCH --mail-user=2928980064@qq.com
#SBATCH --mail-type=BEGIN,END

python3 get_label_word.py

python3 main.py --max_epochs=10  --num_workers=8 \
    --model_name_or_path dmis-lab/biobert-base-cased-v1.2 \
    --accumulate_grad_batches 4 \
    --batch_size 32 \
    --data_dir dataset/biored_mapped \
    --check_val_every_n_epoch 1 \
    --data_class WIKI80 \
    --max_seq_length 256 \
    --model_class BertForPrompt \
    --t_lambda 0.01 \
    --litmodel_class BertLitModel \
    --task_name wiki80 \
    --lr 1e-5 \
    --align_lambda 0 \
    --lp_lambda 0

# python3 main.py --max_epochs=10  --num_workers=8 \
#     --model_name_or_path microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract-fulltext \
#     --accumulate_grad_batches 4 \
#     --batch_size 32 \
#     --data_dir dataset/biored_mapped \
#     --check_val_every_n_epoch 1 \
#     --data_class WIKI80 \
#     --max_seq_length 256 \
#     --model_class BertForPrompt \
#     --t_lambda 0.01 \
#     --litmodel_class BertLitModel \
#     --task_name wiki80 \
#     --lr 1e-5 \
#     --align_lambda 0.005 \
#     --lp_lambda 0.001

