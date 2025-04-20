#!/bin/bash

#SBATCH --job-name=knowprompt
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --gpus=a40:1
#SBATCH --mem=50g
#SBATCH --time=1-00:00:00
#SBATCH --account=drjieliu99
#SBATCH --partition=spgpu
#SBATCH --output=runlog/gpt.log
#SBATCH --mail-user=2928980064@qq.com
#SBATCH --mail-type=BEGIN,END

python3 get_label_word.py

python3 main.py --max_epochs=10  --num_workers=8 \
    --model_name_or_path gpt2-large \
    --accumulate_grad_batches 4 \
    --batch_size 8 \
    --data_dir dataset/biored1 \
    --check_val_every_n_epoch 1 \
    --data_class WIKI80 \
    --max_seq_length 256 \
    --model_class GPTForPrompt \
    --wandb \
    --t_lambda 0.001 \
    --litmodel_class GPTLitModel2 \
    --task_name wiki80 \
    --lr 3e-5 
