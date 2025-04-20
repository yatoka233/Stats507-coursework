#!/bin/bash

#SBATCH --job-name=kg
#SBATCH --nodes=1
#SBATCH --cpus-per-task=4
#SBATCH --gpus=1
#SBATCH --mem=50g
#SBATCH --time=1-00:00:00
#SBATCH --account=drjieliu
#SBATCH --partition=drjieliu
#SBATCH --output=runlog/hgt_glkb.log
#SBATCH --mail-user=hyhao@umich.edu
#SBATCH --mail-type=BEGIN,END

# python3 hgt_lightning_lp.py
# python /nfs/turbo/umms-drjieliu/proj/prompt_re/kg_embeddings/train_hgt_lp.py
python /nfs/turbo/umms-drjieliu/proj/prompt_re/kg_embeddings/train_hgt_lp_binary.py