#!/bin/bash

#SBATCH --job-name=knowprompt
#SBATCH --nodes=1
#SBATCH --cpus-per-task=2
#SBATCH --gpus=1
#SBATCH --mem=50g
#SBATCH --time=1-00:00:00
#SBATCH --account=drjieliu99
#SBATCH --partition=gpu
#SBATCH --output=test.log
#SBATCH --mail-user=2928980064@qq.com
#SBATCH --mail-type=BEGIN,END

python3 test.py