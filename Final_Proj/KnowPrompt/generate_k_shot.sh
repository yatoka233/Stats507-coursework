#!/bin/bash

#SBATCH --job-name=knowprompt
#SBATCH --nodes=1
#SBATCH --account=drjieliu99
#SBATCH --partition=standard
#SBATCH --output=runlog/generate_k_shot.log
#SBATCH --mail-user=2928980064@qq.com
#SBATCH --mail-type=BEGIN,END

python3 generate_k_shot.py --k=16 --data_file train.txt

python3 generate_k_shot.py --k=8 --data_file train.txt

python3 generate_k_shot.py --k=4 --data_file train.txt


cd /nfs/turbo/umms-drjieliu/proj/prompt_re/KnowPrompt/dataset/biored_mapped
cp type_list.json rel2id.json val.txt test.txt ./k-shot/4-1
cp type_list.json rel2id.json val.txt test.txt ./k-shot/4-2
cp type_list.json rel2id.json val.txt test.txt ./k-shot/4-3
cp type_list.json rel2id.json val.txt test.txt ./k-shot/4-4
cp type_list.json rel2id.json val.txt test.txt ./k-shot/4-5
cp type_list.json rel2id.json val.txt test.txt ./k-shot/8-1
cp type_list.json rel2id.json val.txt test.txt ./k-shot/8-2
cp type_list.json rel2id.json val.txt test.txt ./k-shot/8-3
cp type_list.json rel2id.json val.txt test.txt ./k-shot/8-4
cp type_list.json rel2id.json val.txt test.txt ./k-shot/8-5
cp type_list.json rel2id.json val.txt test.txt ./k-shot/16-1
cp type_list.json rel2id.json val.txt test.txt ./k-shot/16-2
cp type_list.json rel2id.json val.txt test.txt ./k-shot/16-3
cp type_list.json rel2id.json val.txt test.txt ./k-shot/16-4
cp type_list.json rel2id.json val.txt test.txt ./k-shot/16-5