#!/bin/bash
set -e

#python preprocess.py \
#    -train_src dataset/train.src.txt \
#    -train_tgt dataset/train.tgt.txt \
#    -valid_src dataset/valid.src.txt \
#    -valid_tgt dataset/valid.tgt.txt \
#    -save_data dataset/dataset.pt

python train.py \
    -data dataset/dataset.pt \
        -d_model 512 \
        -d_inner_hid 512 \
        -n_head 4 \
        -n_warmup_step 10000 \
        -save_model trained \
        -save_mode best \
        -embs_share_weight \
        -proj_share_weight
