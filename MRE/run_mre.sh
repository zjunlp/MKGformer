#!/usr/bin/env bash
# Required environment variables:
# batch_size (recommendation: 32 / 16)
# lr: learning rate (recommendation: 1e-5 / 3e-5)
# seed: random seed, default is 1234
# BERT_NAME: pre-trained text model name ( bert-*)
# VIT_NAME: pre-trained visual model name ( openai/clip-vit-base-patch32 / google/vit-base-patch32-224-in21k)
# max_seq: max sequence length
# aux_size: image size of visual grouding images
# rcnn_size: image size of RCNN detected object images

DATASET_NAME="MRE"
BERT_NAME="bert-base-uncased"
VIT_NAME="openai/clip-vit-base-patch32"

CUDA_VISIBLE_DEVICES=2 python -u run.py \
        --model_name="bert" \
        --vit_name=$VIT_NAME \
        --dataset_name=${DATASET_NAME} \
        --bert_name=${BERT_NAME} \
        --num_epochs=12 \
        --batch_size=32 \
        --lr=1e-5 \
        --warmup_ratio=0.06 \
        --eval_begin_epoch=1 \
        --seed=1234 \
        --do_train \
        --max_seq=80 \
        --prompt_len=4 \
        --aux_size=128 \
        --rcnn_size=64 \
        --save_path="ckpt"
