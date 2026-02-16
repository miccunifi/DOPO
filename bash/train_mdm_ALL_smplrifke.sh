#!/bin/bash

# HYDRA_FULL_ERROR=1 python train_model.py \
#     data=kitml \
#     model=mdm \
#     data.with_noise=false \
#     data.text_to_token_emb=false \
#     data.text_to_sent_emb=false

# HYDRA_FULL_ERROR=1 python train_model.py \
#     data=humanml3d \
#     model=mdm \
#     data.with_noise=false \
#     data.text_to_token_emb=false \
#     data.text_to_sent_emb=false

# HYDRA_FULL_ERROR=1 python train_model.py \
#     data=babel \
#     model=mdm \
#     data.with_noise=false\
#     data.text_to_token_emb=false \
#     data.text_to_sent_emb=false

HYDRA_FULL_ERROR=1 python train_model.py \
    data=motionx \
    model=mdm \
    data.with_noise=false \
    data.text_to_token_emb=false \
    data.text_to_sent_emb=false