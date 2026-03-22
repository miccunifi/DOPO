#!/bin/bash

HYDRA_FULL_ERROR=1 python train_model.py \
    data=motionx \
    model=mdm \
    data.with_noise=false \
    data.text_to_token_emb=false \
    data.text_to_sent_emb=false \
    resume_dir="outputs/mdm_motionx_smplrifke/"    