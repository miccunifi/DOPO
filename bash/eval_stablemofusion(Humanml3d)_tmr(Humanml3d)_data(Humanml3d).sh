#!/bin/bash

HYDRA_FULL_ERROR=1 python eval_model.py \
    model=stablemofusion \
    evaluator=tmr \
    data=humanml3d \
    model_checkpoint_dir='outputs/stablemofusion_humanml3d_smplrifke' \
    evaluator_checkpoint_dir='outputs/tmr_humanml3d_smplrifke' \
    data.clip_embedder=false \
    data.text_to_token_emb=false \
    data.text_to_sent_emb=false \
    distance_metric='cosine' \