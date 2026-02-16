#!/bin/bash

HYDRA_FULL_ERROR=1 python eval_evaluator.py \
    data=humanml3d \
    evaluator=densetmr \
    evaluator.checkpoint_dir="/deck/groups/MotionRL/DensePreferenceOptimization/outputs/densetmr_humanml3d_smplrifke" \
    retrieval_batch_size=-1 \
    distance_metric='cosine' \