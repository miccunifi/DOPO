#!/bin/bash

HYDRA_FULL_ERROR=1 python eval_evaluator.py \
    data=humanml3d \
    evaluator=guo \
    retrieval_batch_size=-1 \
    distance_metric='euclidean' \
    data/motion_loader=guoh3dfeats \
    data.with_noise=false \