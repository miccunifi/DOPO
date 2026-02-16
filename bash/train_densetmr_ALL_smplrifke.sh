#!/bin/bash

# HYDRA_FULL_ERROR=1 python train_evaluator.py \
#     data=babel \
#     model=densetmr \
#     data.with_noise=true

HYDRA_FULL_ERROR=1 python train_evaluator.py \
    data=kitml \
    model=densetmr \
    data.with_noise=true \
    data.clip_embedder=false

HYDRA_FULL_ERROR=1 python train_evaluator.py \
    data=humanml3d \
    model=densetmr \
    data.with_noise=true \
    data.clip_embedder=false

HYDRA_FULL_ERROR=1 python train_evaluator.py \
    data=motionx \
    model=densetmr \
    data.with_noise=true\
    data.clip_embedder=false

HYDRA_FULL_ERROR=1 python train_evaluator.py \
    data=kitml++ \
    model=densetmr++ \
    data.with_noise=true\
    data.clip_embedder=false

HYDRA_FULL_ERROR=1 python train_evaluator.py \
    data=babel++ \
    model=densetmr++ \
    data.with_noise=true\
    data.clip_embedder=false

HYDRA_FULL_ERROR=1 python train_evaluator.py \
    data=humanml3d++ \
    model=densetmr++ \
    data.with_noise=true\
    data.clip_embedder=false