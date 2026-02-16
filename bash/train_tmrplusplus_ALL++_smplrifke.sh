#!/bin/bash

HYDRA_FULL_ERROR=1 python train_evaluator.py \
    data=humanml3d++ \
    model=tmr++ \
    data.with_noise=false \
    data.clip_embedder=false

HYDRA_FULL_ERROR=1 python train_evaluator.py \
    data=kitml++ \
    model=tmr++ \
    data.with_noise=false \
    data.clip_embedder=false

HYDRA_FULL_ERROR=1 python train_evaluator.py \
    data=babel++ \
    model=tmr++ \
    data.with_noise=false \
    data.clip_embedder=false