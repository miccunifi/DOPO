#!/bin/bash
HYDRA_FULL_ERROR=1 python train_evaluator.py \
    data=humanml3d \
    model=densetmr \
    data.with_noise=true