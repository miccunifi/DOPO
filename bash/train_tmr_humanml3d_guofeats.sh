#!/bin/bash

HYDRA_FULL_ERROR=1 python train_evaluator.py \
    data=humanml3d \
    model=tmr \
    data/motion_loader=guoh3dfeats \
    data.with_noise=false \
