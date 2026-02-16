#!/bin/bash

HYDRA_FULL_ERROR=1 python train_model.py \
    data=humanml3d \
    model=stablemofusion \
    data.with_noise=false

HYDRA_FULL_ERROR=1 python train_model.py \
    data=babel \
    model=stablemofusion \
    data.with_noise=false

HYDRA_FULL_ERROR=1 python train_model.py \
    data=kitml \
    model=stablemofusion \
    data.with_noise=false

HYDRA_FULL_ERROR=1 python train_model.py \
    data=motionx \
    model=stablemofusion \
    data.with_noise=false

HYDRA_FULL_ERROR=1 python train_model.py \
    data=humanml3d \
    model=stablemofusion \
    data/motion_loader=guoh3dfeats \
    data.with_noise=false