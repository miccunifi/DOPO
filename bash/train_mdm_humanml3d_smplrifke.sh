#!/bin/bash

HYDRA_FULL_ERROR=1 python train_model.py \
    data=humanml3d \
    model=mdm \
    data.with_noise=false \
    data/motion_loader=smplrifke 