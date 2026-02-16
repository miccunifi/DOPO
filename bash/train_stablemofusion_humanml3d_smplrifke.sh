#!/bin/bash

HYDRA_FULL_ERROR=1 python train_model.py \
    data=humanml3d \
    model=stablemofusion \
    data.with_noise=false