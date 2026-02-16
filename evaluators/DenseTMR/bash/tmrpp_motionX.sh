ulimit -n 65536
#HYDRA_FULL_ERROR=1 python train.py --config-name=train_with_augmentation data=humanml3d run_dir=outputs/humantmrpp 
#python -m prepare.text_embeddings data=kitml
#HYDRA_FULL_ERROR=1 python train.py --config-name=train_with_augmentation data=kitml run_dir=outputs/kitmltmrpp
# python -m prepare.text_embeddings data=motionx
#HYDRA_FULL_ERROR=1 python -m prepare.motion_stats data=motionx data.motion_loader.base_dir="/deck/datasets/MotionX\(smplrifke\)/"
HYDRA_FULL_ERROR=1 python train.py --config-name=train_with_augmentation  data=motionx run_dir=outputs/motionXtmrpp data.motion_loader.base_dir="/deck/datasets/MotionX\(smplrifke\)/"
# HYDRA_FULL_ERROR=1 python train.py data=motionx run_dir=outputs/motionXtmrpp data.motion_loader.base_dir="/deck/datasets/MotionX\(smplrifke\)/"
