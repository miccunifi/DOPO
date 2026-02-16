ulimit -n 65536
#HYDRA_FULL_ERROR=1 python train.py --config-name=train_with_augmentation data=humanml3d run_dir=outputs/humantmrpp 
#python -m prepare.text_embeddings data=kitml
#HYDRA_FULL_ERROR=1 python train.py --config-name=train_with_augmentation data=kitml run_dir=outputs/kitmltmrpp
python -m prepare.text_embeddings data=babel
HYDRA_FULL_ERROR=1 python train.py --config-name=train_with_augmentation_babel data=babel run_dir=outputs/babeltmrpp
