ulimit -n 65536
python -m prepare.text_embeddings data=kitml
HYDRA_FULL_ERROR=1 python train.py --config-name=train_with_augmentation_babel data=kitml run_dir=outputs/kitml_debug