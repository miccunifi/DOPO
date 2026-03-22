
# MotionX -> HumanML3D (lr=1e-6)
HYDRA_FULL_ERROR=1 python train_model_spo.py \
    data=humanml3d \
    model=mdm_spo \
    data.with_noise=false \
    data.text_to_token_emb=false \
    data.text_to_sent_emb=false \
    dataloader.batch_size=190 \
    model.train_batch_size=190 \
    model.lr=1e-6 \
    model.lora=True \
    model.ckpt="last" \
    evaluator.checkpoint_dir='outputs/densetmr_humanml3d_smplrifke' \
    model.checkpoint_dir='outputs/mdm_motionx_smplrifke' \
    trainer.max_epochs=20 \
    group_name='M2H'