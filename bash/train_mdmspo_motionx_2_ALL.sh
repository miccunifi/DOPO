
# MotionX -> HumanML3D (lr=1e-6)
HYDRA_FULL_ERROR=1 python train_model_spo.py \
    data=humanml3d \
    model=mdm_spo \
    data.with_noise=false \
    data.text_to_token_emb=false \
    data.text_to_sent_emb=false \
    dataloader.batch_size=160 \
    model.train_batch_size=160 \
    model.lr=1e-6 \
    model.lora=True \
    model.ckpt="last" \
    evaluator.checkpoint_dir='outputs/densetmr_humanml3d_smplrifke' \
    model.checkpoint_dir='outputs/mdm_motionx_smplrifke' \
    run_dir='outputs/mdmspo_motionx_to_humanml3d_smplrifke_lr1e-6_bs160' \
    trainer.max_epochs=20 \
    group_name='M2H'


# MotionX -> KiTML (lr=1e-6)
HYDRA_FULL_ERROR=1 python train_model_spo.py \
    data=kitml \
    model=mdm_spo \
    data.with_noise=false \
    data.text_to_token_emb=false \
    data.text_to_sent_emb=false \
    dataloader.batch_size=160 \
    model.train_batch_size=160 \
    model.lr=1e-6 \
    model.lora=True \
    model.ckpt="last" \
    evaluator.checkpoint_dir='outputs/densetmr_kitml_smplrifke' \
    model.checkpoint_dir='outputs/mdm_motionx_smplrifke' \
    run_dir='outputs/mdmspo_motionx_to_kitml_smplrifke_lr1e-6_bs160' \
    trainer.max_epochs=20 \
    group_name='M2K'


# MotionX -> BABEL (lr=1e-6)
HYDRA_FULL_ERROR=1 python train_model_spo.py \
    data=babel \
    model=mdm_spo \
    data.with_noise=false \
    data.text_to_token_emb=false \
    data.text_to_sent_emb=false \
    dataloader.batch_size=160 \
    model.train_batch_size=160 \
    model.lr=1e-6 \
    model.lora=True \
    model.ckpt="last" \
    evaluator.checkpoint_dir='outputs/densetmr_babel_smplrifke' \
    model.checkpoint_dir='outputs/mdm_motionx_smplrifke' \
    run_dir='outputs/mdmspo_motionx_to_babel_smplrifke_lr1e-6_bs160' \
    trainer.max_epochs=20 \
    group_name='M2B'


# MotionX -> HumanML3D (lr=1e-7)
HYDRA_FULL_ERROR=1 python train_model_spo.py \
    data=humanml3d \
    model=mdm_spo \
    data.with_noise=false \
    data.text_to_token_emb=false \
    data.text_to_sent_emb=false \
    dataloader.batch_size=160 \
    model.train_batch_size=160 \
    model.lr=1e-7 \
    model.lora=True \
    model.ckpt="last" \
    evaluator.checkpoint_dir='outputs/densetmr_humanml3d_smplrifke' \
    model.checkpoint_dir='outputs/mdm_motionx_smplrifke' \
    run_dir='outputs/mdmspo_motionx_to_humanml3d_smplrifke_lr1e-7_bs160' \
    trainer.max_epochs=20 \
    group_name='M2H'


# MotionX -> KiTML (lr=1e-7)
HYDRA_FULL_ERROR=1 python train_model_spo.py \
    data=kitml \
    model=mdm_spo \
    data.with_noise=false \
    data.text_to_token_emb=false \
    data.text_to_sent_emb=false \
    dataloader.batch_size=160 \
    model.train_batch_size=160 \
    model.lr=1e-7 \
    model.lora=True \
    model.ckpt="last" \
    evaluator.checkpoint_dir='outputs/densetmr_kitml_smplrifke' \
    model.checkpoint_dir='outputs/mdm_motionx_smplrifke' \
    run_dir='outputs/mdmspo_motionx_to_kitml_smplrifke_lr1e-7_bs160' \
    trainer.max_epochs=20 \
    group_name='M2K'


# MotionX -> BABEL (lr=1e-7)
HYDRA_FULL_ERROR=1 python train_model_spo.py \
    data=babel \
    model=mdm_spo \
    data.with_noise=false \
    data.text_to_token_emb=false \
    data.text_to_sent_emb=false \
    dataloader.batch_size=160 \
    model.train_batch_size=160 \
    model.lr=1e-7 \
    model.lora=True \
    model.ckpt="last" \
    evaluator.checkpoint_dir='outputs/densetmr_babel_smplrifke' \
    model.checkpoint_dir='outputs/mdm_motionx_smplrifke' \
    run_dir='outputs/mdmspo_motionx_to_babel_smplrifke_lr1e-7_bs160' \
    trainer.max_epochs=20 \
    group_name='M2B'
