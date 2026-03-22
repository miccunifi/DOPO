
HYDRA_FULL_ERROR=1 python train_model_spo.py \
    data=kitml \
    model=mdm_spo \
    data.with_noise=false \
    data.text_to_token_emb=false \
    data.text_to_sent_emb=false \
    dataloader.batch_size=160 \
    model.train_batch_size=160 \
    model.lr=1e-7 \
    model.ckpt="last" \
    evaluator.checkpoint_dir='outputs/densetmr_kitml_smplrifke' \
    model.checkpoint_dir='outputs/mdm_humanml3d_smplrifke' \
    run_dir='outputs/mdmspo_humanml3d_to_kitml_smplrifke_lr1e-7' \
    trainer.max_epochs=20 \
    group_name='H2K' 
