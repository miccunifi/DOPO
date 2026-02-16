
HYDRA_FULL_ERROR=1 python train_model_spo.py \
    data=kitml \
    model=mdm_spo \
    data.with_noise=false \
    data.text_to_token_emb=false \
    data.text_to_sent_emb=false \
    dataloader.batch_size=128 \
    model.train_batch_size=128 \
    model.lr=1e-6 \
    model.lora=False \
    model.ckpt="last" \
    evaluator.checkpoint_dir='outputs/densetmr_kitml_smplrifke' \
    model.checkpoint_dir='outputs/mdm_babel_smplrifke' \
    run_dir='outputs/mdmspo_babel_to_kitml_smplrifke_lr1e-6_bs128' \
    trainer.max_epochs=20 \
    group_name='B2K' 
    