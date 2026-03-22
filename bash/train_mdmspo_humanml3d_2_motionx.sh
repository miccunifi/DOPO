
HYDRA_FULL_ERROR=1 python train_model_spo.py \
    data=motionx \
    model=mdm_spo \
    data.with_noise=false \
    data.text_to_token_emb=false \
    data.text_to_sent_emb=false \
    dataloader.batch_size=190 \
    model.train_batch_size=190 \
    model.lr=1e-6 \
    model.lora=True \
    evaluator.checkpoint_dir='outputs/densetmr_motionx_smplrifke' \
    model.checkpoint_dir='outputs/mdm_humanml3d_smplrifke' \
    run_dir='outputs/mdmspo_humanml3d_to_motionx_smplrifke_lr1e-6' \
    trainer.max_epochs=20 \
    group_name='H2M' 
    
