#!/bin/bash
# Evaluate baseline MDM models (before finetuning) on each cross-dataset target.
# Provides the reference scores to confirm SPO finetuning actually helps.
# Results saved to evaluation_results/mdm_{source}_on_{target}/results.json


# --- HumanML3D model on other datasets ---

HYDRA_FULL_ERROR=1 python eval_model.py \
    model=mdm \
    evaluator=densetmr \
    data=kitml \
    model_checkpoint_dir='outputs/mdm_humanml3d_smplrifke' \
    evaluator_checkpoint_dir='outputs/densetmr_kitml_smplrifke' \
    data.clip_embedder=false \
    data.text_to_token_emb=false \
    data.text_to_sent_emb=false \
    distance_metric='cosine' \
    model_ckpt="last" \
    output_dir='evaluation_results/mdm_humanml3d_on_kitml'

HYDRA_FULL_ERROR=1 python eval_model.py \
    model=mdm \
    evaluator=densetmr \
    data=babel \
    model_checkpoint_dir='outputs/mdm_humanml3d_smplrifke' \
    evaluator_checkpoint_dir='outputs/densetmr_babel_smplrifke' \
    data.clip_embedder=false \
    data.text_to_token_emb=false \
    data.text_to_sent_emb=false \
    distance_metric='cosine' \
    model_ckpt="last" \
    output_dir='evaluation_results/mdm_humanml3d_on_babel'

HYDRA_FULL_ERROR=1 python eval_model.py \
    model=mdm \
    evaluator=densetmr \
    data=motionx \
    model_checkpoint_dir='outputs/mdm_humanml3d_smplrifke' \
    evaluator_checkpoint_dir='outputs/densetmr_motionx_smplrifke' \
    data.clip_embedder=false \
    data.text_to_token_emb=false \
    data.text_to_sent_emb=false \
    distance_metric='cosine' \
    model_ckpt="last" \
    output_dir='evaluation_results/mdm_humanml3d_on_motionx'


# --- KiTML model on other datasets ---

HYDRA_FULL_ERROR=1 python eval_model.py \
    model=mdm \
    evaluator=densetmr \
    data=humanml3d \
    model_checkpoint_dir='outputs/mdm_kitml_smplrifke' \
    evaluator_checkpoint_dir='outputs/densetmr_humanml3d_smplrifke' \
    data.clip_embedder=false \
    data.text_to_token_emb=false \
    data.text_to_sent_emb=false \
    distance_metric='cosine' \
    model_ckpt="last" \
    output_dir='evaluation_results/mdm_kitml_on_humanml3d'

HYDRA_FULL_ERROR=1 python eval_model.py \
    model=mdm \
    evaluator=densetmr \
    data=babel \
    model_checkpoint_dir='outputs/mdm_kitml_smplrifke' \
    evaluator_checkpoint_dir='outputs/densetmr_babel_smplrifke' \
    data.clip_embedder=false \
    data.text_to_token_emb=false \
    data.text_to_sent_emb=false \
    distance_metric='cosine' \
    model_ckpt="last" \
    output_dir='evaluation_results/mdm_kitml_on_babel'

HYDRA_FULL_ERROR=1 python eval_model.py \
    model=mdm \
    evaluator=densetmr \
    data=motionx \
    model_checkpoint_dir='outputs/mdm_kitml_smplrifke' \
    evaluator_checkpoint_dir='outputs/densetmr_motionx_smplrifke' \
    data.clip_embedder=false \
    data.text_to_token_emb=false \
    data.text_to_sent_emb=false \
    distance_metric='cosine' \
    model_ckpt="last" \
    output_dir='evaluation_results/mdm_kitml_on_motionx'


# --- BABEL model on other datasets ---

HYDRA_FULL_ERROR=1 python eval_model.py \
    model=mdm \
    evaluator=densetmr \
    data=humanml3d \
    model_checkpoint_dir='outputs/mdm_babel_smplrifke' \
    evaluator_checkpoint_dir='outputs/densetmr_humanml3d_smplrifke' \
    data.clip_embedder=false \
    data.text_to_token_emb=false \
    data.text_to_sent_emb=false \
    distance_metric='cosine' \
    model_ckpt="last" \
    output_dir='evaluation_results/mdm_babel_on_humanml3d'

HYDRA_FULL_ERROR=1 python eval_model.py \
    model=mdm \
    evaluator=densetmr \
    data=kitml \
    model_checkpoint_dir='outputs/mdm_babel_smplrifke' \
    evaluator_checkpoint_dir='outputs/densetmr_kitml_smplrifke' \
    data.clip_embedder=false \
    data.text_to_token_emb=false \
    data.text_to_sent_emb=false \
    distance_metric='cosine' \
    model_ckpt="last" \
    output_dir='evaluation_results/mdm_babel_on_kitml'

HYDRA_FULL_ERROR=1 python eval_model.py \
    model=mdm \
    evaluator=densetmr \
    data=motionx \
    model_checkpoint_dir='outputs/mdm_babel_smplrifke' \
    evaluator_checkpoint_dir='outputs/densetmr_motionx_smplrifke' \
    data.clip_embedder=false \
    data.text_to_token_emb=false \
    data.text_to_sent_emb=false \
    distance_metric='cosine' \
    model_ckpt="last" \
    output_dir='evaluation_results/mdm_babel_on_motionx'


# --- MotionX model on other datasets ---

HYDRA_FULL_ERROR=1 python eval_model.py \
    model=mdm \
    evaluator=densetmr \
    data=humanml3d \
    model_checkpoint_dir='outputs/mdm_motionx_smplrifke' \
    evaluator_checkpoint_dir='outputs/densetmr_humanml3d_smplrifke' \
    data.clip_embedder=false \
    data.text_to_token_emb=false \
    data.text_to_sent_emb=false \
    distance_metric='cosine' \
    model_ckpt="last" \
    output_dir='evaluation_results/mdm_motionx_on_humanml3d'

HYDRA_FULL_ERROR=1 python eval_model.py \
    model=mdm \
    evaluator=densetmr \
    data=kitml \
    model_checkpoint_dir='outputs/mdm_motionx_smplrifke' \
    evaluator_checkpoint_dir='outputs/densetmr_kitml_smplrifke' \
    data.clip_embedder=false \
    data.text_to_token_emb=false \
    data.text_to_sent_emb=false \
    distance_metric='cosine' \
    model_ckpt="last" \
    output_dir='evaluation_results/mdm_motionx_on_kitml'

HYDRA_FULL_ERROR=1 python eval_model.py \
    model=mdm \
    evaluator=densetmr \
    data=babel \
    model_checkpoint_dir='outputs/mdm_motionx_smplrifke' \
    evaluator_checkpoint_dir='outputs/densetmr_babel_smplrifke' \
    data.clip_embedder=false \
    data.text_to_token_emb=false \
    data.text_to_sent_emb=false \
    distance_metric='cosine' \
    model_ckpt="last" \
    output_dir='evaluation_results/mdm_motionx_on_babel'
