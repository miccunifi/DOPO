#!/bin/bash
# Evaluate best finetuned MDM-SPO checkpoint for each completed cross-dataset pair.
# Uses DenseTMR evaluator (same metric as training) on the full test set.
# Results saved to evaluation_results/model_mdm_spo_eval_densetmr_{target}_test/results.json


# B -> H
HYDRA_FULL_ERROR=1 python eval_model.py \
    model=mdm_spo \
    evaluator=densetmr \
    data=humanml3d \
    model_checkpoint_dir='outputs/mdmspo_babel_to_humanml3d_smplrifke_lr1e-7' \
    evaluator_checkpoint_dir='outputs/densetmr_humanml3d_smplrifke' \
    data.clip_embedder=false \
    data.text_to_token_emb=false \
    data.text_to_sent_emb=false \
    distance_metric='cosine' \
    model_ckpt="best" \
    output_dir='evaluation_results/mdmspo_babel_to_humanml3d_lr1e-7_best'


# B -> K
HYDRA_FULL_ERROR=1 python eval_model.py \
    model=mdm_spo \
    evaluator=densetmr \
    data=kitml \
    model_checkpoint_dir='outputs/mdmspo_babel_to_kitml_smplrifke_lr1e-7_bs160' \
    evaluator_checkpoint_dir='outputs/densetmr_kitml_smplrifke' \
    data.clip_embedder=false \
    data.text_to_token_emb=false \
    data.text_to_sent_emb=false \
    distance_metric='cosine' \
    model_ckpt="best" \
    output_dir='evaluation_results/mdmspo_babel_to_kitml_lr1e-7_bs160_best'


# B -> M  (run after mdmspo_babel_to_motionx_smplrifke_lr1e-7_bs256 finishes)
HYDRA_FULL_ERROR=1 python eval_model.py \
    model=mdm_spo \
    evaluator=densetmr \
    data=motionx \
    model_checkpoint_dir='outputs/mdmspo_babel_to_motionx_smplrifke_lr1e-7_bs256' \
    evaluator_checkpoint_dir='outputs/densetmr_motionx_smplrifke' \
    data.clip_embedder=false \
    data.text_to_token_emb=false \
    data.text_to_sent_emb=false \
    distance_metric='cosine' \
    model_ckpt="best" \
    output_dir='evaluation_results/mdmspo_babel_to_motionx_lr1e-7_bs256_best'


# H -> B  (peaked at 0.71 — best ckpt saved before degradation)
HYDRA_FULL_ERROR=1 python eval_model.py \
    model=mdm_spo \
    evaluator=densetmr \
    data=babel \
    model_checkpoint_dir='outputs/mdmspo_humanml3d_to_babel_smplrifke_lr1e-6' \
    evaluator_checkpoint_dir='outputs/densetmr_babel_smplrifke' \
    data.clip_embedder=false \
    data.text_to_token_emb=false \
    data.text_to_sent_emb=false \
    distance_metric='cosine' \
    model_ckpt="best" \
    output_dir='evaluation_results/mdmspo_humanml3d_to_babel_lr1e-6_best'


# H -> K
HYDRA_FULL_ERROR=1 python eval_model.py \
    model=mdm_spo \
    evaluator=densetmr \
    data=kitml \
    model_checkpoint_dir='outputs/mdmspo_humanml3d_to_kitml_smplrifke_lr1e-6_ckptLast' \
    evaluator_checkpoint_dir='outputs/densetmr_kitml_smplrifke' \
    data.clip_embedder=false \
    data.text_to_token_emb=false \
    data.text_to_sent_emb=false \
    distance_metric='cosine' \
    model_ckpt="best" \
    output_dir='evaluation_results/mdmspo_humanml3d_to_kitml_lr1e-6_ckptLast_best'


# H -> M
HYDRA_FULL_ERROR=1 python eval_model.py \
    model=mdm_spo \
    evaluator=densetmr \
    data=motionx \
    model_checkpoint_dir='outputs/mdmspo_humanml3d_to_motionx_smplrifke_lr1e-6' \
    evaluator_checkpoint_dir='outputs/densetmr_motionx_smplrifke' \
    data.clip_embedder=false \
    data.text_to_token_emb=false \
    data.text_to_sent_emb=false \
    distance_metric='cosine' \
    model_ckpt="best" \
    output_dir='evaluation_results/mdmspo_humanml3d_to_motionx_lr1e-6_best'


# K -> H
HYDRA_FULL_ERROR=1 python eval_model.py \
    model=mdm_spo \
    evaluator=densetmr \
    data=humanml3d \
    model_checkpoint_dir='outputs/mdmspo_kitml_to_humanml3d_smplrifke_lr1e-7_bs256' \
    evaluator_checkpoint_dir='outputs/densetmr_humanml3d_smplrifke' \
    data.clip_embedder=false \
    data.text_to_token_emb=false \
    data.text_to_sent_emb=false \
    distance_metric='cosine' \
    model_ckpt="best" \
    output_dir='evaluation_results/mdmspo_kitml_to_humanml3d_lr1e-7_bs256_best'
