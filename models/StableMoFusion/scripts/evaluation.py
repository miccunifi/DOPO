import sys
import torch
from motion_loader import get_dataset_loader, get_motion_loader
from datasets import get_dataset
from models import build_models
from eval import EvaluatorModelWrapper, evaluation
from utils.utils import *
from utils.model_load import load_model_weights
import os
from os.path import join as pjoin
from peft import LoraConfig, get_peft_model
from models.gaussian_diffusion import DiffusePipeline
from accelerate.utils import set_seed
import copy
from options.evaluate_options import TestOptions


if __name__ == '__main__':
    parser = TestOptions()
    opt = parser.parse()
    set_seed(0)

    # opt.batch_size = 128

    device_id = 0
    device = torch.device('cuda:%d' % device_id if torch.cuda.is_available() else 'cpu')
    # device = "cpu"
    if device != "cpu" and device.type == 'cuda':
        torch.cuda.set_device(device)

    opt.device = device
    # opt.mm_num_samples = 0

    # load evaluator
    opt_eval = copy.deepcopy(opt)
    opt_eval.dim_pose = 263
    opt_eval.dataset_name = 't2m'
    eval_wrapper = EvaluatorModelWrapper(opt_eval)

    # load model
    opt_model = copy.deepcopy(opt)
    opt_model.dim_pose = 263
    opt_model.dataset_name = 't2m'
    if opt.dataset_name == 'kit':
        opt_model.model_dir = './checkpoints/t2m/probabilmente_kit/model/'
        opt.meta_dir = './checkpoints/t2m/probabilmente_kit/meta' 
        ckpt_path_finetuned = "./Models/kit_to_human/checkpoint_best.pth"
    elif opt.dataset_name == 't2m':
        opt_model.model_dir = './checkpoints/t2m/t2m_condunet1d_batch64/model/'
        opt.meta_dir = './checkpoints/t2m/t2m_condunet1d_batch64/meta'
        ckpt_path_finetuned = "./Models/10Step_Guidance_LORA_VANILLA_t2m_to_kit_dpmsolver_sanityCheck/checkpoint_best.pth"
    else:
        raise ValueError("Unsupported dataset name: {}".format(opt.dataset_name))
    model = build_models(opt_model)
    ckpt_path_pretrained = pjoin(opt_model.model_dir, opt_model.which_ckpt + '.tar')
    load_model_weights(model, ckpt_path_pretrained, use_ema=opt_model.no_ema, device=device) #TODO GROSSO COME UNA CASA c'era un not prima di opt.no_ema
    # Create a pipeline for generation in diffusion model framework
    pipeline = DiffusePipeline(
        opt=opt,
        model=model,
        diffuser_name=opt.diffuser_name,
        device=device,
        num_inference_steps=opt.num_inference_steps,
        torch_dtype=torch.float32 if opt.no_fp16 else torch.float16)
    if device == "cpu": # mycode to make it work on CPU
        pipeline.model = pipeline.model.float()

    # Cross dataset
    if opt.dataset_name == 't2m':
        opt.dataset_name = 'kit_22'
    elif opt.dataset_name == 'kit':
        opt.dataset_name = 't2m'
    else:
        raise ValueError("Unsupported dataset name: {}".format(opt.dataset_name))
    
    # cpu debug mode
    if device == "cpu": # mycode to make it work on CPU (in a reasonable amount of time)
        opt.mm_num_samples = 1
        opt.diversity_times = 5

    # load dataset 
    # gt_loader = get_dataset_loader(opt, opt.batch_size, mode='gt_eval', split='test')
    gen_dataset = get_dataset(opt, mode='eval_rl', split='test')

    gt_motion_loaders = {}

    if opt.dataset_name == 'kit':
        gt_opt = copy.deepcopy(opt)
        gt_opt.dataset_name = 'kit_22'
        gt_loader = get_dataset_loader(gt_opt, opt.batch_size, mode='gt_eval', split='test')
        gt_motion_loaders['kit 22 gt'] = gt_loader

        opt_momask_generated = copy.deepcopy(opt)
        opt_momask_generated.dataset_name = 'kit_generated_momask'
        dataset_momask_generated = get_dataset_loader(opt_momask_generated, opt.batch_size, mode='gt_eval', split='test')
        gt_motion_loaders['momask'] = dataset_momask_generated

        opt_motiongpt_generated = copy.deepcopy(opt)
        opt_motiongpt_generated.dataset_name = 'kit_generated_motiongpt'
        dataset_motiongpt_generated = get_dataset_loader(opt_motiongpt_generated, opt.batch_size, mode='gt_eval', split='test')
        gt_motion_loaders['motiongpt'] = dataset_motiongpt_generated


    elif opt.dataset_name == 't2m':
        gt_opt = copy.deepcopy(opt)
        gt_opt.dataset_name = 't2m'
        gt_loader = get_dataset_loader(gt_opt, opt.batch_size, mode='gt_eval', split='test')
        gt_motion_loaders['t2m gt'] = gt_loader

        opt_momask_generated = copy.deepcopy(opt)
        opt_momask_generated.dataset_name = 'humanml3d_generated_momask'
        dataset_momask_generated = get_dataset_loader(opt_momask_generated, opt.batch_size, mode='gt_eval', split='test')
        gt_motion_loaders['momask'] = dataset_momask_generated

        opt_motiongpt_generated = copy.deepcopy(opt)
        opt_motiongpt_generated.dataset_name = 'humanml3d_generated_motiongpt'
        dataset_motiongpt_generated = get_dataset_loader(opt_motiongpt_generated, opt.batch_size, mode='gt_eval', split='test')
        gt_motion_loaders['motiongpt'] = dataset_motiongpt_generated


    diffusion_rl_ = copy.deepcopy(pipeline)
    # Apply LoRA only to UNet cross-attention
    lora_config = LoraConfig(
        inference_mode=True,
        r=4,
        lora_alpha=16,
        lora_dropout=0.0,
        bias="none",
        target_modules=["query", "key", "value"],
    )
    diffusion_rl_.model.unet = get_peft_model(diffusion_rl_.model.unet, lora_config)

    # Create a pipeline for generation in diffusion model framework
    checkpoint = torch.load(ckpt_path_finetuned,map_location=device)
    diffusion_rl_.model.load_state_dict(checkpoint)
    diffusion_rl_.model.eval()

    eval_motion_loaders = {
        'ours pretrained': lambda: get_motion_loader(
            opt,
            opt.batch_size,
            pipeline,
            gen_dataset,
            opt.mm_num_samples,
            opt.mm_num_repeats,
        ),
        'ours finetuned': lambda: get_motion_loader(
            opt,
            opt.batch_size,
            diffusion_rl_,
            gen_dataset,
            opt.mm_num_samples,
            opt.mm_num_repeats,
        )
    }

    save_dir = pjoin(opt.save_root, 'eval')
    os.makedirs(save_dir, exist_ok=True)
    if opt.no_ema:
        log_file = pjoin(save_dir, opt.diffuser_name) + f'_{str(opt.num_inference_steps)}setps.log'
    else:
        log_file = pjoin(save_dir, opt.diffuser_name) + f'_{str(opt.num_inference_steps)}steps_ema.log'

    if not os.path.exists(log_file):
        config_dict = dict(pipeline.scheduler.config)
        config_dict['no_ema'] = opt.no_ema
        with open(log_file, 'wt') as f:
            f.write('------------ Options -------------\n')
            for k, v in sorted(config_dict.items()):
                f.write('%s: %s\n' % (str(k), str(v)))
            f.write('-------------- End ----------------\n')

    all_metrics = evaluation(eval_wrapper, gt_motion_loaders, eval_motion_loaders, log_file, opt.replication_times,
                             opt.diversity_times, opt.mm_num_times, run_mm=True)
