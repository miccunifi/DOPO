import sys
import torch
from Unimotion.utils.dist_util import dev
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

    device_id = 0
    device = torch.device('cuda:%d' % device_id if torch.cuda.is_available() else 'cpu')
    # device = "cpu"
    if device != "cpu" and device.type == 'cuda':
        torch.cuda.set_device(device)

    opt.device = device
    opt.mm_num_samples = 0

    # load evaluator
    eval_wrapper = EvaluatorModelWrapper(opt)

    opt.dataset_name = 'kit'  #TODO GROSSO COME UNA CASA questo codice è una merda!
    if device == "cpu": # mycode to make it work on CPU (in a reasonable amount of time)
        opt.mm_num_samples = 1
        opt.diversity_times = 5

    # load dataset 
    # gt_loader = get_dataset_loader(opt, opt.batch_size, mode='gt_eval', split='test')
    gen_dataset = get_dataset(opt, mode='eval_rl', split='test')

    if opt.dataset_name == 'kit':
        gt_opt = copy.deepcopy(opt)
        gt_opt.dataset_name = 'kit_22'
        gt_loader = get_dataset_loader(gt_opt, opt.batch_size, mode='gt_eval', split='test')

        opt_momask_generated = copy.deepcopy(opt)
        opt_momask_generated.dataset_name = 'kit_generated_momask'
        dataset_momask_generated = get_dataset_loader(opt_momask_generated, opt.batch_size, mode='gt_eval', split='test')
        
    elif opt.dataset_name == 't2m':
        gt_opt = copy.deepcopy(opt)
        gt_opt.dataset_name = 't2m'
        gt_loader = get_dataset_loader(gt_opt, opt.batch_size, mode='gt_eval', split='test')

        opt_momask_generated = copy.deepcopy(opt)
        opt_momask_generated.dataset_name = 'humanml3d_generated_momask'
        dataset_momask_generated = get_dataset_loader(opt_momask_generated, opt.batch_size, mode='gt_eval', split='test')

    # load model
    model = build_models(opt)
    ckpt_path = pjoin(opt.model_dir, opt.which_ckpt + '.tar')
    load_model_weights(model, ckpt_path, use_ema=opt.no_ema, device=device) #TODO GROSSO COME UNA CASA c'era un not prima di opt.no_ema

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

    eval_motion_loaders = {
        'text2motion': lambda: get_motion_loader(
            opt,
            opt.batch_size,
            pipeline,
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

    print("Pretrained")
    all_metrics = evaluation(eval_wrapper, gt_loader, eval_motion_loaders, log_file, opt.replication_times,
                             opt.diversity_times, opt.mm_num_times, run_mm=True)
    

    print("Model 4 - or best now")
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
    ckpt_path = "./Models/10Step_Guidance_LORA_VANILLA_t2m_to_kit_dpmsolver_sanityCheck/checkpoint_best.pth"
    checkpoint = torch.load(ckpt_path,map_location=device)
    diffusion_rl_.model.load_state_dict(checkpoint)
    diffusion_rl_.model.eval()
    
    eval_motion_loaders = {
        'text2motion': lambda: get_motion_loader(
            opt,
            opt.batch_size,
            diffusion_rl_,
            gen_dataset,
            opt.mm_num_samples,
            opt.mm_num_repeats,
        )
    }

    all_metrics = evaluation(eval_wrapper, gt_loader, dataset_momask_generated, eval_motion_loaders, log_file, opt.replication_times, opt.diversity_times, opt.mm_num_times, run_mm=True)
