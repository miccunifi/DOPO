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

    opt.batch_size = 128

    experiment = "manipulation"
    # experiment = "performance"
    
    device_id = 0
    device = torch.device('cuda:%d' % device_id if torch.cuda.is_available() else 'cpu')
    # device = "cpu"
    if device != "cpu" and device.type == 'cuda':
        torch.cuda.set_device(device)
    print("Using device:", device)

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
        
    if experiment == "manipulation":
        print("MANIPULATION")
        ckpt_path_finetuned = "./Models/Manipulation/checkpoint_best.pth"
        ckpt_path_pretrained = "./checkpoints/t2m/manipulation/model/latest.tar"
        ckpt_path_pretrained_oracle = "./checkpoints/t2m/t2m_condunet1d_batch64/model/latest.tar"
        # opt_model.dataset_name = 'manipulation'
        # opt_eval.dataset_name = 'manipulation'
        # opt.dataset_name = 'manipulation'
    elif experiment == "posturebalance":
        print("POSTURE - BALANCE")
        ckpt_path_finetuned = "./Models/PostureBalance/checkpoint_best.pth"
        ckpt_path_pretrained = "./checkpoints/t2m/posturebalance/model/latest.tar"
        ckpt_path_pretrained_oracle = "./checkpoints/t2m/t2m_condunet1d_batch64/model/latest.tar"
        opt_model.dataset_name = 'posture'
        # opt_eval.dataset_name = 'posture'
        opt.dataset_name = 'posture'
    elif experiment == "performance":
        print("PERFORMANCE")
        ckpt_path_finetuned = "./Models/Performance/checkpoint_best.pth"
        ckpt_path_pretrained = "./checkpoints/t2m/performance/model/latest.tar"
        ckpt_path_pretrained_oracle = "./checkpoints/t2m/t2m_condunet1d_batch64/model/latest.tar"
        opt_model.dataset_name = 'performance'
        # opt_eval.dataset_name = 'performance'
        opt.dataset_name = 'performance'

    # opt_model.model_dir = './checkpoints/t2m/t2m_condunet1d_batch64/model/'
    # opt.meta_dir = './checkpoints/t2m/t2m_condunet1d_batch64/meta'
    opt_model.model_dir = os.path.dirname(ckpt_path_pretrained)
    opt.meta_dir = os.path.dirname(opt_model.model_dir) + '/meta'
    opt.save_root = os.path.dirname(opt_model.model_dir)    

    print("Loading pretrained on split...")
    model = build_models(opt_model)
    # ckpt_path_pretrained = pjoin(opt_model.model_dir, opt_model.which_ckpt + '.tar')
    load_model_weights(model, ckpt_path_pretrained, use_ema=opt_model.no_ema, device=device) #TODO GROSSO COME UNA CASA c'era un not prima di opt.no_ema
    # Create a pipeline for generation in diffusion model framework
    pipeline = DiffusePipeline(
        opt=opt,
        model=model,
        diffuser_name=opt.diffuser_name,
        device=device,
        num_inference_steps=opt.num_inference_steps,
        torch_dtype=torch.float32 if opt.no_fp16 else torch.float16)
    if device == "cpu" or device == torch.device("cpu"): # mycode to make it work on CPU
        pipeline.model = pipeline.model.float()
    
    # cpu debug mode
    if device == "cpu" or device == torch.device("cpu"): # mycode to make it work on CPU (in a reasonable amount of time)
        opt.mm_num_samples = 1
        opt.diversity_times = 5

    # load dataset 
    # gt_loader = get_dataset_loader(opt, opt.batch_size, mode='gt_eval', split='test')
    opt_gen_dataset = copy.deepcopy(opt)
    opt_gen_dataset.dataset_name = 'manipulation'
    gen_dataset = get_dataset(opt_gen_dataset, mode='eval_rl', split='test')

    gt_motion_loaders = {}

    print("Loading finetuned on split...")
    gt_split_opt = copy.deepcopy(opt)
    gt_split_opt.dataset_name = experiment
    gt_manipulation_loader = get_dataset_loader(gt_split_opt, opt.batch_size, mode='gt_eval', split='test')
    gt_motion_loaders[f't2m {experiment} gt'] = gt_manipulation_loader

    diffusion_rl_split = copy.deepcopy(pipeline)
    # Apply LoRA only to UNet cross-attention
    lora_config = LoraConfig(
        inference_mode=True,
        r=4,
        lora_alpha=16,
        lora_dropout=0.0,
        bias="none",
        target_modules=["query", "key", "value"],
    )
    diffusion_rl_split.model.unet = get_peft_model(diffusion_rl_split.model.unet, lora_config)

    # Create a pipeline for generation in diffusion model framework
    checkpoint_finetuned = torch.load(ckpt_path_finetuned, map_location=device)
    diffusion_rl_split.model.load_state_dict(checkpoint_finetuned)
    diffusion_rl_split.model.eval()
    if device == "cpu" or device == torch.device("cpu"): # mycode to make it work on CPU
        diffusion_rl_split.model = diffusion_rl_split.model.float()

    print("Loading pretrained on ALL...")
    model_oracle = build_models(opt_model)
    # ckpt_path_pretrained = pjoin(opt_model.model_dir, opt_model.which_ckpt + '.tar')
    load_model_weights(model_oracle, ckpt_path_pretrained_oracle, use_ema=opt_model.no_ema, device=device) #TODO GROSSO COME UNA CASA c'era un not prima di opt.no_ema
    # Create a pipeline for generation in diffusion model framework
    pipeline_oracle = DiffusePipeline(
        opt=opt,
        model=model_oracle,
        diffuser_name=opt.diffuser_name,
        device=device,
        num_inference_steps=opt.num_inference_steps,
        torch_dtype=torch.float32 if opt.no_fp16 else torch.float16)
    if device == "cpu" or device == torch.device("cpu"): # mycode to make it work on CPU
        pipeline_oracle.model = pipeline_oracle.model.float()

    eval_motion_loaders = {
        # 'ours pretrained':  lambda: get_motion_loader(
        #     opt_eval, # opt
        #     opt.batch_size,
        #     pipeline,
        #     gen_dataset,
        #     opt.mm_num_samples,
        #     opt.mm_num_repeats,
        # ),
        # 'ours finetuned': lambda: get_motion_loader(
        #     opt_eval, # opt
        #     opt.batch_size,
        #     diffusion_rl_split,
        #     gen_dataset,
        #     opt.mm_num_samples,
        #     opt.mm_num_repeats,
        # ),
        'pretrained on all data':  lambda: get_motion_loader(
            opt_eval, # opt
            opt.batch_size,
            pipeline_oracle,
            gen_dataset,
            opt.mm_num_samples,
            opt.mm_num_repeats,
        ),
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

    all_metrics = evaluation(eval_wrapper, gt_motion_loaders, eval_motion_loaders, log_file, opt.replication_times,opt.diversity_times, opt.mm_num_times, run_mm=True)

    ckpt_path_finetuned_manipulation = "./Models/Manipulation/checkpoint_best.pth"
