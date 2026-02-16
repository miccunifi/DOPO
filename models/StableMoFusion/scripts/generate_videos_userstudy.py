import sys
import os
import torch
import numpy as np
from os.path import join as pjoin
from RL.reward_model import metric_fast
import utils.paramUtil as paramUtil
from utils.plot_script import *

from utils.utils import *
from utils.motion_process import recover_from_ric
from accelerate.utils import set_seed
from models.gaussian_diffusion import DiffusePipeline
from options.generate_options import GenerateOptions
from utils.model_load import load_model_weights
from motion_loader import get_dataset_loader
from models import build_models
import copy
from peft import LoraConfig, get_peft_model
from TMR.mtt.load_tmr_model import load_tmr_model_easy, easy_forward, load_tmr_model_complete


if __name__ == '__main__':


    parser = GenerateOptions()
    opt = parser.parse()
    set_seed(opt.seed)
    device_id = opt.gpu_id
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    # device = "cpu"
    opt.device = device

    experiment = "Manipulation" #  "Human2Kit" # TODO va cambiato ogni volta a mano
    # Human 2 Kit
    if experiment == "Human2Kit":
        ckpt_path_finetuned = "./Models/10Step_Guidance_LORA_VANILLA_t2m_to_kit_dpmsolver_sanityCheck/checkpoint_best.pth"
        ckpt_path = pjoin(opt.model_dir, opt.which_ckpt + '.tar')  
    # Kit 2 Human
    elif experiment == "Kit2Human":
        ckpt_path_finetuned = "./Models/kit_to_human/checkpoint_best.pth"
        ckpt_path = "./checkpoints/t2m/probabilmente_kit/model/latest.tar"
    # Manipulation
    elif experiment == "Manipulation":
        ckpt_path_finetuned = "./Models/Manipulation_2/checkpoint_best.pth"
        ckpt_path = "./checkpoints/t2m/manipulation_2/model/latest.tar"


    tmr_forward_complete = load_tmr_model_complete(device="cpu", dataset="tmr_humanml3d_kitml_guoh3dfeats")

    assert opt.dataset_name == 't2m' or 'kit'

    if opt.input_text != '':
        with open(opt.input_text.replace("prompts","ids"), 'r') as fr:
            ids = [line.strip() for line in fr.readlines()]
        with open(opt.input_text, 'r') as fr:
            texts = [line.strip() for line in fr.readlines()]
        opt.num_samples = len(texts)
        if opt.input_lens != '':
            with open(opt.input_lens, 'r') as fr:
                motion_lens = [int(line.strip()) for line in fr.readlines()]
            assert len(texts)==len(motion_lens), f'Please ensure that the motion length in {opt.input_lens} corresponds to the text in {opt.input_text}.'
        else:
            motion_lens = [opt.motion_length * opt.fps for _ in range(opt.num_samples)]
    # Or usining texts in dataset 
    else:
        gen_datasetloader = get_dataset_loader(opt, opt.num_samples, mode='hml_gt',split='test')
        texts, _, motion_lens = next(iter(gen_datasetloader))

    print(f"PRE Trained Model")
    # load model
    model = build_models(opt)
    niter = load_model_weights(model, ckpt_path, use_ema=not opt.no_ema, device=device)

    from types import SimpleNamespace

    c = SimpleNamespace()
    c.reward_scale = 1

    # Create a pipeline for generation in diffusion model framework
    diffusion_pre_trained = DiffusePipeline(
        opt = opt,
        model = model, 
        diffuser_name = opt.diffuser_name, 
        device=device,
        num_inference_steps=opt.num_inference_steps,
        torch_dtype=torch.float16,
        )
    
    # generate
    pred_motions_pre_trained, _ = diffusion_pre_trained.generate(texts, torch.LongTensor([int(x) for x in motion_lens]))
    pred_motions_pre_trained = [motion.detach().cpu().numpy() for motion in pred_motions_pre_trained]
    tmr_pre_trained, _ = metric_fast(tmr_forward_complete, pred_motions_pre_trained, texts, c, device=device)

    print(f"POST Trained Model")
    diffusion_post_trained = copy.deepcopy(diffusion_pre_trained)
    # Apply LoRA only to UNet cross-attention
    lora_config = LoraConfig(
        inference_mode=True,
        r=4,
        lora_alpha=16,
        lora_dropout=0.0,
        bias="none",
        target_modules=["query", "key", "value"],
    )
    diffusion_post_trained.model.unet = get_peft_model(diffusion_post_trained.model.unet, lora_config)

    # Create a pipeline for generation in diffusion model framework
    checkpoint = torch.load(ckpt_path_finetuned,map_location=device)
    if type(checkpoint) is dict and "state_dict" in checkpoint:
        checkpoint = checkpoint["state_dict"]
    diffusion_post_trained.model.load_state_dict(checkpoint)
    diffusion_post_trained.model.eval()

    pred_motions_post_trained, _ = diffusion_post_trained.generate(texts, torch.LongTensor([int(x) for x in motion_lens]))
    pred_motions_post_trained = [motion.detach().cpu().numpy() for motion in pred_motions_post_trained]
    tmr_post_trained, _ = metric_fast(tmr_forward_complete, pred_motions_post_trained, texts, c, device=device)

    # make save dir
    out_path = opt.output_dir
    if out_path == '':
        out_path = pjoin(opt.save_root,'samples_{}_seed{}'.format(niter, opt.seed))
        if opt.footskate_cleanup:
            out_path += '_with_footskate_cleanup'
        if opt.text_prompt != '':
            out_path += '_' + opt.text_prompt.replace(' ', '_').replace('.', '')
        elif opt.input_text != '':
            out_path += '_' + os.path.basename(opt.input_text).replace('.txt', '').replace(' ', '_').replace('.', '')
    os.makedirs(out_path,exist_ok=True)

    import pandas as pd
    import os

    # Create DataFrame
    df = pd.DataFrame({
        "ID": ids,
        "Pre-trained": tmr_pre_trained,
        "Post-trained": tmr_post_trained
    })

    # Ensure output folder exists
    os.makedirs(out_path, exist_ok=True)

    # Save to Excel if possible, otherwise CSV
    try:
        df.to_excel(os.path.join(out_path, "table.xlsx"), index=False)
        df.to_csv(out_path + "/table.csv", index=False)
    except ModuleNotFoundError:
        print("⚠️ openpyxl not installed, saving as CSV instead.")
        df.to_csv(os.path.join(out_path, "table.csv"), index=False)
        
    treshold = 0.02
    indices = [i for i, (pre, post) in enumerate(zip(tmr_pre_trained, tmr_post_trained)) if True] # [i for i, (pre, post) in enumerate(zip(tmr_pre_trained, tmr_post_trained)) if post - pre > treshold]

    # Convert the generated motion representaion into 3D joint coordinates and save as npy file
    npy_dir = pjoin(out_path, 'joints_npy')
    os.makedirs(npy_dir,exist_ok=True)
    print(f"saving results npy file (3d joints) to [{npy_dir}]")
    mean = np.load(pjoin(opt.meta_dir, 'mean.npy'))
    std = np.load(pjoin(opt.meta_dir, 'std.npy'))
    samples_pre = []
    samples_post = []
    for i in indices:
        idx = ids[i]
        motion_pre = pred_motions_pre_trained[i]
        motion_post = pred_motions_post_trained[i]

        motion_pre = motion_pre * std + mean
        motion_post = motion_post * std + mean
        npy_name_pre = f'{str(idx)}_pre.npy'
        npy_name_post = f'{str(idx)}_post.npy'
        # 1. recover 3d joints representation by ik
        motion_pre = recover_from_ric(torch.from_numpy(motion_pre).float(), opt.joints_num)
        motion_post = recover_from_ric(torch.from_numpy(motion_post).float(), opt.joints_num)
        
        # 2. put on Floor (Y axis)
        floor_height_pre = motion_pre.min(dim=0)[0].min(dim=0)[0][1]
        floor_height_post = motion_post.min(dim=0)[0].min(dim=0)[0][1]
        motion_pre[:, :, 1] -= floor_height_pre
        motion_post[:, :, 1] -= floor_height_post
        motion_pre = motion_pre.numpy()
        motion_post = motion_post.numpy()
        # 3. remove jitter
        motion_pre = motion_temporal_filter(motion_pre, sigma=1)
        motion_post = motion_temporal_filter(motion_post, sigma=1)
        # 4. save
        np.save(pjoin(npy_dir, npy_name_pre), motion_pre)
        np.save(pjoin(npy_dir, npy_name_post), motion_post)

        samples_pre.append(motion_pre)
        samples_post.append(motion_post)

    # save the text and length conditions used for this generation
    with open(pjoin(out_path, 'results_texts.txt'), 'w') as fw:
        fw.write('\n'.join([texts[i] for i in indices]))
    with open(pjoin(out_path, 'results_lens.txt'), 'w') as fw:
        fw.write('\n'.join([str(l) for l in [motion_lens[j] for j in indices]]))
    with open(pjoin(out_path, 'results_ids.txt'), 'w') as fw:
        fw.write('\n'.join([str(l) for l in [ids[j] for j in indices]]))
    with open(pjoin(out_path, 'results.txt'), 'w') as fw:
        fw.write('\n'.join([str(l) for l in [f"{ids[j]} - {texts[i]} - {motion_lens[j]}" for j in indices]]))
    
    # skeletal animation visualization
    print(f"saving motion videos to [{out_path}]...")
    count = 0
    for i in indices:
        idx = ids[i]
        motion_pre = samples_pre[count]
        motion_post = samples_post[count]
        count += 1
        title = texts[i]
        fname_pre = f'{str(idx)}_pre.mp4'
        fname_post = f'{str(idx)}_post.mp4'
        kinematic_tree = paramUtil.t2m_kinematic_chain if (opt.dataset_name == 't2m') else paramUtil.kit_kinematic_chain
        plot_3d_motion(pjoin(out_path, fname_pre), kinematic_tree, motion_pre, title=title, fps=opt.fps, radius=opt.radius)
        plot_3d_motion(pjoin(out_path, fname_post), kinematic_tree, motion_post, title=title, fps=opt.fps, radius=opt.radius)

