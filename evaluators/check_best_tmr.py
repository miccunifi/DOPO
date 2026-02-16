import torch
import numpy as np
from pathlib import Path
from hydra.utils import instantiate
from src.config import read_config
from tqdm import tqdm
import pandas as pd
import json
import os 
from src.model.tmr import TMR
import hydra


def extract_ckpt(run_dir, ckpt_name="last"):
    import torch

    ckpt_path = os.path.join(run_dir, f"logs/checkpoints/{ckpt_name}.ckpt")

    extracted_path = os.path.join(run_dir, f"{ckpt_name}_weights")
    os.makedirs(extracted_path, exist_ok=True)

    new_path_template = os.path.join(extracted_path, "{}.pt")
    ckpt_dict = torch.load(ckpt_path)
    state_dict = ckpt_dict["state_dict"]
    module_names = list(set([x.split(".")[0] for x in state_dict.keys()]))

    # should be ['motion_encoder', 'text_encoder', 'motion_decoder'] for example
    for module_name in module_names:
        path = new_path_template.format(module_name)
        sub_state_dict = {
            ".".join(x.split(".")[1:]): y.cpu()
            for x, y in state_dict.items()
            if x.split(".")[0] == module_name
        }
        torch.save(sub_state_dict, path)


def extract(coef, t, tensor):
    shape = tensor.shape
    return coef[t].reshape(shape[0], *((1,) * (len(shape) - 1))).to(tensor)


def load_model_from_cfg(cfg, ckpt_name="last", device="cpu", eval_mode=True):
    import src.prepare  # noqa
    import torch

    run_dir = cfg.run_dir
    model = hydra.utils.instantiate(cfg.model)

    # Loading modules one by one
    # motion_encoder / text_encoder / text_decoder
    pt_path = os.path.join(run_dir, f"{ckpt_name}_weights")

    if not os.path.exists(pt_path):
        print("The extracted model is not found. Split into submodules..")
        extract_ckpt(run_dir, ckpt_name)

    for fname in os.listdir(pt_path):
        module_name, ext = os.path.splitext(fname)

        if ext != ".pt":
            continue

        module = getattr(model, module_name, None)
        if module is None:
            continue

        module_path = os.path.join(pt_path, fname)
        state_dict = torch.load(module_path)
        module.load_state_dict(state_dict)
        print(f"    {module_name} loaded")

    print("Loading previous checkpoint done")
    model = model.to(device)
    print(f"Put the model on {device}")
    if eval_mode:
        model = model.eval()
        print("Put the model in eval mode")
    return model


def load_normalizer(dataset_name):
    """Load normalizer for denormalization based on dataset"""
    from our_loader_motion import Normalizer
    
    stats_paths = {
        # "babel": "stats/babel/guoh3dfeats",
        "humanml": "stats/humanml3d/guoh3dfeats",
        # "kitml": "stats/kitml/guoh3dfeats",
        # "motionx": "stats/motionx/guoh3dfeats",
    }
    
    base_dir = stats_paths.get(dataset_name, f"stats/{dataset_name}/guoh3dfeats")
    
    try:
        normalizer = Normalizer(base_dir=base_dir, eps=1e-12, disable=False)
        return normalizer
    except Exception as e:
        print(f"Warning: Could not load normalizer for {dataset_name}: {e}")
        return None


def all_contrastive_metrics(sim_matrix, rounding=2):
    t2m_ranks = []
    for i in range(len(sim_matrix)):
        rank = np.where(np.argsort(-sim_matrix[i]) == i)[0][0] + 1
        t2m_ranks.append(rank)
    
    m2t_ranks = []
    for i in range(len(sim_matrix)):
        rank = np.where(np.argsort(-sim_matrix[:, i]) == i)[0][0] + 1
        m2t_ranks.append(rank)
    
    t2m_ranks = np.array(t2m_ranks)
    m2t_ranks = np.array(m2t_ranks)
    
    metrics = {
        "t2m_R1": np.mean(t2m_ranks <= 1) * 100,
        "t2m_R5": np.mean(t2m_ranks <= 5) * 100,
        "t2m_R10": np.mean(t2m_ranks <= 10) * 100,
        "t2m_MedR": np.median(t2m_ranks),
        "m2t_R1": np.mean(m2t_ranks <= 1) * 100,
        "m2t_R5": np.mean(m2t_ranks <= 5) * 100,
        "m2t_R10": np.mean(m2t_ranks <= 10) * 100,
        "m2t_MedR": np.median(m2t_ranks),
    }
    
    if rounding is not None:
        metrics = {k: round(v, rounding) for k, v in metrics.items()}
    
    return metrics


def compute_sim_matrix(model, dataset, keyids, batch_size=256, time=0, save_samples=False, output_dir=None, normalizer=None):
    from src.model.tmr import get_sim_matrix
    
    device = model.device
    nsplit = int(np.ceil(len(keyids) / batch_size))
    
    with torch.inference_mode():
        all_data = [dataset.load_keyid(keyid) for keyid in keyids]
        all_data_splitted = np.array_split(all_data, nsplit)
        
        latent_texts = []
        latent_motions = []
        sent_embs = []
        
        for batch_idx, data_batch in enumerate(tqdm(all_data_splitted, leave=False, desc="Computing embeddings")):
            batch = dataset.collate_fn(list(data_batch))
            
            def to_device(x):
                if isinstance(x, torch.Tensor):
                    return x.to(device)
                elif isinstance(x, dict):
                    return {k: to_device(v) for k, v in x.items()}
                elif isinstance(x, list):
                    return [to_device(v) for v in x]
                return x
            
            batch = to_device(batch)
            
            text_x_dict = batch["text_x_dict"]
            motion_x_dict = batch["motion_x_dict"]
            sent_emb = batch["sent_emb"]
            
            # Clip time to 0-99 range
            t_val = min(time, 99)
            t = torch.tensor(t_val, device=motion_x_dict["x"].device).repeat(
                motion_x_dict["x"].shape[0]
            )
            motion_x_dict["t"] = t
            
            noise = torch.randn_like(motion_x_dict['x'])
            linear_scale = 1
            sqrt_alphas_cumprod = torch.from_numpy(np.load('sqrt_alphas_cumprod.npy')).to(device)
            sqrt_one_minus_alphas_cumprod = torch.from_numpy(np.load('sqrt_one_minus_alphas_cumprod.npy')).to(device)
            mean = linear_scale * extract(sqrt_alphas_cumprod, t, motion_x_dict['x']) * motion_x_dict['x']
            sigma = extract(sqrt_one_minus_alphas_cumprod, t, motion_x_dict['x'])
            motion_x_dict['x'] = mean + sigma * noise
            
            # Save first batch samples if requested
            if save_samples and batch_idx == 0 and output_dir is not None:
                sample_dir = output_dir / f"samples_t{time}"
                sample_dir.mkdir(exist_ok=True, parents=True)
                
                for i in range(min(3, motion_x_dict["x"].shape[0])):
                    try:
                        motion_sample = motion_x_dict["x"][i].cpu()
                        
                        if normalizer is not None:
                            motion_sample = normalizer.inverse(motion_sample)
                        
                        motion_sample = motion_sample.numpy()
                        np.save(sample_dir / f"motion_sample_{i}.npy", motion_sample)
                        
                        if "text" in batch:
                            with open(sample_dir / f"text_sample_{i}.txt", "w") as f:
                                f.write(str(batch["text"][i]))
                    except Exception as e:
                        print(f"Warning: Could not save sample {i}: {e}")
            
            latent_text = model.encode(text_x_dict, sample_mean=True)
            latent_motion = model.encode(motion_x_dict, sample_mean=True)
            
            latent_texts.append(latent_text)
            latent_motions.append(latent_motion)
            sent_embs.append(sent_emb)
        
        latent_texts = torch.cat(latent_texts)
        latent_motions = torch.cat(latent_motions)
        sent_embs = torch.cat(sent_embs)
        sim_matrix = get_sim_matrix(latent_texts, latent_motions)
    
    return {
        "sim_matrix": sim_matrix.cpu().numpy(),
        "sent_emb": sent_embs.cpu().numpy(),
    }

def evaluate_model_on_dataset(
    model, dataset, batch_size=31, num_samples=None, seed=0, time=0, 
    save_samples=False, output_dir=None, normalizer=None
):
    keyids = sorted(dataset.keyids)
    
    if num_samples is not None:
        keyids = keyids[:num_samples]
    
    N = len(keyids)
    
    idx = np.arange(N)
    np.random.seed(seed)
    np.random.shuffle(idx)
    
    idx_batches = [
        idx[batch_size * i : batch_size * (i + 1)]
        for i in range(len(keyids) // batch_size)
    ]
    
    all_results = []
    for batch_idx, idx_batch in enumerate(tqdm(idx_batches, desc="Processing batches")):
        save_this_batch = save_samples and batch_idx == 0
        result = compute_sim_matrix(
            model,
            dataset,
            np.array(keyids)[idx_batch],
            batch_size=batch_size,
            time=time,
            save_samples=save_this_batch,
            output_dir=output_dir,
            normalizer=normalizer
        )
        all_results.append(result)
    
    all_metrics = []
    for result in all_results:
        sim_matrix = result["sim_matrix"]
        metrics = all_contrastive_metrics(sim_matrix, rounding=None)
        all_metrics.append(metrics)
    
    avg_metrics = {}
    for key in all_metrics[0].keys():
        avg_metrics[key] = float(np.mean([m[key] for m in all_metrics]))
    
    return avg_metrics


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # motionx first
    models = {
        # "babel": "/deck/groups/MotionRL/DenseTMR/outputs/babeltmrpp",
        "humanml": "/deck/groups/MotionRL/TMR/models/tmr_humanml3d_guoh3dfeats",
        "kitml": "/deck/groups/MotionRL/TMR/models/tmr_kitml_guoh3dfeats",
        # "motionx": "/deck/groups/MotionRL/DenseTMR/outputs/motionXtmrpp",
    }
    # timesteps = [0, 25, 50, 75, 99]
    timesteps = [0, 5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 55, 60 , 65, 70, 75, 80, 85, 90, 95, 99]

    all_results = {}
    output_dir = Path("evaluation_results")
    output_dir.mkdir(exist_ok=True)
    
    for model_name, model_dir in models.items():
        print(f"\n{'='*80}")
        print(f"Loading model: {model_name}")
        print(f"{'='*80}")
        
        model_path = Path(model_dir)
        cfg = read_config(model_path)
        
        model = load_model_from_cfg(cfg, "last", eval_mode=True, device=device)
        
        all_results[model_name] = {}
        
        for dataset_name in models.keys():
            print(f"\n{'-'*80}")
            print(f"Evaluating on dataset: {dataset_name}")
            print(f"{'-'*80}")
            
            dataset_dir = Path(models[dataset_name])
            dataset_cfg = read_config(dataset_dir)
            dataset_cfg.data.path = "/deck/datasets/Black_datasets(H_K_B)/annotations/humanml3d"
            dataset_cfg.data.text_to_sent_emb.path =  "/deck/datasets/Black_datasets(H_K_B)/annotations/humanml3d"
            dataset_cfg.data.text_to_token_emb.path =  "/deck/datasets/Black_datasets(H_K_B)/annotations/humanml3d"
            dataset_cfg.data.motion_loader.base_dir  =  "/deck/datasets/Black_datasets(H_K_B)/motions/guoh3dfeats"
            # val_dataset = instantiate(dataset_cfg.data, split="val") # TODO FIX THIS
            val_dataset = instantiate(dataset_cfg.data, split="test")

            # Load normalizer for this dataset
            # normalizer = load_normalizer(dataset_name)
            normalizer = None
            all_results[model_name][dataset_name] = {}
            
            sample_output_dir = output_dir / "samples" / f"{model_name}_on_{dataset_name}"
            
            for time in timesteps:
                print(f"Timestep: {time}")
                
                save_samples = (model_name == "motionx")
                
                metrics = evaluate_model_on_dataset(
                    model,
                    val_dataset,
                    batch_size=31,
                    num_samples=None,
                    seed=0,
                    time=time,
                    save_samples=save_samples,
                    output_dir=sample_output_dir,
                    normalizer=normalizer
                )
                
                all_results[model_name][dataset_name][f"t{time}"] = metrics
                
                print(f"Results at t={time}:")
                for metric_name, value in metrics.items():
                    print(f"  {metric_name}: {value:.2f}")
        
        del model
        torch.cuda.empty_cache()
    
    with open(output_dir / "results.json", "w") as f:
        json.dump(all_results, f, indent=2)
    
    create_summary_tables(all_results, output_dir)
    
    print(f"\n{'='*80}")
    print(f"Results saved to: {output_dir}")
    print(f"{'='*80}")


def create_summary_tables(results, output_dir):
    timesteps = list(next(iter(next(iter(results.values())).values())).keys())
    
    for timestep in timesteps:
        rows = []
        for model_name, model_results in results.items():
            for dataset_name, dataset_results in model_results.items():
                if timestep in dataset_results:
                    metrics = dataset_results[timestep]
                    row = {
                        "Model": model_name,
                        "Dataset": dataset_name,
                        **{k: f"{v:.2f}" for k, v in metrics.items()},
                    }
                    rows.append(row)
        
        df = pd.DataFrame(rows)
        df.to_csv(output_dir / f"results_{timestep}.csv", index=False)
        
        print(f"\n{timestep} Results:")
        print("\n" + df.to_string(index=False))
    
    print("\n" + "="*80)
    print("Generalization Summary (Average T2M R@1 on other datasets):")
    print("="*80)

    #timesteps = [0, 10, 20, 30, 40, 50, 60 , 70, 80, 90, 99]
    timesteps = [0, 5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 55, 60 , 65, 70, 75, 80, 85, 90, 95, 99]


    for t in timesteps:
        timestep = f"t{t}"
        print(f"\nTimestep {t}:")
        for model_name in results.keys():
            other_datasets = [d for d in results.keys() if d != model_name]
            avg_r1 = np.mean([
                results[model_name][dataset][timestep]["t2m_R1"]
                for dataset in other_datasets
            ])
            print(f"  {model_name}: {avg_r1:.2f}")


if __name__ == "__main__":
    main()