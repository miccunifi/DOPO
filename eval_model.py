import logging
import hydra
from omegaconf import DictConfig
from hydra.utils import instantiate
import torch
import numpy as np
from pathlib import Path
import json
from tqdm import tqdm
from scipy import linalg
from scipy.spatial.distance import cdist

from src.config import read_config
from src.utils import set_preload_false

logger = logging.getLogger(__name__)


def calculate_activation_statistics_normalized(activations, normalize=True):
    """
    with normalization (as TMR embeddings should be used with norm 1)
    Params:
    -- activation: num_samples x dim_feat
    Returns:
    -- mu: dim_feat
    -- sigma: dim_feat x dim_feat
    """
    if normalize:
        activations = activations.numpy() / np.linalg.norm(activations, axis=-1)[:, None]
    mu = np.mean(activations, axis=0)
    cov = np.cov(activations, rowvar=False)
    return mu, cov



def calculate_frechet_distance(mu1, sigma1, mu2, sigma2, eps=1e-6):
    """Numpy implementation of the Frechet Distance.
    The Frechet distance between two multivariate Gaussians X_1 ~ N(mu_1, C_1)
    and X_2 ~ N(mu_2, C_2) is
            d^2 = ||mu_1 - mu_2||^2 + Tr(C_1 + C_2 - 2*sqrt(C_1*C_2)).
    Stable version by Dougal J. Sutherland.
    Params:
    -- mu1   : Numpy array containing the activations of a layer of the
               inception net (like returned by the function 'get_predictions')
               for generated samples.
    -- mu2   : The sample mean over activations, precalculated on an
               representative data set.
    -- sigma1: The covariance matrix over activations for generated samples.
    -- sigma2: The covariance matrix over activations, precalculated on an
               representative data set.
    Returns:
    --   : The Frechet Distance.
    """

    mu1 = np.atleast_1d(mu1)
    mu2 = np.atleast_1d(mu2)

    sigma1 = np.atleast_2d(sigma1)
    sigma2 = np.atleast_2d(sigma2)

    assert mu1.shape == mu2.shape, \
        'Training and test mean vectors have different lengths'
    assert sigma1.shape == sigma2.shape, \
        'Training and test covariances have different dimensions'

    diff = mu1 - mu2

    # Product might be almost singular
    covmean, _ = linalg.sqrtm(sigma1.dot(sigma2), disp=False)
    if not np.isfinite(covmean).all():
        msg = ('fid calculation produces singular product; '
               'adding %s to diagonal of cov estimates') % eps
        print(msg)
        offset = np.eye(sigma1.shape[0]) * eps
        covmean = linalg.sqrtm((sigma1 + offset).dot(sigma2 + offset))

    # Numerical error might give slight imaginary component
    if np.iscomplexobj(covmean):
        if not np.allclose(np.diagonal(covmean).imag, 0, atol=1e-3):
            m = np.max(np.abs(covmean.imag))
            raise ValueError('Imaginary component {}'.format(m))
        covmean = covmean.real

    tr_covmean = np.trace(covmean)

    return (diff.dot(diff) + np.trace(sigma1) +
            np.trace(sigma2) - 2 * tr_covmean)


def calculate_diversity(embeddings, diversity_times=300):
    """Calculate diversity of embeddings."""
    embeddings = embeddings.cpu().numpy() if isinstance(embeddings, torch.Tensor) else embeddings
    num_samples = embeddings.shape[0]
    
    if num_samples < diversity_times:
        diversity_times = num_samples
    
    first_indices = np.random.choice(num_samples, diversity_times, replace=False)
    second_indices = np.random.choice(num_samples, diversity_times, replace=False)
    dist = linalg.norm(embeddings[first_indices] - embeddings[second_indices], axis=1)
    
    return dist.mean()


def calculate_multimodality(motion_embeddings, text_labels, multimodality_times=20):
    """Calculate multimodality: diversity of motions for the same text."""
    motion_embeddings = motion_embeddings.cpu().numpy() if isinstance(motion_embeddings, torch.Tensor) else motion_embeddings
    unique_texts = np.unique(text_labels)
    multimodality_scores = []
    
    for text_idx in unique_texts:
        motion_mask = text_labels == text_idx
        text_motions = motion_embeddings[motion_mask]
        
        if len(text_motions) < 2:
            continue
        
        num_samples = min(multimodality_times, len(text_motions))
        first_indices = np.random.choice(len(text_motions), num_samples, replace=True)
        second_indices = np.random.choice(len(text_motions), num_samples, replace=True)
        dist = linalg.norm(text_motions[first_indices] - text_motions[second_indices], axis=1)
        multimodality_scores.append(dist.mean())
    
    return np.mean(multimodality_scores) if multimodality_scores else 0.0


def calculate_retrieval_metrics(text_embeddings, motion_embeddings, distance_metric='euclidean'):
    """Calculate retrieval metrics: R@1, R@2, R@3, R@5, R@10, MedR for both T2M and M2T."""
    text_embeddings = text_embeddings.cpu().numpy() if isinstance(text_embeddings, torch.Tensor) else text_embeddings
    motion_embeddings = motion_embeddings.cpu().numpy() if isinstance(motion_embeddings, torch.Tensor) else motion_embeddings
    
    if distance_metric == 'euclidean':
        dist_matrix = cdist(text_embeddings, motion_embeddings, metric='euclidean')
    elif distance_metric == 'cosine':
        text_norm = text_embeddings / (np.linalg.norm(text_embeddings, axis=1, keepdims=True) + 1e-8)
        motion_norm = motion_embeddings / (np.linalg.norm(motion_embeddings, axis=1, keepdims=True) + 1e-8)
        dist_matrix = 1 - np.dot(text_norm, motion_norm.T)
    else:
        raise ValueError(f"Unknown distance metric: {distance_metric}")
    
    # Text-to-Motion retrieval
    t2m_ranks = []
    for i in range(len(dist_matrix)):
        rank = np.where(np.argsort(dist_matrix[i]) == i)[0][0] + 1
        t2m_ranks.append(rank)
    
    # Motion-to-Text retrieval
    m2t_ranks = []
    for i in range(len(dist_matrix)):
        rank = np.where(np.argsort(dist_matrix[:, i]) == i)[0][0] + 1
        m2t_ranks.append(rank)
    
    t2m_ranks = np.array(t2m_ranks)
    m2t_ranks = np.array(m2t_ranks)
    
    metrics = {
        "t2m_R1": float(np.mean(t2m_ranks <= 1) * 100),
        "t2m_R2": float(np.mean(t2m_ranks <= 2) * 100),
        "t2m_R3": float(np.mean(t2m_ranks <= 3) * 100),
        "t2m_R5": float(np.mean(t2m_ranks <= 5) * 100),
        "t2m_R10": float(np.mean(t2m_ranks <= 10) * 100),
        "t2m_MedR": float(np.median(t2m_ranks)),
        "m2t_R1": float(np.mean(m2t_ranks <= 1) * 100),
        "m2t_R2": float(np.mean(m2t_ranks <= 2) * 100),
        "m2t_R3": float(np.mean(m2t_ranks <= 3) * 100),
        "m2t_R5": float(np.mean(m2t_ranks <= 5) * 100),
        "m2t_R10": float(np.mean(m2t_ranks <= 10) * 100),
        "m2t_MedR": float(np.median(m2t_ranks)),
    }
    
    matching_score = float(dist_matrix.diagonal().mean())
    metrics["matching_score"] = matching_score
    
    return metrics


def generate_motions_from_dataset(model, dataset, num_samples=None, batch_size=32, device="cuda"):
    """
    Generate motions from text prompts in the dataset.
    
    Returns:
        generated_motions: List of generated motion arrays
        texts: List of corresponding text prompts
    """
    keyids = sorted(dataset.keyids)
    
    if num_samples is not None:
        keyids = keyids[:num_samples]
    
    generated_motions = []
    texts = []
    
    logger.info(f"Generating {len(keyids)} motions...")
    
    for i in tqdm(range(0, len(keyids), batch_size), desc="Generating motions"):
        batch_keyids = keyids[i:i + batch_size]
        batch_data = [dataset.load_keyid(keyid) for keyid in batch_keyids]
        
        # Extract texts
        batch_texts = [data["text"] if isinstance(data["text"], str) else data["text"][0] for data in batch_data]
        batch_lengths = [data["motion_x_dict"]["length"] for data in batch_data]

        # Generate motions from text
        with torch.no_grad():
            # The model should have a generate() method that takes texts and returns motions
            batch_generated = model.generate(batch_texts, batch_lengths)
        
        # Convert to numpy and crop if needed
        for i, motion in enumerate(batch_generated):
            if isinstance(motion, torch.Tensor):
                generated_motions.append(motion.cpu().numpy()[:batch_lengths[i]])
            else:
                generated_motions.append(motion[:batch_lengths[i]])
        
        texts.extend(batch_texts)
    
    return generated_motions, texts


def evaluate_generated_motions(generated_motions, texts, real_motions, evaluator, device="cuda", batch_size=32, retrieval_batch_size=-1, distance_metric='euclidean', normalize_fid=True):
    """
    Evaluate generated motions using an evaluator.
    
    Args:
        generated_motions: List of generated motion arrays
        texts: List of text prompts
        real_motions: List of real motion arrays (ground truth)
        evaluator: Evaluator model
        device: Device to use
        batch_size: Batch size for encoding
        retrieval_batch_size: Batch size for retrieval calculation (-1 for entire dataset)
    
    Returns:
        Dictionary of evaluation metrics
    """
    logger.info("Encoding generated motions...")
    
    # Encode generated motions and texts in batches
    all_gen_motion_embs = []
    all_text_embs = []
    
    for i in tqdm(range(0, len(generated_motions), batch_size), desc="Encoding"):
        batch_gen_motions = generated_motions[i:i + batch_size]
        batch_texts = texts[i:i + batch_size]
        
        # Pad motions
        motion_lengths = [len(m) for m in batch_gen_motions]
        
        lengths_tensor = torch.tensor(motion_lengths, dtype=torch.long).to(device)
        batch_gen_motions = [torch.tensor(motion, dtype=torch.float32).to(device) for motion in batch_gen_motions]

        with torch.no_grad():
            text_emb, motion_emb = evaluator.encode(batch_texts, batch_gen_motions, lengths_tensor)
        
        all_text_embs.append(text_emb.cpu())
        all_gen_motion_embs.append(motion_emb.cpu())
    
    text_embs = torch.cat(all_text_embs, dim=0)
    gen_motion_embs = torch.cat(all_gen_motion_embs, dim=0)
    
    # Encode real motions
    logger.info("Encoding real motions...")
    all_real_motion_embs = []
    
    for i in tqdm(range(0, len(real_motions), batch_size), desc="Encoding real"):
        batch_real_motions = real_motions[i:i + batch_size]
        
        motion_lengths = [len(m) for m in batch_real_motions]
        
        lengths_tensor = torch.tensor(motion_lengths, dtype=torch.long).to(device)
        batch_real_motions = [torch.tensor(motion, dtype=torch.float32).to(device) for motion in batch_real_motions]

        with torch.no_grad():
            _, motion_emb = evaluator.encode(texts[i:i + batch_size], batch_real_motions, lengths_tensor)
        
        all_real_motion_embs.append(motion_emb.cpu())
    
    real_motion_embs = torch.cat(all_real_motion_embs, dim=0)
    
    # Calculate metrics
    results = {}
    
    # 1. Retrieval metrics (text vs generated motion)
    logger.info("Calculating retrieval metrics...")
    if retrieval_batch_size == -1:
        retrieval_metrics = calculate_retrieval_metrics(text_embs, gen_motion_embs, distance_metric=distance_metric)
    else:
        # Batched retrieval
        N = len(text_embs)
        num_batches = (N + retrieval_batch_size - 1) // retrieval_batch_size
        all_metrics = []
        
        for i in range(num_batches):
            start_idx = i * retrieval_batch_size
            end_idx = min((i + 1) * retrieval_batch_size, N)
            batch_metrics = calculate_retrieval_metrics(
                text_embs[start_idx:end_idx],
                gen_motion_embs[start_idx:end_idx],
                distance_metric='euclidean'
            )
            all_metrics.append(batch_metrics)
        
        retrieval_metrics = {k: float(np.mean([m[k] for m in all_metrics])) for k in all_metrics[0].keys()}
    
    results.update(retrieval_metrics)
    
    # 2. FID between generated and real motions
    logger.info("Calculating FID...")
    # gen_mu = gen_motion_embs.mean(dim=0).numpy()
    # gen_sigma = np.cov(gen_motion_embs.numpy(), rowvar=False)
    # real_mu = real_motion_embs.mean(dim=0).numpy()
    # real_sigma = np.cov(real_motion_embs.numpy(), rowvar=False)
    print(" ⚠️  - ! Using NORMALIZED FID")
    gen_mu, gen_cov = calculate_activation_statistics_normalized(
        gen_motion_embs, normalize=normalize_fid
    )
    real_mu, real_cov = calculate_activation_statistics_normalized(
        real_motion_embs, normalize=normalize_fid
    )

    fid = calculate_frechet_distance(real_mu.astype(float),real_cov.astype(float),gen_mu.astype(float),gen_cov.astype(float))
    results["FID"] = float(fid)
    
    # 3. Diversity
    logger.info("Calculating diversity...")
    gen_diversity = calculate_diversity(gen_motion_embs, diversity_times=300)
    real_diversity = calculate_diversity(real_motion_embs, diversity_times=300)
    results["gen_diversity"] = float(gen_diversity)
    results["real_diversity"] = float(real_diversity)
    
    # 4. Multimodality
    logger.info("Calculating multimodality...")
    text_labels = np.arange(len(texts))  # Each text is unique
    multimodality = calculate_multimodality(gen_motion_embs, text_labels, multimodality_times=20)
    results["multimodality"] = float(multimodality)
    
    return results


@hydra.main(config_path="configs", config_name="eval_model", version_base="1.3")
def evaluate(cfg: DictConfig):
    logger.info("Evaluation script for generative models")
    
    set_preload_false(cfg)
    # cfg.data.motion_loader.normalizer = None
    
    import src.prepare  # noqa
    
    # Set device
    device = cfg.get("device", "cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")
    
    # Load model from checkpoint
    checkpoint_dir = Path(cfg.model_checkpoint_dir)
    
    # Load config
    model_cfg = read_config(checkpoint_dir)
    model = instantiate(model_cfg.model)
    
    # Load checkpoint weights — support both standard (logs/checkpoints/) and
    # SPO-style (checkpoints/) directory layouts, plus glob for best-epoch=*.ckpt
    def _resolve_ckpt(base: Path, ckpt_name: str) -> Path:
        candidates = [
            base / "logs/checkpoints" / f"{ckpt_name}.ckpt",
            base / "checkpoints" / f"{ckpt_name}.ckpt",
        ]
        for p in candidates:
            if p.exists():
                return p
        # Handle "best" glob (e.g. best-epoch=3-Validation.tmr=0.657.ckpt)
        if ckpt_name == "best":
            for ckpt_dir in [base / "logs/checkpoints", base / "checkpoints"]:
                matches = sorted(ckpt_dir.glob("best-*.ckpt")) if ckpt_dir.exists() else []
                if matches:
                    # Pick the one with the highest TMR value in the filename
                    def _tmr(p):
                        import re
                        m = re.search(r'tmr=([0-9.]+)', p.name)
                        return float(m.group(1)) if m else 0.0
                    return max(matches, key=_tmr)
        raise FileNotFoundError(
            f"Checkpoint '{ckpt_name}' not found in {base}/logs/checkpoints/ "
            f"or {base}/checkpoints/"
        )

    model_checkpoint_dir = _resolve_ckpt(checkpoint_dir, cfg.model_ckpt)
    logger.info(f"Loading model from checkpoint: {model_checkpoint_dir}")

    if not model_checkpoint_dir.exists():
        raise FileNotFoundError(f"Checkpoint not found: {model_checkpoint_dir}")
    
    checkpoint = torch.load(model_checkpoint_dir, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint['state_dict'])
    model.to(device)
    model.eval()
    
    logger.info("Model loaded successfully")
    
    # Load evaluator
    logger.info(f"Loading evaluator from '{cfg.evaluator._target_}'")
    evaluator = instantiate(cfg.evaluator)
    logger.info(f"Loading evaluator from checkpoint: {cfg.evaluator_checkpoint_dir}")
    evaluator_checkpoint_dir = Path(cfg.evaluator_checkpoint_dir)

    evaluator_ckpt_path = evaluator_checkpoint_dir / "logs/checkpoints" / f"{cfg.evaluator_ckpt}.ckpt"
    if not evaluator_ckpt_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {evaluator_ckpt_path}")
    
    evaluator_checkpoint = torch.load(evaluator_ckpt_path, map_location=device, weights_only=False)
    evaluator.load_state_dict(evaluator_checkpoint['state_dict'])    
    if hasattr(evaluator, 'to'):
        evaluator = evaluator.to(device)
    if hasattr(evaluator, 'eval'):
        evaluator.eval()

    logger.info("Evaluator loaded successfully")
    
    # Load dataset
    logger.info("Loading dataset")
    dataset = instantiate(cfg.data, split=cfg.split, shuffle=True)
    logger.info(f"Dataset size: {len(dataset.keyids)}")
    
    # Generate motions
    generated_motions, texts = generate_motions_from_dataset(
        model,
        dataset,
        num_samples=cfg.get("num_samples", None),
        batch_size=cfg.batch_size,
        device=device
    )
    
    # Get real motions
    logger.info("Loading real motions...")
    keyids = sorted(dataset.keyids)[:len(generated_motions)]
    real_motions = []
    for keyid in keyids:
        data = dataset.load_keyid(keyid)
        motion = data["motion_x_dict"]["x"]
        if isinstance(motion, torch.Tensor):
            motion = motion.cpu().numpy()
        real_motions.append(motion)
    
    # Evaluate
    results = evaluate_generated_motions(
        generated_motions,
        texts,
        real_motions,
        evaluator,
        device=device,
        batch_size=cfg.batch_size,
        retrieval_batch_size=cfg.get("retrieval_batch_size", -1),
        distance_metric=cfg.get("distance_metric", 'euclidean'),
        normalize_fid=cfg.get("normalize_fid", True)
    )
    
    # Print results
    logger.info("\n" + "="*80)
    logger.info("EVALUATION RESULTS")
    logger.info("="*80)
    
    logger.info("\nRetrieval Metrics (Generated):")
    logger.info(f"  T2M R@1:  {results['t2m_R1']:.2f}%")
    logger.info(f"  T2M R@2:  {results['t2m_R2']:.2f}%")
    logger.info(f"  T2M R@3:  {results['t2m_R3']:.2f}%")
    logger.info(f"  T2M R@5:  {results['t2m_R5']:.2f}%")
    logger.info(f"  T2M R@10: {results['t2m_R10']:.2f}%")
    logger.info(f"  T2M MedR: {results['t2m_MedR']:.2f}")
    logger.info(f"  M2T R@1:  {results['m2t_R1']:.2f}%")
    logger.info(f"  M2T R@2:  {results['m2t_R2']:.2f}%")
    logger.info(f"  M2T R@3:  {results['m2t_R3']:.2f}%")
    logger.info(f"  M2T R@5:  {results['m2t_R5']:.2f}%")
    logger.info(f"  M2T R@10: {results['m2t_R10']:.2f}%")
    logger.info(f"  M2T MedR: {results['m2t_MedR']:.2f}")
    
    logger.info("\nQuality Metrics:")
    logger.info(f"  Matching Score: {results['matching_score']:.4f}")
    logger.info(f"  FID:           {results['FID']:.4f}")
    
    logger.info("\nDiversity Metrics:")
    logger.info(f"  Generated Diversity: {results['gen_diversity']:.4f}")
    logger.info(f"  Real Diversity:      {results['real_diversity']:.4f}")
    logger.info(f"  Multimodality:       {results['multimodality']:.4f}")
    
    logger.info("="*80)
    
    # Save results
    output_dir = Path(cfg.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    results_file = output_dir / "results.json"
    results_meta = {
        "results": results,
        "config": {
            "checkpoint_dir": str(checkpoint_dir),
            "num_samples": len(generated_motions),
            "evaluator": cfg.evaluator._target_,
            "dataset": cfg.data.dataset_name,
        }
    }
    
    with open(results_file, "w") as f:
        json.dump(results_meta, f, indent=2)
    
    logger.info(f"\nResults saved to: {results_file}")
    
    return results


if __name__ == "__main__":
    evaluate()