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

from src.utils import set_preload_false

logger = logging.getLogger(__name__)


def calculate_frechet_distance(mu1, sigma1, mu2, sigma2, eps=1e-6):
    """
    Calculate Frechet Distance between two multivariate Gaussians.
    """
    mu1 = np.atleast_1d(mu1)
    mu2 = np.atleast_1d(mu2)
    sigma1 = np.atleast_2d(sigma1)
    sigma2 = np.atleast_2d(sigma2)

    diff = mu1 - mu2

    # Product might be almost singular
    covmean, _ = linalg.sqrtm(sigma1.dot(sigma2), disp=False)
    if not np.isfinite(covmean).all():
        offset = np.eye(sigma1.shape[0]) * eps
        covmean = linalg.sqrtm((sigma1 + offset).dot(sigma2 + offset))

    # Numerical error might give slight imaginary component
    if np.iscomplexobj(covmean):
        if not np.allclose(np.diagonal(covmean).imag, 0, atol=1e-3):
            m = np.max(np.abs(covmean.imag))
            raise ValueError(f'Imaginary component {m}')
        covmean = covmean.real

    tr_covmean = np.trace(covmean)

    return diff.dot(diff) + np.trace(sigma1) + np.trace(sigma2) - 2 * tr_covmean


def calculate_diversity(embeddings, diversity_times=300):
    """
    Calculate diversity of embeddings.
    """
    embeddings = embeddings.cpu().numpy() if isinstance(embeddings, torch.Tensor) else embeddings
    
    num_samples = embeddings.shape[0]
    
    if num_samples < diversity_times:
        diversity_times = num_samples
    
    first_indices = np.random.choice(num_samples, diversity_times, replace=False)
    second_indices = np.random.choice(num_samples, diversity_times, replace=False)
    
    dist = linalg.norm(embeddings[first_indices] - embeddings[second_indices], axis=1)
    
    return dist.mean()


def calculate_multimodality(motion_embeddings, text_labels, multimodality_times=20):
    """
    Calculate multimodality: diversity of motions for the same text.
    
    Args:
        motion_embeddings: (N, D) motion embeddings
        text_labels: (N,) indices indicating which text each motion corresponds to
        multimodality_times: number of sample pairs per text
    """
    motion_embeddings = motion_embeddings.cpu().numpy() if isinstance(motion_embeddings, torch.Tensor) else motion_embeddings
    
    unique_texts = np.unique(text_labels)
    multimodality_scores = []
    
    for text_idx in unique_texts:
        # Get all motions for this text
        motion_mask = text_labels == text_idx
        text_motions = motion_embeddings[motion_mask]
        
        if len(text_motions) < 2:
            continue
        
        # Sample pairs
        num_samples = min(multimodality_times, len(text_motions))
        first_indices = np.random.choice(len(text_motions), num_samples, replace=True)
        second_indices = np.random.choice(len(text_motions), num_samples, replace=True)
        
        dist = linalg.norm(text_motions[first_indices] - text_motions[second_indices], axis=1)
        multimodality_scores.append(dist.mean())
    
    return np.mean(multimodality_scores) if multimodality_scores else 0.0


def calculate_retrieval_metrics(text_embeddings, motion_embeddings, distance_metric='euclidean'):
    """
    Calculate retrieval metrics: R@1, R@2, R@3, R@5, R@10, MedR for both T2M and M2T.
    
    Args:
        text_embeddings: (N, D) tensor
        motion_embeddings: (N, D) tensor
        distance_metric: 'euclidean' or 'cosine'
    """
    text_embeddings = text_embeddings.cpu().numpy() if isinstance(text_embeddings, torch.Tensor) else text_embeddings
    motion_embeddings = motion_embeddings.cpu().numpy() if isinstance(motion_embeddings, torch.Tensor) else motion_embeddings
    
    # Calculate distance matrix
    if distance_metric == 'euclidean':
        dist_matrix = cdist(text_embeddings, motion_embeddings, metric='euclidean')
    elif distance_metric == 'cosine':
        # Normalize embeddings
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
    
    # Average matching score (diagonal of distance matrix)
    matching_score = float(dist_matrix.diagonal().mean())
    metrics["matching_score"] = matching_score
    
    return metrics


def extract_embeddings_batch(evaluator, dataset, batch_size=32, max_samples=None, device="cuda"):
    """
    Extract embeddings from dataset using the evaluator.
    
    Returns:
        text_embeddings: (N, D) tensor
        motion_embeddings: (N, D) tensor
        text_labels: (N,) array - indices to track which motions share the same text
    """
    keyids = sorted(dataset.keyids)
    
    if max_samples is not None:
        keyids = keyids[:max_samples]
    
    all_text_embs = []
    all_motion_embs = []
    all_text_labels = []
    
    # Process in batches
    for i in tqdm(range(0, len(keyids), batch_size), desc="Extracting embeddings"):
        batch_keyids = keyids[i:i + batch_size]
        batch_data = [dataset.load_keyid(keyid) for keyid in batch_keyids]
        
        # Extract motions and texts
        motions = [data["motion_x_dict"]["x"] for data in batch_data]
        texts = [data["text"] if isinstance(data["text"], str) else data["text"][0] for data in batch_data]
        
        # Convert to numpy if needed
        motions_np = []
        for motion in motions:
            if isinstance(motion, torch.Tensor):
                motions_np.append(motion.cpu().numpy())
            else:
                motions_np.append(motion)
        
        # Get motion lengths
        motion_lengths = [len(m) for m in motions_np]
        max_len = max(motion_lengths)
        
        # Pad motions
        padded_motions = []
        for motion in motions_np:
            if len(motion) < max_len:
                padding = np.zeros((max_len - len(motion), motion.shape[1]))
                padded_motion = np.concatenate([motion, padding], axis=0)
            else:
                padded_motion = motion
            padded_motions.append(padded_motion)
        
        motions_tensor = torch.tensor(np.array(padded_motions), dtype=torch.float32).to(device)
        lengths_tensor = torch.tensor(motion_lengths, dtype=torch.long).to(device)
        
        # Encode using unified interface
        with torch.no_grad():
            text_emb, motion_emb = evaluator.encode(texts, motions_tensor, lengths_tensor)
        
        all_text_embs.append(text_emb.cpu())
        all_motion_embs.append(motion_emb.cpu())
        
        # Track text labels (for multimodality)
        all_text_labels.extend([i] * len(batch_keyids))
    
    text_embeddings = torch.cat(all_text_embs, dim=0)
    motion_embeddings = torch.cat(all_motion_embs, dim=0)
    text_labels = np.array(all_text_labels)
    
    return text_embeddings, motion_embeddings, text_labels


def calculate_retrieval_metrics_batched(text_embeddings, motion_embeddings, batch_size=32, distance_metric='euclidean'):
    """
    Calculate retrieval metrics on batches and average the results.
    
    Args:
        text_embeddings: (N, D) tensor
        motion_embeddings: (N, D) tensor
        batch_size: Size of each batch for retrieval calculation
        distance_metric: 'euclidean' or 'cosine'
        
    Returns:
        Average metrics across all batches
    """
    N = len(text_embeddings)
    num_batches = (N + batch_size - 1) // batch_size  # Ceiling division
    
    all_metrics = []
    
    for i in range(num_batches):
        start_idx = i * batch_size
        end_idx = min((i + 1) * batch_size, N)
        
        batch_text_emb = text_embeddings[start_idx:end_idx]
        batch_motion_emb = motion_embeddings[start_idx:end_idx]
        
        # Calculate metrics for this batch
        batch_metrics = calculate_retrieval_metrics(batch_text_emb, batch_motion_emb, distance_metric)
        all_metrics.append(batch_metrics)
    
    # Average all metrics
    avg_metrics = {}
    for key in all_metrics[0].keys():
        avg_metrics[key] = float(np.mean([m[key] for m in all_metrics]))
    
    return avg_metrics


@hydra.main(config_path="configs", config_name="eval_evaluator", version_base="1.3")
def evaluate(cfg: DictConfig):
    logger.info("Evaluation script for motion-text evaluators")

    set_preload_false(cfg)
    
    import src.prepare  # noqa
    
    # Set device
    device = cfg.get("device", "cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")
    
    # Load evaluator
    logger.info(f"Loading evaluator from '{cfg.evaluator._target_}'")
    evaluator = instantiate(cfg.evaluator)
    
    if hasattr(evaluator, 'to'):
        evaluator = evaluator.to(device)
    if hasattr(evaluator, 'eval'):
        evaluator.eval()
    
    # Load dataset
    logger.info("Loading dataset")
    dataset = instantiate(cfg.data, split=cfg.split, shuffle=False)
    logger.info(f"Dataset size: {len(dataset.keyids)}")
    
    # Extract embeddings
    logger.info("Extracting embeddings...")
    text_embs, motion_embs, text_labels = extract_embeddings_batch(
        evaluator,
        dataset,
        batch_size=cfg.batch_size,
        max_samples=cfg.get("max_samples", None),
        device=device
    )
    
    logger.info(f"Extracted {len(text_embs)} text embeddings and {len(motion_embs)} motion embeddings")
    
    # Calculate metrics
    results = {}
    
    # 1. Retrieval metrics
    logger.info("Calculating retrieval metrics...")
    distance_metric = cfg.get("distance_metric", "euclidean")
    print(f"⚠️  -  Using distance metric: {distance_metric}")
    retrieval_batch_size = cfg.get("retrieval_batch_size", -1)
    
    if retrieval_batch_size == -1:
        # Calculate on entire dataset
        logger.info("Computing retrieval metrics on entire dataset")
        retrieval_metrics = calculate_retrieval_metrics(text_embs, motion_embs, distance_metric)
    else:
        # Calculate on batches and average
        logger.info(f"Computing retrieval metrics on batches of size {retrieval_batch_size}")
        retrieval_metrics = calculate_retrieval_metrics_batched(
            text_embs, motion_embs, retrieval_batch_size, distance_metric
        )
    
    results.update(retrieval_metrics)

    print("TODO RIFARE TUTTA QUESTA PARTE COME IN EVAL MODEL")
    
    # 2. FID (Frechet Inception Distance)
    logger.info("Calculating FID...")
    text_mu = text_embs.mean(dim=0).numpy()
    text_sigma = np.cov(text_embs.numpy(), rowvar=False)
    motion_mu = motion_embs.mean(dim=0).numpy()
    motion_sigma = np.cov(motion_embs.numpy(), rowvar=False)
    
    print("⚠️  -  FID SBAGLIATA: tra motion e text")
    fid = calculate_frechet_distance(text_mu, text_sigma, motion_mu, motion_sigma)
    results["FID"] = float(fid)
    
    # 3. Diversity
    logger.info("Calculating diversity...")
    text_diversity = calculate_diversity(text_embs, diversity_times=cfg.get("diversity_times", 300))
    motion_diversity = calculate_diversity(motion_embs, diversity_times=cfg.get("diversity_times", 300))
    results["text_diversity"] = float(text_diversity)
    results["motion_diversity"] = float(motion_diversity)
    
    # 4. Multimodality
    logger.info("Calculating multimodality...")
    multimodality = calculate_multimodality(
        motion_embs,
        text_labels,
        multimodality_times=cfg.get("multimodality_times", 20)
    )
    results["multimodality"] = float(multimodality)
    
    # Print results
    logger.info("\n" + "="*80)
    logger.info("EVALUATION RESULTS")
    logger.info("="*80)
    
    if retrieval_batch_size != -1:
        logger.info(f"\n[Retrieval computed on batches of {retrieval_batch_size}]")
    
    logger.info("\nRetrieval Metrics:")
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
    
    logger.info("\nMatching & Quality Metrics:")
    logger.info(f"  Matching Score: {results['matching_score']:.4f}")
    logger.info(f"  FID:           {results['FID']:.4f}")
    
    logger.info("\nDiversity Metrics:")
    logger.info(f"  Text Diversity:   {results['text_diversity']:.4f}")
    logger.info(f"  Motion Diversity: {results['motion_diversity']:.4f}")
    logger.info(f"  Multimodality:    {results['multimodality']:.4f}")
    
    logger.info("="*80)
    
    # Save results
    output_dir = Path(cfg.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    results_file = output_dir / "results.json"
    
    # Save also the configuration used
    results_meta = {
        "results": results,
        "config": {
            "retrieval_batch_size": retrieval_batch_size,
            "distance_metric": distance_metric,
            "num_samples": len(text_embs),
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