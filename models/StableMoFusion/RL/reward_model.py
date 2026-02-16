import numpy as np
import torch
from colorama import Fore, Style, init
from torch import cosine_similarity

from TMR.src.model.tmr import get_sim_matrix
from TMR.mtt.load_tmr_model import load_tmr_model_easy, easy_forward, load_tmr_model_complete
from TM2T.load_tm2t_model import load_tm2t_model_easy
from src.tools.guofeats.motion_representation import guofeats_to_joints
# from MotionCritic.MotionCritic.lib.model.load_critic import load_critic
# from MotionCritic.MotionCritic.parsedata import into_critic


class RewardModels():
    # classe che contiene i reward models
    # si inizializza con None e poi una volta caricata la configurazione in main.py si istanziano i modelli relativi al dataset preso in considerazione
    def __init__(self):
        self.tmr_forward_plus_plus_complete = None
        self.tmr_forward_complete = None
        self.guo_forward = None

    def init_models(self, dataset="humanml3d"):
        # dataset: "humanml3d" OR "kit"
        
        # todo tenere il modello di reward su gpu (se serve solo mentre lo calcolo)
        self.guo_forward = load_tm2t_model_easy(device="cpu", dataset='humanml3d') # humanml3d OR humanml3d_kitml_augmented_and_hn OR tmr_humanml3d_kitml_guoh3dfeats
        
        self.tmr_forward_complete = load_tmr_model_complete(device="cpu", dataset=dataset)

        self.tmr_forward_plus_plus_complete = load_tmr_model_complete(device="cpu", dataset="tmr_humanml3d_kitml_guoh3dfeats") # TODO BHO!


# todo tenere il modello di reward su gpu (se serve solo mentre lo calcolo)
# tmr_forward_plus_plus_complete = load_tmr_model_complete(device="cpu", dataset="tmr_humanml3d_kitml_guoh3dfeats")
# tmr_forward_complete = load_tmr_model_complete(device="cpu", dataset="humanml3d")
# guo_forward = load_tm2t_model_easy(device="cpu", dataset="humanml3d") # humanml3d OR humanml3d_kitml_augmented_and_hn OR tmr_humanml3d_kitml_guoh3dfeats

# critic_model = load_critic("./MotionCritic/MotionCritic/pretrained/motioncritic_pre.pth", "cpu")

reward_models = RewardModels()


def calc_eval_stats(x_guofeats, forward):
    x_latents = forward(x_guofeats)# tensor(N, 256)
    return x_latents


def is_list_of_strings(var):
    return isinstance(var, list) and all(isinstance(item, str) for item in var)


def print_matrix_nicely(matrix: np.ndarray, mmax=True):
    init(autoreset=True)
    for row in matrix:
        if mmax:
            max_val = np.max(row)
        else:
            max_val = np.min(row)
        line = ""
        for val in row:
            truncated = int(val * 1000) / 1000
            formatted = f"{truncated:.3f}"
            if val == max_val:
                line += f"{Fore.GREEN}{formatted}{Style.RESET_ALL}  "
            else:
                line += f"{formatted}  "
        print(line)

#
# def tmr_metrics(motions_guofeats,real_texts, c):
#     texts = tmr_forward(real_texts)
#     x_latents = calc_eval_stats(motions_guofeats, tmr_forward)
#     sim_matrix = get_sim_matrix(x_latents, texts.detach().cpu().type(x_latents.dtype)).numpy()
#     sim_matrix = torch.tensor(sim_matrix)
#     sim_matrix = (sim_matrix + 1) / 2
#     tmr = sim_matrix.diagonal()
#
#     reward = tmr * c.reward_scale
#
#     return tmr, reward
#
# def tmr_plus_plus_metrics(motions_guofeats,real_texts, c):
#     texts_plus_plus = tmr_forward_plus_plus(real_texts)
#     x_latents_plus_plus = calc_eval_stats(motions_guofeats, tmr_forward_plus_plus)
#
#     sim_matrix_plus_plus = get_sim_matrix(x_latents_plus_plus,texts_plus_plus.detach().cpu().type(texts_plus_plus.dtype)).numpy()
#
#     sim_matrix_plus_plus = torch.tensor(sim_matrix_plus_plus)
#     sim_matrix_plus_plus = (sim_matrix_plus_plus + 1) / 2
#     tmr_plus_plus = sim_matrix_plus_plus.diagonal()
#
#     reward = tmr_plus_plus * c.reward_scale
#
#     return tmr_plus_plus, reward

def metric_fast(model, motions_guofeats,real_texts, c, device="cuda:0"):
    texts_plus_plus = easy_forward(*model, motions_or_texts=real_texts, device=device)
    x_latents_plus_plus = easy_forward(*model, motions_or_texts=motions_guofeats, device=device)

    sim_matrix_plus_plus = get_sim_matrix(x_latents_plus_plus,texts_plus_plus.detach().cpu().type(texts_plus_plus.dtype)).numpy()

    sim_matrix_plus_plus = torch.tensor(sim_matrix_plus_plus)
    sim_matrix_plus_plus = (sim_matrix_plus_plus + 1) / 2
    tmr_plus_plus = sim_matrix_plus_plus.diagonal()

    reward = tmr_plus_plus * c.reward_scale

    return tmr_plus_plus, reward

def guo_metrics(motions_guofeats, real_texts, c, normalize=True):

    motion_embeddings, text_embeddings = reward_models.guo_forward(motions=motions_guofeats, texts=real_texts, normalize=normalize)
    dist_mat = euclidean_distance_matrix(motion_embeddings.detach().cpu().numpy(), text_embeddings.detach().cpu().numpy())

    # dist_mat = torch.tensor(dist_mat)
    matching_score_sum = dist_mat.diagonal()
    reward = (1 / (matching_score_sum.sum() + 1)) * c.reward_scale
    # reward = -guo_et_al
    return matching_score_sum, reward

def guo_metrics_eval(motions_guofeats, real_texts, c, normalize=True):

    text_embeddings, motion_embeddings = reward_models.guo_forward(motions=motions_guofeats, texts=real_texts, normalize=normalize)
    dist_mat = euclidean_distance_matrix(text_embeddings.detach().cpu().numpy(), motion_embeddings.detach().cpu().numpy())

    # dist_mat = torch.tensor(dist_mat)
    matching_score_sum = dist_mat.diagonal()
    reward = (1 / (matching_score_sum.sum() + 1)) * c.reward_scale
    # reward = -guo_et_al
    return matching_score_sum, reward, dist_mat, motion_embeddings


def crop_motion_original_size(sequences, infos, smplh=None):
    motions = []
    for idx in range(sequences.shape[0]):
        x_start = sequences[idx]
        length = infos["all_lengths"][idx].item()
        x_start = x_start[:length]
        
        motions.append(x_start.detach().cpu().numpy())

    return motions


def reward_model(sequences, infos, smplh, real_texts, c):
    metrics = {}

    motions = crop_motion_original_size(sequences, infos, smplh)

    if c.reward == "stilness":
        reward = stillness_reward(motions, infos, None)
        metrics = {
                "tmr": reward,
                "reward": reward
            }

    if c.reward == "TMR":
        tmr, reward = metric_fast(reward_models.tmr_forward_complete,motions,real_texts, c)
        metrics = {
            "tmr": tmr,
            "reward": reward
        }


    if c.reward == "TMR++":
        tmr_plus_plus, reward = metric_fast(reward_models.tmr_forward_plus_plus_complete, motions, real_texts, c)
        metrics = {
            "tmr++": tmr_plus_plus,
            "reward" : reward
        }

    if c.reward == "GUO":
        guo_et_al, reward = guo_metrics(motions,real_texts, c, normalize=True) # guo devo normalizzarlo prima di mandarlo in ingresso
        metrics = {
            "guo": guo_et_al,
            "reward": reward
        }


    if c.reward == "motionCritic":

        pass
        # joints = guofeats_to_rot6d(motions_guofeats[0])
        # joints = into_critic(joints)
        # reward = critic_model.module.batch_critic(joints)
        # metrics = {
        #     "reward": reward
        # }

    return metrics

def all_metrics(sequences, infos, smplh, real_texts, c):

    motions = crop_motion_original_size(sequences, infos, smplh)
    
    guo_et_al, _  = guo_metrics(motions, real_texts, c, normalize=True) # guo devo normalizzarlo prima di mandarlo in ingresso
    tmr, _ = metric_fast(reward_models.tmr_forward_complete, motions, real_texts, c)
    tmr_plus_plus, _ = metric_fast(reward_models.tmr_forward_plus_plus_complete, motions,real_texts, c) # TMR credo di non doverlo normalizzare

    metrics = {
        "tmr": tmr,
        "tmr++": tmr_plus_plus,
        "guo": guo_et_al,
    }

    return metrics


def all_metrics_eval(sequences, infos, smplh, real_texts, c):
    motions = crop_motion_original_size(sequences, infos, smplh)

    guo_et_al, _, dist_matrix, motion_embeddings = guo_metrics_eval(motions, real_texts, c,
                               normalize=True)  # guo devo normalizzarlo prima di mandarlo in ingresso
    tmr, _ = metric_fast(reward_models.tmr_forward_complete, motions, real_texts, c)
    tmr_plus_plus, _ = metric_fast(reward_models.tmr_forward_plus_plus_complete, motions, real_texts,
                                   c)  # TMR credo di non doverlo normalizzare

    metrics = {
        "tmr": tmr,
        "tmr++": tmr_plus_plus,
        "guo": guo_et_al,
        "guo_dist_matrix": dist_matrix,
        "guo_motion_embeddings": motion_embeddings,
    }

    return metrics

"""
    TMR SPECIAL

    sim_matrix_tmp = get_sim_matrix(x_latents, all_embedding_tmr.detach().cpu().type(x_latents.dtype)).numpy()
    # print_matrix_nicely(sim_matrix_tmp)

    sim_matrix_tmp = (sim_matrix_tmp + 1) / 2
    diagonal_values = sim_matrix.diagonal()

    # Calculate similarity between texts and all_embedding_tmr and find the most similar embedding in all_embedding_tmr
    text_to_all_sim = torch.matmul(texts.detach().cpu(), all_embedding_tmr.transpose(0, 1))

    matching_indices = torch.argmax(text_to_all_sim, dim=1)

    special = []
    for i in range(sim_matrix_tmp.shape[0]):
        # Get the index to exclude for this row
        exclude_idx = matching_indices[i].item()
        # Make a copy of the row and set the element to exclude to NaN
        row_copy = sim_matrix_tmp[i].copy()
        row_copy[exclude_idx] = np.nan
        row_copy[row_copy > c.masking_ratio] = np.nan

        # Calculate mean without the excluded element
        row_mean = np.nanmean(row_copy)
        # Calculate special value for this row (real - mean of row of all emb)
        special_value = diagonal_values[i] - row_mean
        special.append(special_value)

    special = torch.tensor(special)

return special * c.reward_scale
"""

def euclidean_distance_matrix(matrix1, matrix2):
    """
        Params:
        -- matrix1: N1 x D
        -- matrix2: N2 x D
        Returns:
        -- dist: N1 x N2
        dist[i, j] == distance(matrix1[i], matrix2[j])
    """
    assert matrix1.shape[1] == matrix2.shape[1]
    d1 = -2 * np.dot(matrix1, matrix2.T)    # shape (num_test, num_train)
    d2 = np.sum(np.square(matrix1), axis=1, keepdims=True)    # shape (num_test, 1)
    d3 = np.sum(np.square(matrix2), axis=1)     # shape (num_train, )
    dists = np.sqrt(d1 + d2 + d3)  # broadcasting
    return dists


def collate_tensor_with_padding(batch):
    dims = batch[0].dim()
    max_size = [max([b.size(i) for b in batch]) for i in range(dims)]
    size = (len(batch),) + tuple(max_size)
    canvas = batch[0].new_zeros(size=size)
    for i, b in enumerate(batch):
        sub_tensor = canvas[i]
        for d in range(dims):
            sub_tensor = sub_tensor.narrow(d, 0, b.size(d))
        sub_tensor.add_(b)
    return canvas



def stillness_reward(sequences, infos, smplh):
    joint_positions = []
    for idx in range(len(sequences)):
        x_start = sequences[idx]
        # the mask here should already be done here, right?
        joints = guofeats_to_joints(torch.tensor(x_start))
        joint_positions.append(joints)

    joints = collate_tensor_with_padding(joint_positions)
    dt = 1.0 / 200

    velocities = torch.diff(joints, dim=1) / dt
    velocity_loss = torch.mean(velocities.pow(2), dim=(1, 2, 3))

    reward = velocity_loss
    return - reward



def sim_matrix_retrival(sequences, texts, lengths, smplh, features_extractor="GUO", device="cpu"):

    if features_extractor=="TMR++":
        texts_latents = easy_forward(*reward_models.tmr_forward_plus_plus_complete, motions_or_texts=texts, device=device)
        x_latents = easy_forward(*reward_models.tmr_forward_plus_plus_complete, motions_or_texts=sequences, device=device)
        sim_matrix = get_sim_matrix(x_latents, texts_latents.detach().cpu().type(x_latents.dtype)).numpy()

    elif features_extractor=="GUO":
        motions_latents, texts_latents = reward_models.guo_forward(motions=sequences, texts=texts, normalize=True)
        sim_matrix = euclidean_distance_matrix(motions_latents.detach().cpu().numpy(), texts_latents.detach().cpu().numpy())

    elif features_extractor=="TMR":
        texts_latents = easy_forward(*reward_models.tmr_forward_complete, motions_or_texts=texts, device=device)
        x_latents = easy_forward(*reward_models.tmr_forward_complete, motions_or_texts=sequences, device=device)
        sim_matrix = get_sim_matrix(x_latents, texts_latents.detach().cpu().type(x_latents.dtype)).numpy()

    else:
        raise NotImplementedError

    return sim_matrix



def retrival_score(sim_matrix_m2t, text_embeds=None, retrival_threshold=None, batch_size=32):

    if retrival_threshold is not None:
        text_sim_matrix = cosine_similarity(text_embeds)

    m2t_top_1_lst = []
    m2t_top_3_lst = []
    m2t_top_10_lst = []

    if batch_size == 0:
        batch_size = sim_matrix_m2t.shape[0]

    n_batches = sim_matrix_m2t.shape[0] // batch_size
    # Store the distances
    block_distances = []  # list of 16 distance matrices, each 32 x 32
    for i in range(n_batches):
        start = i * batch_size
        end = start + batch_size

        dist = sim_matrix_m2t[start:end, start:end]
        for idx in range(len(dist)):

            asort = np.argsort(dist[idx])[::-1]
            if retrival_threshold is None:
                m2t_top_1_lst.append(1 * (idx in asort[:1]))
                m2t_top_3_lst.append(1 * (idx in asort[:3]))
                m2t_top_10_lst.append(1 * (idx in asort[:10]))
            else:
                true_matches = np.where(text_sim_matrix[idx] >= retrival_threshold)[0] # Threshold-based text similarity match
                m2t_top_1_lst.append(1 * any(i in true_matches for i in asort[:1]))
                m2t_top_3_lst.append(1 * any(i in true_matches for i in asort[:3]))
                m2t_top_10_lst.append(1 * any(i in true_matches for i in asort[:10]))

    m2t_top_1 = np.mean(m2t_top_1_lst)
    m2t_top_3 = np.mean(m2t_top_3_lst)
    m2t_top_10 = np.mean(m2t_top_10_lst)

    rs = {
        "m2t_top_1": m2t_top_1*100,
        "m2t_top_3": m2t_top_3*100,
        "m2t_top_10": m2t_top_10*100
    }

    return rs

