import numpy as np
import torch
from colorama import Fore, Style, init
from src.tools.guofeats.motion_representation import joints_to_guofeats
from src.tools.extract_joints import extract_joints
from TMR.src.guofeats import joints_to_guofeats
from TMR_SMPL.src.model.tmr import get_sim_matrix
from TMR_SMPL.mtt.load_tmr_model import load_tmr_model_easy
from TM2T.load_tm2t_model import load_tm2t_model_easy

tmr_forward_plus_plus = None
tmr_forward = None
guo_forward = None
guo_forward_kit = None
guo_forward_babel = None
guo_forward_motionx = None


def load_models(c):
    global tmr_forward_plus_plus, tmr_forward, guo_forward, guo_forward_kit, guo_forward_babel, guo_forward_motionx
    
    tmr_forward_plus_plus = load_tmr_model_easy(device="cpu", dataset="humantmrpp")
    tmr_forward = load_tmr_model_easy(device="cuda:0", dataset=c.tmr_name)
    
    guo_forward = load_tm2t_model_easy(device="cpu", dataset="humanml3d")
    guo_forward_kit = load_tm2t_model_easy(device="cpu", dataset="kit_22")
    guo_forward_babel = load_tm2t_model_easy(device="cpu", dataset="babel")
    guo_forward_motionx = load_tm2t_model_easy(device="cpu", dataset="motionx")


def smpl_to_guofeats(smpl, smplh):
    guofeats = []
    for i in smpl:
        i_output = extract_joints(
            i,
            'smplrifke',
            fps=20,
            value_from='smpl',
            smpl_layer=smplh,
        )
        i_joints = i_output["joints"]  # tensor(N, 22, 3)
        # convert to guofeats, first, make sure to revert the axis, as guofeats have gravity axis in Y
        x, y, z = i_joints.T
        i_joints = np.stack((x, z, -y), axis=0).T
        i_guofeats = joints_to_guofeats(i_joints)
        guofeats.append(i_guofeats)

    return guofeats


def calc_eval_stats(x_guofeats, forward):
    x_latents = forward(x_guofeats)# tensor(N, 256)
    return x_latents

def calc_eval_stats_t(x_smplrifke, forward, t):
    x_latents = forward(x_smplrifke, t)# tensor(N, 256)
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


def only_tmr_plus_plus(sequences, infos, smplh, real_texts, all_embedding_tmr, c):
    texts_plus_plus = tmr_forward_plus_plus(real_texts)

    motions = []
    for idx in range(sequences.shape[0]):
        x_start = sequences[idx]
        length = infos["all_lengths"][idx].item()
        x_start = x_start[:length]
        motions.append(x_start.detach().cpu())

    motions_guofeats = smpl_to_guofeats(motions, smplh=smplh)

    x_latents_plus_plus = calc_eval_stats(motions_guofeats, tmr_forward_plus_plus)
    sim_matrix_plus_plus = get_sim_matrix(x_latents_plus_plus, texts_plus_plus.detach().cpu().type(x_latents_plus_plus.dtype)).numpy()

    sim_matrix_plus_plus = torch.tensor(sim_matrix_plus_plus)
    sim_matrix_plus_plus = (sim_matrix_plus_plus + 1) / 2
    tmr_plus_plus = sim_matrix_plus_plus.diagonal()

    return tmr_plus_plus

def tmr_metrics(motions_guofeats,real_texts, c):
    texts = tmr_forward(real_texts)
    x_latents = calc_eval_stats(motions_guofeats, tmr_forward)
    sim_matrix = get_sim_matrix(x_latents, texts.detach().cpu().type(x_latents.dtype)).numpy()
    sim_matrix = torch.tensor(sim_matrix)
    sim_matrix = (sim_matrix + 1) / 2
    tmr = sim_matrix.diagonal()

    reward = tmr * c.reward_scale

    return tmr, reward

def tmr_metrics_fast(motions_guofeats, texts, t):

    x_latents = calc_eval_stats_t(motions_guofeats, tmr_forward, t)
    sim_matrix = get_sim_matrix(x_latents, texts).numpy()
    sim_matrix = torch.tensor(sim_matrix)
    sim_matrix = (sim_matrix + 1) / 2
    tmr = sim_matrix.diagonal()



    return tmr

def embed_text(real_texts):
    texts = tmr_forward(real_texts)
    return texts.detach().cpu()


def tmr_plus_plus_metrics(motions_guofeats,real_texts, c):
    texts_plus_plus = tmr_forward_plus_plus(real_texts)
    x_latents_plus_plus = calc_eval_stats(motions_guofeats, tmr_forward_plus_plus)

    sim_matrix_plus_plus = get_sim_matrix(x_latents_plus_plus,texts_plus_plus.detach().cpu().type(texts_plus_plus.dtype)).numpy()

    sim_matrix_plus_plus = torch.tensor(sim_matrix_plus_plus)
    sim_matrix_plus_plus = (sim_matrix_plus_plus + 1) / 2
    tmr_plus_plus = sim_matrix_plus_plus.diagonal()

    reward = tmr_plus_plus * c.reward_scale

    return tmr_plus_plus, reward

def guo_metrics(motions_guofeats, real_texts, c):
    motions_latents, texts_latents = guo_forward(motions=motions_guofeats, texts=real_texts)
    sim_matrix = euclidean_distance_matrix(motions_latents.cpu().numpy(), texts_latents.cpu().numpy())

    sim_matrix = torch.tensor(sim_matrix)
    guo_et_al = sim_matrix.diagonal()
    #reward = (1 / (guo_et_al + 1)) * c.reward_scale
    reward = -guo_et_al
    return guo_et_al, reward

def guo_kit_metrics(motions_guofeats, real_texts, c):
    motions_latents, texts_latents = guo_forward_kit(motions=motions_guofeats, texts=real_texts)
    sim_matrix = euclidean_distance_matrix(motions_latents.cpu().numpy(), texts_latents.cpu().numpy())

    sim_matrix = torch.tensor(sim_matrix)
    guo_et_al = sim_matrix.diagonal()
    #reward = (1 / (guo_et_al + 1)) * c.reward_scale
    reward = -guo_et_al
    return guo_et_al, reward

def guo_babel_metrics(motions_guofeats, real_texts, c):
    motions_latents, texts_latents = guo_forward_babel(motions=motions_guofeats, texts=real_texts)
    sim_matrix = euclidean_distance_matrix(motions_latents.cpu().numpy(), texts_latents.cpu().numpy())

    sim_matrix = torch.tensor(sim_matrix)
    guo_et_al = sim_matrix.diagonal()
    #reward = (1 / (guo_et_al + 1)) * c.reward_scale
    reward = -guo_et_al
    return guo_et_al, reward

def guo_motionx_metrics(motions_guofeats, real_texts, c):
    motions_latents, texts_latents = guo_forward_motionx(motions=motions_guofeats, texts=real_texts)
    sim_matrix = euclidean_distance_matrix(motions_latents.cpu().numpy(), texts_latents.cpu().numpy())

    sim_matrix = torch.tensor(sim_matrix)
    guo_et_al = sim_matrix.diagonal()
    #reward = (1 / (guo_et_al + 1)) * c.reward_scale
    reward = -guo_et_al
    return guo_et_al, reward

def guo_metrics_eval(motions_guofeats, real_texts, c):
    texts_latents, motions_latents = guo_forward(motions=motions_guofeats, texts=real_texts)
    sim_matrix = euclidean_distance_matrix(texts_latents.cpu().numpy(), motions_latents.cpu().numpy())

    sim_matrix = torch.tensor(sim_matrix)
    guo_et_al = sim_matrix.diagonal()
    #reward = (1 / (guo_et_al + 1)) * c.reward_scale
    reward = -guo_et_al
    return guo_et_al, reward, sim_matrix,motions_latents

def guo_kit_metrics_eval(motions_guofeats, real_texts, c):
    texts_latents, motions_latents = guo_forward_kit(motions=motions_guofeats, texts=real_texts)
    sim_matrix = euclidean_distance_matrix(motions_latents.cpu().numpy(), texts_latents.cpu().numpy())

    sim_matrix = torch.tensor(sim_matrix)
    guo_et_al = sim_matrix.diagonal()
    #reward = (1 / (guo_et_al + 1)) * c.reward_scale
    reward = -guo_et_al
    return guo_et_al, reward, sim_matrix ,motions_latents

def guo_babel_metrics_eval(motions_guofeats, real_texts, c):
    texts_latents, motions_latents = guo_forward_babel(motions=motions_guofeats, texts=real_texts)
    sim_matrix = euclidean_distance_matrix(motions_latents.cpu().numpy(), texts_latents.cpu().numpy())

    sim_matrix = torch.tensor(sim_matrix)
    guo_et_al = sim_matrix.diagonal()
    #reward = (1 / (guo_et_al + 1)) * c.reward_scale
    reward = -guo_et_al
    return guo_et_al, reward, sim_matrix ,motions_latents

def guo_motionx_metrics_eval(motions_guofeats, real_texts, c):
    texts_latents, motions_latents = guo_forward_motionx(motions=motions_guofeats, texts=real_texts)
    sim_matrix = euclidean_distance_matrix(motions_latents.cpu().numpy(), texts_latents.cpu().numpy())

    sim_matrix = torch.tensor(sim_matrix)
    guo_et_al = sim_matrix.diagonal()
    #reward = (1 / (guo_et_al + 1)) * c.reward_scale
    reward = -guo_et_al
    return guo_et_al, reward, sim_matrix ,motions_latents


def get_motion_guofeats(sequences, infos, smplh, guo=True):
    motions = []
    for idx in range(sequences.shape[0]):
        x_start = sequences[idx]
        length = infos["all_lengths"][idx].item()
        x_start = x_start[:length]
        motions.append(x_start.detach().float().cpu())

    if guo:
        motions_guofeats = smpl_to_guofeats(motions, smplh=smplh)
    else:
        motions_guofeats = motions

    return motions_guofeats

def mask(sequences, infos):
    motions = []
    for idx in range(sequences.shape[0]):
        x_start = sequences[idx]
        length = infos["all_lengths"][idx].item()
        x_start = x_start[:length]
        motions.append(x_start)

    return motions


def reward_model(sequences, infos, smplh, real_texts, c):
    metrics = {}

    motions_guofeats = get_motion_guofeats(sequences, infos, smplh)

    if c.reward == "TMR":
        tmr, reward = tmr_metrics(motions_guofeats,real_texts, c)
        metrics = {
            "tmr": tmr,
            "reward": reward
        }

    if c.reward == "TMR++":
        tmr_plus_plus, reward = tmr_plus_plus_metrics(motions_guofeats,real_texts, c)
        metrics = {
            "tmr++": tmr_plus_plus,
            "reward" : reward
        }

    if c.reward == "GUO":
        guo_et_al, reward = guo_metrics(motions_guofeats,real_texts, c)
        metrics = {
            "guo": guo_et_al,
            "reward": reward
        }

    return metrics

def reward_model_smpl(sequences, infos, texts, t):
    masked_sequences = mask(sequences, infos)
    tmr = tmr_metrics_fast(masked_sequences, texts, t)

    return tmr

def guo(sequences, infos, smplh, real_texts, c):
    motions_guofeats = get_motion_guofeats(sequences, infos, smplh)
    guo_et_al, _  = guo_metrics(motions_guofeats, real_texts, c)

    return guo_et_al

def guo_kit(sequences, infos, smplh, real_texts, c):
    motions_guofeats = get_motion_guofeats(sequences, infos, smplh)
    guo_et_al, _  = guo_kit_metrics(motions_guofeats, real_texts, c)

    return guo_et_al

def guo_babel(sequences, infos, smplh, real_texts, c):
    motions_guofeats = get_motion_guofeats(sequences, infos, smplh)
    guo_et_al, _  = guo_babel_metrics(motions_guofeats, real_texts, c)

    return guo_et_al

def guo_motionx(sequences, infos, smplh, real_texts, c):
    motions_guofeats = get_motion_guofeats(sequences, infos, smplh)
    guo_et_al, _  = guo_motionx_metrics(motions_guofeats, real_texts, c)

    return guo_et_al


def guo_eval(sequences, infos, smplh, real_texts, c, guo=True):
    motions_guofeats = get_motion_guofeats(sequences, infos, smplh, guo)
    guo_et_al, _, sim_matrix, motions_latents = guo_metrics_eval(motions_guofeats, real_texts, c)

    return guo_et_al, sim_matrix, motions_latents

def guo_eval_kit(sequences, infos, smplh, real_texts, c, guo=True):
    motions_guofeats = get_motion_guofeats(sequences, infos, smplh, guo)
    guo_et_al, _, sim_matrix, motions_latents = guo_kit_metrics_eval(motions_guofeats, real_texts, c)

    return guo_et_al, sim_matrix, motions_latents

def guo_eval_babel(sequences, infos, smplh, real_texts, c, guo=True):
    motions_guofeats = get_motion_guofeats(sequences, infos, smplh, guo)
    guo_et_al, _, sim_matrix, motions_latents = guo_babel_metrics_eval(motions_guofeats, real_texts, c)

    return guo_et_al, sim_matrix, motions_latents

def guo_eval_motionx(sequences, infos, smplh, real_texts, c, guo=True):
    motions_guofeats = get_motion_guofeats(sequences, infos, smplh, guo)
    guo_et_al, _, sim_matrix, motions_latents = guo_motionx_metrics_eval(motions_guofeats, real_texts, c)

    return guo_et_al, sim_matrix, motions_latents


def all_metrics(sequences, infos, smplh, real_texts, c):

    motions_guofeats = get_motion_guofeats(sequences, infos, smplh)

    guo_et_al, guo_reward  = guo_metrics(motions_guofeats, real_texts, c)
    tmr_plus_plus, tmr_plus_plus_reward = tmr_plus_plus_metrics(motions_guofeats, real_texts, c)
    tmr, tmr_reward = tmr_metrics(motions_guofeats, real_texts, c)

    reward = None

    if c.reward == "TMR":
        reward = tmr_reward
    if c.reward == "TMR++":
        reward = tmr_plus_plus_reward
    if c.reward == "GUO":
        reward = guo_reward

    metrics = {
        "tmr": tmr,
        "tmr++": tmr_plus_plus,
        "guo": guo_et_al,
        "reward": reward,
    }

    return metrics


def all_metrics(sequences, infos, smplh, real_texts, c):

    motions_guofeats = get_motion_guofeats(sequences, infos, smplh)

    guo_et_al, guo_reward  = guo_metrics(motions_guofeats, real_texts, c)
    tmr_plus_plus, tmr_plus_plus_reward = tmr_plus_plus_metrics(motions_guofeats, real_texts, c)
    tmr, tmr_reward = tmr_metrics(motions_guofeats, real_texts, c)

    reward = None

    if c.reward == "TMR":
        reward = tmr_reward
    if c.reward == "TMR++":
        reward = tmr_plus_plus_reward
    if c.reward == "GUO":
        reward = guo_reward

    metrics = {
        "tmr": tmr,
        "tmr++": tmr_plus_plus,
        "guo": guo_et_al,
        "reward": reward,
    }

    return metrics

def all_metrics_eval(sequences, infos, smplh, real_texts, c):

    motions_guofeats = get_motion_guofeats(sequences, infos, smplh)

    guo_et_al, guo_reward , sim_matrix,motions_latents = guo_metrics_eval(motions_guofeats, real_texts, c)
    tmr_plus_plus, tmr_plus_plus_reward =  torch.zeros_like(guo_et_al), 0 # tmr_plus_plus_metrics(motions_guofeats, real_texts, c)
    tmr, tmr_reward = torch.zeros_like(guo_et_al), 0 # tmr_metrics(motions_guofeats, real_texts, c)

    reward = None

    if c.reward == "TMR":
        reward = tmr_reward
    if c.reward == "TMR++":
        reward = tmr_plus_plus_reward
    if c.reward == "GUO":
        reward = guo_reward

    metrics = {
        "tmr": tmr,
        "tmr++": tmr_plus_plus,
        "guo": guo_et_al,
        "reward": reward,
        "guo_dist_matrix": sim_matrix,
        "guo_motion_embeddings": motions_latents,
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


def guo_reward(sequences, infos, smplh, real_texts, all_embedding_tmr, c):

    motions = []
    for idx in range(sequences.shape[0]):
        x_start = sequences[idx]
        length = infos["all_lengths"][idx].item()
        x_start = x_start[:length]
        motions.append(x_start.detach().cpu())

    motions_guofeats = smpl_to_guofeats(motions, smplh=smplh)
    motions_latents, texts_latents = guo_forward(motions=motions_guofeats, texts=real_texts)

    sim_matrix = euclidean_distance_matrix(motions_latents.cpu().numpy(), texts_latents.cpu().numpy())
    # print_matrix_nicely(sim_matrix, mmax=False)

    sim_matrix = torch.tensor(sim_matrix)
    # Normalization (not needed but I wanted):
    # the Trace of the matrix is between [0, inf], so I multiply it by (-1) and add 1, Now is between [-inf, 1]. I divide by 10 for better visualization.
    guo = (sim_matrix.diagonal() * (-1) + 1) / 10 

    metrics = {
        "guo": guo,
        "reward": guo * c.reward_scale
    }

    return metrics


def stillness_reward(sequences, infos, smplh):
    joint_positions = []
    for idx in range(sequences.shape[0]):
        x_start = sequences[idx]
        length = infos["all_lengths"][idx].item()
        x_start = x_start[:length]

        output = extract_joints(
            x_start.detach().cpu(),
            'smplrifke',
            fps=20,
            value_from='smpl',
            smpl_layer=smplh,
        )

        joints = torch.as_tensor(output["joints"])
        joint_positions.append(joints)

    joints = torch.stack(joint_positions)
    dt = 1.0 / 200

    velocities = torch.diff(joints, dim=1) / dt
    velocity_loss = torch.mean(velocities.pow(2), dim=(1, 2, 3))

    reward = velocity_loss
    return - reward
