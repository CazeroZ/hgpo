

import numpy as np
import torch
from collections import defaultdict, Counter, OrderedDict
from verl import DataProto
import uuid
import math
from functools import reduce

# ---------------------------------------------------------- #
# --------------- General Functions of HGPO --------------- #
# ---------------------------------------------------------- #
def to_hashable(x):
    if isinstance(x, (int, float, str, bool)):
        return x
    elif isinstance(x, (np.integer, np.floating)):
        return x.item()
    elif isinstance(x, np.ndarray):
        return tuple(x.flatten())
    elif isinstance(x, (list, tuple)):
        return tuple(to_hashable(e) for e in x)
    elif isinstance(x, dict):
        return tuple(sorted((k, to_hashable(v)) for k, v in x.items()))
    else:
        raise TypeError(f"Unsupported type: {type(x)}")

def summarize_group_size(group_size: list):
    """
    Summarize the dynamics of step-level group.
    Args:
        group_size : List[int]
    """
    counts = Counter(group_size)
    total = sum(counts.values())
    max_size = max(counts)

    summary = {}
    for size in range(1, max_size + 1):
        cnt = counts.get(size, 0)
        prop = cnt / total if total > 0 else 0
        summary[size] = (cnt, prop)

    print("Summary of step-level group sizes:")
    print("Size | Count | Proportion")
    print("-------------------------")
    for size, (cnt, prop) in summary.items():
        if prop:
            print(f"{size:>4} | {cnt:>5} | {prop:>9.2%}")


def compute_step_discounted_returns(batch: DataProto, gamma: float):
    breakpoint()
    print("compute_step_discounted_returns")
    rewards = batch.non_tensor_batch['rewards'].astype(np.float32)
    traj_uids = batch.non_tensor_batch['traj_uid']
    active_masks = batch.non_tensor_batch['active_masks'].astype(np.float32)
    returns_by_traj = {}
    unique_traj_uids = np.unique(traj_uids)
    for uid in unique_traj_uids:
        # Get indices for this trajectory
        traj_indices = np.where(traj_uids == uid)[0]
        
        # Extract rewards and masks for this trajectory
        traj_rewards = rewards[traj_indices]
        traj_active_masks = active_masks[traj_indices]
        assert traj_active_masks.all(), "active_masks should be all 1s for the same trajectory"
        
        # Calculate returns
        traj_returns = np.zeros_like(traj_rewards)
        running_return = 0
        
        # Calculate returns from the end to the start
        for t in reversed(range(len(traj_rewards))):
            running_return = traj_rewards[t] + gamma * running_return
            traj_returns[t] = running_return
        
        # Store the results
        returns_by_traj[uid] = traj_returns
    
    # Recombine the returns into the original batch order
    all_returns = np.zeros_like(rewards)
    for i, uid in enumerate(traj_uids):
        traj_indices = np.where(traj_uids == uid)[0]
        idx_in_traj = np.where(traj_indices == i)[0][0]  # Find position of i in its trajectory
        all_returns[i] = returns_by_traj[uid][idx_in_traj]
    
    all_returns = torch.tensor(all_returns, dtype=torch.float32, device=batch.batch['input_ids'].device)
    return all_returns


def compute_hgpo_outcome_advantage(
                                   step_rewards: torch.Tensor,
                                   response_mask: torch.Tensor,
                                   anchor_obs: np.array,
                                   index: np.array,
                                   traj_index: np.array,
                                   types: str = "hard",
                                   history_length: int = 2,
                                   epsilon: float = 1e-6,
                                    weight_type: str = 'mean',
                                    mode: str = "mean_norm"
                                   ):

    step_advantages = HGPO(anchor_obs, index, traj_index, step_rewards, history_length, epsilon, response_mask=response_mask, weight_type=weight_type, mode=mode)
    return step_advantages, step_advantages


def merge_single_value_clusters(clusters, new_key='merged'):
    new_dict = {}
    merged_values = []

    for k, v in clusters.items():
        if len(v) == 1:
            merged_values.extend(v)
        else:
            new_dict[k] = v

    if merged_values:
        new_dict[new_key] = merged_values

    return new_dict


def HGPO(obs: np.array, group_ids: np.array, traj_ids: np.array, step_rewards: torch.Tensor, history_length: int, epsilon=1e-6, response_mask=None, weight_type='mean', mode=''):
    print("new_find_common_state_subsequences2")
    #breakpoint()
    response_length = response_mask.shape[-1]
    # Initialize the result array with placeholder values
    all_step_advantages = np.zeros(len(obs), dtype=float)
    
    # Get unique indices
    unique_group_ids = np.unique(group_ids)

    # Process each unique index
    for group_i,gid in enumerate(unique_group_ids):
        # Get all observations for this index using np.where
        group_indices = np.where(group_ids == gid)[0]
        group_obs = obs[group_indices] 
        group_step_rewards = step_rewards[group_indices]
        group_traj_ids = traj_ids[group_indices]
        unique_group_traj_ids = np.unique(group_traj_ids)

        group_traj_obs = []
        group_traj_idx = []
        group_traj_step_rewards = []
        for traj_id in unique_group_traj_ids:
            traj_idx = np.where(group_traj_ids==traj_id)[0]
            traj_step_rewards = group_step_rewards[traj_idx]
            group_traj_step_rewards.append(traj_step_rewards)
            group_traj_obs.append(group_obs[traj_idx])
            group_traj_idx.append(traj_idx.tolist())

        group_clusters = defaultdict(list)
        for i,traj_obs in enumerate(group_traj_obs):
            for j,step_obs in enumerate(traj_obs):
                group_clusters[to_hashable(step_obs)].append((i,j)) 

        group_clusters = merge_single_value_clusters(group_clusters, new_key='alone')

        for step_i, (step_obs, indices) in enumerate(group_clusters.items()):
            if step_obs == 'alone':

                    traj_idx, step_idx = idx
                    all_step_advantages[group_indices[group_traj_idx[traj_idx][step_idx]]] = 0
            else:
                
                group_history_clusters = defaultdict(list)
                for l in reversed(range(0,history_length+1,1)):
                    for idx in indices:
                        traj_idx, step_idx = idx
                        if step_idx < l:
                            continue
                        history_obs = group_traj_obs[traj_idx][(step_idx - l):step_idx]
                        group_history_clusters[tuple(history_obs)].append((traj_idx, step_idx))
                    group_history_clusters = defaultdict(list, {k: v for k, v in group_history_clusters.items() if len(v) > 1})  


                multi_advantages = defaultdict(list)
                for key, history_indices in group_history_clusters.items():
                    rewards = []
                    for idx in history_indices:
                        traj_idx, step_idx = idx
                        rewards.append(group_traj_step_rewards[traj_idx][step_idx])

                    mean_reward = np.mean(rewards)
                    std_reward = np.std(rewards)
                    if mode=='mean_std_norm':
                        scores = (rewards - mean_reward) / (std_reward + epsilon)
                    elif mode=='mean_norm':
                        scores = rewards - mean_reward
                    for i, idx in enumerate(history_indices):
                        multi_advantages[idx].append((len(key)+1, scores[i]))

                print_multi_advantages = defaultdict(list)
                alpha = 1
                for idx, all_advantages in multi_advantages.items():
                    weights, advantages = [], []
                    for length, advantage in all_advantages:
                        if advantage != 0:
                            advantages.append(advantage)
                            weights.append(length ** alpha)
                    norm_weights = np.array(weights) / sum(weights)
                    final_advantages = sum(np.array(advantages) * norm_weights)
                    traj_idx, step_idx = idx
                    all_step_advantages[group_indices[group_traj_idx[traj_idx][step_idx]]] = final_advantages
                    print_multi_advantages[idx].append(final_advantages)

    scores = torch.tensor(all_step_advantages, dtype=torch.float32)
    cnt = torch.where(scores == 0)[0].shape[0]
    print(f"step-level zero advantage: {cnt}/ {scores.shape[0]}")
    scores = scores.unsqueeze(-1).tile([1, response_length]) * response_mask
    breakpoint()
    return scores

def step_norm_reward(step_rewards: torch.Tensor,
                      response_mask: torch.Tensor,
                      index: np.array,
                      epsilon: float = 1e-6,
                      remove_std: bool = False,
                      ):
    """
    Compute step-level advantage using mean-std normalization for GiGPO.
    Args:
        step_rewards: `(torch.Tensor)`
            shape: (bs,)
        response_mask: `(torch.Tensor)`
            shape: (bs, response_length)
    
    Returns:
        advantages: `(torch.Tensor)`
            shape: (bs, response_length)
        Returns: `(torch.Tensor)`
            shape: (bs, response_length)
    """
    print("step_norm_reward")
    response_length = response_mask.shape[-1]
    scores = step_rewards.clone()

    id2score = defaultdict(list)
    id2mean = {}
    id2std = {}
    mask = torch.ones_like(scores, dtype=torch.int64)
    
    with torch.no_grad():
        bsz = scores.shape[0]
        for i in range(bsz):
            id2score[index[i]].append(scores[i])

        for idx in id2score:
            if len(id2score[idx]) == 1:
                id2mean[idx] = torch.mean(torch.tensor(id2score[idx]))
                id2std[idx] = torch.tensor(1.0)
            elif len(id2score[idx]) > 1:
                id2mean[idx] = torch.mean(torch.tensor(id2score[idx]))
                id2std[idx] = torch.std(torch.tensor([id2score[idx]]))
            else:
                print(f"id2score: {id2score}")
                print(f"len(id2score[idx]): {len(id2score[idx])}")
                raise ValueError(f"no score in prompt index: {idx}")
        for i in range(bsz):
            if remove_std:
                scores[i] = scores[i] - id2mean[index[i]]
            else:
                scores[i] = (scores[i] - id2mean[index[i]]) / (id2std[index[i]] + epsilon)
        step_advantages = scores.unsqueeze(-1).tile([1, response_length]) * response_mask
    
    return step_advantages

def episode_norm_reward(token_level_rewards: torch.Tensor,
                        response_mask: torch.Tensor,
                        index: np.array,
                        traj_index: np.array,
                        epsilon: float = 1e-6,
                        remove_std: bool = True,
                        compute_mean_std_cross_all_data: bool = True,
                        ):
    """
    Compute episode-level advantage using mean-std normalization for GiGPO.
    (with only one scalar reward for each episode).
    Args:
        token_level_rewards: `(torch.Tensor)`
            shape: (bs, response_length)
        response_mask: `(torch.Tensor)`
            shape: (bs, response_length)
        index: `(np.array)`
            shape: (bs,)
        traj_index: `(np.array)`
            shape: (bs,)
        epsilon: float
            A small value to avoid division by zero.
        remove_std: bool
            If True, the standard deviation is removed from the normalization.
        compute_mean_std_cross_all_data: bool
            If True (more stable), the mean and std are computed across all data in the batch. 
            If False (i.e., standard episode-level adv), the mean and std are computed across N trajectories.
    
    Returns:
        advantages: `(torch.Tensor)`
            shape: (bs, response_length)
        Returns: `(torch.Tensor)`
            shape: (bs, response_length)
    """
    print("episode_norm_reward")
    response_length = token_level_rewards.shape[-1]
    scores = token_level_rewards.sum(dim=-1)

    id2score = defaultdict(list)
    id2mean = {}
    id2std = {}
    seen_pairs = set()
    with torch.no_grad():
        bsz = scores.shape[0]
        for i in range(bsz):
            if (index[i], traj_index[i]) in seen_pairs:
                continue
            id2score[index[i]].append(scores[i])
            if not compute_mean_std_cross_all_data:
                seen_pairs.add((index[i], traj_index[i]))

        for idx in id2score:
            if len(id2score[idx]) == 1:
                id2mean[idx] = torch.tensor(0.0)
                id2std[idx] = torch.tensor(1.0)
            elif len(id2score[idx]) > 1:
                id2mean[idx] = torch.mean(torch.tensor(id2score[idx]))
                id2std[idx] = torch.std(torch.tensor([id2score[idx]]))
            else:
                raise ValueError(f"no score in prompt index: {idx}")
        for i in range(bsz):
            if remove_std:
                scores[i] = scores[i] - id2mean[index[i]]
            else:
                scores[i] = (scores[i] - id2mean[index[i]]) / (id2std[index[i]] + epsilon)
        episode_advantages = scores.unsqueeze(-1).tile([1, response_length]) * response_mask
    
    return episode_advantages

