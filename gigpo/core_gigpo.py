"""
Core functions to implement GiGPO algorithms.
The function implemented in this file should be used by trainer with different distributed strategies to implement GiGPO
"""

import numpy as np
import torch
from collections import defaultdict, Counter, OrderedDict
from verl import DataProto
import uuid
import math
from functools import reduce

# ---------------------------------------------------------- #
# --------------- General Functions of GiGPO --------------- #
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

def lcs_all(sequences):
    def lcs_two(seq1, seq2):
        m, n = len(seq1), len(seq2)
        dp = [[[] for _ in range(n + 1)] for _ in range(m + 1)]
        for i in range(m):
            for j in range(n):
                if seq1[i] == seq2[j]:
                    dp[i + 1][j + 1] = dp[i][j] + [seq1[i]]
                else:
                    dp[i + 1][j + 1] = max(dp[i][j + 1], dp[i + 1][j], key=len)
        return dp[m][n]
    return reduce(lcs_two, sequences)

def constrained_lcs_with_fixed_start_end(sequences):
    # 所有轨迹共享起点和终点
    common_start = sequences[0][0]
    common_end = sequences[0][-1]
    trimmed = [seq[1:-1] for seq in sequences]
    middle_lcs = lcs_all(trimmed)
    return [common_start] + middle_lcs + [common_end]

def all_substrings(seq):
    n = len(seq)
    return [tuple(seq[i:j]) for i in range(n) for j in range(i+1, n+1)]

def unordered_longest_common_subtrajectory(trajectories):
    # 使用集合交集，找出所有轨迹中都出现的状态
    common_states = reduce(lambda a, b: set(a) & set(b), trajectories)
    return list(common_states)

def new_compute_step_discounted_returns(batch: DataProto, gamma: float, old=False):
    if old:
        print("new_compute_step_discounted_return, originalGIGPO")
    else:
        print("new_compute_step_discounted_return, newReward")
    rewards = batch.non_tensor_batch['rewards'].astype(np.float32)
    traj_uids = batch.non_tensor_batch['traj_uid']
    group_uids = batch.non_tensor_batch['uid']
    obs = batch.non_tensor_batch['anchor_obs']
    active_masks = batch.non_tensor_batch['active_masks'].astype(np.float32)
    returns_by_traj = {}
    unique_group_uids = np.unique(group_uids)

    common_success_seq_size = []
    common_failed_seq_size = []
    success_group_size = []
    failed_group_size = []
    for g_uid in unique_group_uids:

        group_indices = np.where(group_uids == g_uid)[0]
        group_obs = obs[group_indices]
        group_step_rewards = rewards[group_indices]
        group_traj_uids = traj_uids[group_indices]

        success_group_traj_obs = []
        success_group_traj_indeces = []
        failed_group_traj_obs = []
        failed_group_traj_indeces = []

        for tid in np.unique(group_traj_uids):
            group_traj_indices = np.where(group_traj_uids == tid)[0]
            group_traj_step_rewards = group_step_rewards[group_traj_indices]
            group_traj_obs = group_obs[group_traj_indices]
            if group_traj_step_rewards.sum(axis=-1) == 0:
                failed_group_traj_obs.append(group_traj_obs)
                failed_group_traj_indeces.append(group_traj_indices.tolist())
            else:
                success_group_traj_obs.append(group_traj_obs)
                success_group_traj_indeces.append(group_traj_indices.tolist())
                
        success_group_size.append(len(success_group_traj_obs))
        failed_group_size.append(len(failed_group_traj_obs))

        if len(success_group_traj_obs)!=0:
            common_success_seq = unordered_longest_common_subtrajectory(success_group_traj_obs)
        else:
            common_success_seq = []
        common_success_seq_size.append(len(common_success_seq))

        if len(failed_group_traj_obs)!=0:
            common_failed_seq = unordered_longest_common_subtrajectory(failed_group_traj_obs)
        else:
            common_failed_seq = []
        common_failed_seq_size.append(len(common_failed_seq))

        intersection = list(set(common_success_seq).intersection(set(common_failed_seq)))

        cnt_s1 = 0
        cnt_f1 = 0
        for s in common_success_seq:
            for i in success_group_traj_obs:
                if s in i:
                    cnt_s1 += 1
            for j in failed_group_traj_obs:
                if s in j:
                    cnt_f1 += 1

        cnt_s2 = 0
        cnt_f2 = 0
        for s in common_failed_seq:
            for i in success_group_traj_obs:
                if s in i:
                    cnt_s2 += 1
            for j in failed_group_traj_obs:
                if s in j:
                    cnt_f2 += 1

        print(f"common success size:{len(common_success_seq)}")
        print(f"common failed size:{len(common_failed_seq)}")
        print(f"common intersection size:{len(intersection)}")

        print(f"cnt common_success_seq:{cnt_s1}/{cnt_f1}/{len(success_group_traj_obs)}")
        print(f"cnt common_failed_seq:{cnt_s2}/{cnt_f2}/{len(failed_group_traj_obs)}")

        reward_success_step_size = []
        reward_failed_step_size = []
        for tid in np.unique(group_traj_uids):
            # Get indices for this trajectory
            group_traj_indices = np.where(group_traj_uids == tid)[0]

            # Extract rewards and masks for this trajectory
            group_traj_step_rewards = group_step_rewards[group_traj_indices]
            
            group_traj_obs = group_obs[group_traj_indices]

            # Calculate returns
            traj_returns = np.zeros_like(group_traj_step_rewards)
            running_return = 0

            if group_traj_step_rewards.sum()==0:
                is_failed = True
            else:
                is_failed = False
                
            # all trajectory are failed
            if len(common_success_seq) != 0:
                avg_common_success_seq_reward = 10 / len(common_success_seq)
                # Calculate returns from the end to the start
                if not old:
                    if is_failed:
                        group_traj_step_rewards[-1] = 10
                for t in reversed(range(len(group_traj_step_rewards))):
                        running_return = group_traj_step_rewards[t] + gamma * running_return
                        if old:#gigpo
                            traj_returns[t] = running_return
                        else:#
                            # if is_failed:
                            #     if group_traj_obs[t] in common_success_seq:
                            #         traj_returns[t] = running_return 
                            # else:
                            #     traj_returns[t] = running_return    
                            if group_traj_obs[t] in common_success_seq:
                                    traj_returns[t] = running_return + avg_common_success_seq_reward
                
                if is_failed:
                    traj_returns[-1] = 0

                cnt = len(np.where(traj_returns != 0)[0])
                if is_failed:
                    reward_failed_step_size.append(cnt)
                else:
                    reward_success_step_size.append(cnt)

            # Store the results
            returns_by_traj[tid] = traj_returns
        avg_success_step = sum(len(i) for i in success_group_traj_obs)/len(success_group_traj_obs) if len(success_group_traj_obs) > 0 else 0
        avg_failed_step = sum(len(i) for i in failed_group_traj_obs)/len(failed_group_traj_obs) if len(failed_group_traj_obs) > 0 else 0
        avg_reward_success_step_size = np.mean(reward_success_step_size) if len(reward_success_step_size) > 0 else 0
        avg_reward_failed_step_size = np.mean(reward_failed_step_size) if len(reward_failed_step_size) > 0 else 0
        print(f"Group {g_uid} - reward Success step size: {avg_reward_success_step_size}/{avg_success_step}, reward Failed step size: {avg_reward_failed_step_size}/{avg_failed_step}")
    print(f"common success seq size: {common_success_seq_size}")
    print(f"common failed seq size: {common_failed_seq_size}")
    print(f"group success traj size: {success_group_size}")
    print(f"group failed traj size: {failed_group_size}")
    

    # Recombine the returns into the original batch order
    all_returns = np.zeros_like(rewards)
    for i, uid in enumerate(traj_uids):
        traj_indices = np.where(traj_uids == uid)[0]
        idx_in_traj = np.where(traj_indices == i)[0][0]  # Find position of i in its trajectory
        all_returns[i] = returns_by_traj[uid][idx_in_traj]


    all_returns = torch.tensor(all_returns, dtype=torch.float32, device=batch.batch['input_ids'].device)
    return all_returns

def compute_step_discounted_returns(batch: DataProto, gamma: float):
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

# ---------------------------------------------------------- #
# ---------------- Core Functions of GiGPO ----------------- #
# ---------------------------------------------------------- #

def compute_gigpo_outcome_advantage(token_level_rewards: torch.Tensor,
                                   step_rewards: torch.Tensor,
                                   response_mask: torch.Tensor,
                                   anchor_obs: np.array,
                                   index: np.array,
                                   traj_index: np.array,
                                   epsilon: float = 1e-6,
                                   step_advantage_w: float = 1.0,
                                   mode: str = "mean_norm"
                                   ):
    
    if mode == "mean_std_norm":
        remove_std = False
    elif mode == "mean_norm":
        remove_std = True
    else:
        raise ValueError(f"Unknown mode: {mode}")

    # Compute episode-level group reward

    breakpoint()
    episode_advantages= episode_norm_reward(token_level_rewards, response_mask, index, traj_index, epsilon, remove_std)
    
    # Compute step_group_uids
    #step_group_uids = build_step_group(anchor_obs, index)
    #step_group_uids = build_step_group2(anchor_obs, index)
    
    #step_group_uids = find_common_state_subsequences(anchor_obs, index, traj_index, step_rewards)
    #step_group_uids = new_find_common_state_subsequences(anchor_obs, index, traj_index, step_rewards, history_length=5)
    #step_advantages = new_find_common_state_subsequences2(anchor_obs, index, traj_index, step_rewards, history_length=3, response_mask=response_mask)

    # Compute step-level group reward
    #step_advantages= step_norm_reward(step_rewards, response_mask, step_group_uids, epsilon, remove_std)

    #scores = episode_advantages + step_advantage_w * step_advantages
    scores = episode_advantages
    return scores, scores


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
        cnt = torch.where(scores == 0)[0].shape[0]
        print(f"episode-level zero advantage: {cnt}/ {bsz}")
        episode_advantages = scores.unsqueeze(-1).tile([1, response_length]) * response_mask
    
    return episode_advantages

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

def build_step_group2(anchor_obs: np.array, index: np.array, summarize: bool = False):
    """
    Group observations by index and then cluster identical observations within each index group.
    Assigns a unique step_group_uid (UUID) to each cluster.
    
    Parameters:
    -----------
    anchor_obs : np.array
        Array of observation strings
    index : np.array
        Array of corresponding indices for each observation
    summarize : bool
        Whether to summarize the group sizes (default: True)
    
    Returns:
    --------
    np.array
        Array of step_group_uid values corresponding to the original anchor_obs array
    """
    print("build_step_group2")
    # Initialize the result array with placeholder values
    step_group_uids = np.empty(len(anchor_obs), dtype=object)
    
    # Get unique indices
    unique_indices = np.unique(index)

    group_size = []
    # Process each unique index
    for idx in unique_indices:
        # Get all observations for this index using np.where
        indices = np.where(index == idx)[0]
        obs_group = anchor_obs[indices]
        
        # Create clusters for identical observations
        clusters = defaultdict(list)
        for i, obs in enumerate(obs_group):
            clusters[to_hashable(obs)].append(indices[i])  # Store the original index position
        
        breakpoint()
        clusters = merge_single_value_clusters(clusters)

        # Assign unique step_group_uid to each cluster
        for obs, original_indices  in clusters.items():
            # Generate a UUID for this cluster
            uid = str(uuid.uuid4())

            # Assign the same step_group_uid to all elements in this cluster
            group_size.append(len(original_indices))
            for original_idx in original_indices:
                step_group_uids[original_idx] = uid

        # Validate that all elements have been assigned a uid
    if None in step_group_uids or np.any(step_group_uids == None):
        missing_indices = np.where(step_group_uids == None)[0]
        raise ValueError(f"Failed to assign UIDs to all observations. Missing at indices: {missing_indices}")

    if summarize:
        summarize_group_size(group_size)
    print(f"Avg size of step-level group: {np.mean(group_size)}")
    print(f"merged alone steps: {len(clusters['merged'])}")
    return step_group_uids

def build_step_group(anchor_obs: np.array, index: np.array, summarize: bool = False):
    """
    Group observations by index and then cluster identical observations within each index group.
    Assigns a unique step_group_uid (UUID) to each cluster.
    
    Parameters:
    -----------
    anchor_obs : np.array
        Array of observation strings
    index : np.array
        Array of corresponding indices for each observation
    summarize : bool
        Whether to summarize the group sizes (default: True)
    
    Returns:
    --------
    np.array
        Array of step_group_uid values corresponding to the original anchor_obs array
    """
    print("build_step_group")
    # Initialize the result array with placeholder values
    step_group_uids = np.empty(len(anchor_obs), dtype=object)
    
    # Get unique indices
    unique_indices = np.unique(index)

    group_size = []
    # Process each unique index
    for idx in unique_indices:
        # Get all observations for this index using np.where
        indices = np.where(index == idx)[0]
        obs_group = anchor_obs[indices]
        
        # Create clusters for identical observations
        clusters = defaultdict(list)
        for i, obs in enumerate(obs_group):
            clusters[to_hashable(obs)].append(indices[i])  # Store the original index position
        
        # Assign unique step_group_uid to each cluster
        for obs, original_indices in clusters.items():
            # Generate a UUID for this cluster
            uid = str(uuid.uuid4())
            
            # Assign the same step_group_uid to all elements in this cluster
            group_size.append(len(original_indices))
            for original_idx in original_indices:
                step_group_uids[original_idx] = uid

        # Validate that all elements have been assigned a uid
    if None in step_group_uids or np.any(step_group_uids == None):
        missing_indices = np.where(step_group_uids == None)[0]
        raise ValueError(f"Failed to assign UIDs to all observations. Missing at indices: {missing_indices}")

    if summarize:
        summarize_group_size(group_size)
    print(f"Avg size of step-level group: {np.mean(group_size)}")
    return step_group_uids


def all_substrings(seq):
    n = len(seq)
    return [tuple(seq[i:j]) for i in range(n) for j in range(i+1, n+1)]

def find_longest_common_subtrajectory(trajs, min_frac: float = 0.8):
    # """
    # 从多个轨迹中找出最长公共连续子串a
    # """
    # commons = all_substrings(trajs[0])
    # for traj in trajs[1:]:
    #     temp_commons = all_substrings(traj)
    #     commons = commons & temp_commons
    #     if not commons:
    #         return []
    # #return list(max(commons, key=len)) if commons else []
    # return list(commons)

    substring_counter = defaultdict(list)  # substring -> list(trajectory indices)
    for i, traj in enumerate(trajs):
        all_subs = all_substrings(traj)
        for sub in all_subs:
            substring_counter[sub].append(i)
    min_count = math.floor(len(trajs) * min_frac)

    candidates = [s for s, idx_list in substring_counter.items() if len(idx_list) > 1 and len(s) < 5]
    if not candidates:
        return []
    return sorted(candidates, key=lambda x: len(x), reverse=True)

def find_common_state_subsequences(obs: np.array, group_ids: np.array, traj_ids: np.array, step_rewards: torch.Tensor, summarize: bool = False):
    print("find_common_state_subsequences")

    # Initialize the result array with placeholder values
    step_group_uids = np.empty(len(obs), dtype=object)
    
    # Get unique indices
    unique_group_ids = np.unique(group_ids)

    group_size = []
    len_lcs_stats=np.zeros(50, dtype=int)
    # Process each unique index
    for gid in unique_group_ids:
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

        all_lcs = find_longest_common_subtrajectory(group_traj_obs)

        group_size1 = []
        group_size2 = []
        for l in all_lcs:
            uid = str(uuid.uuid4())
            cnt = 0
            for traj, indices in zip(group_traj_obs, group_traj_idx):
                for i in range(len(traj) - len(l) + 1):
                    if list(traj[i:i+len(l)]) == list(l):
                        # only assign uid to the last occurrence of l in each trajectory
                        last_obs_index = group_indices[indices[i+len(l)-1]]
                        if step_group_uids[last_obs_index] == None:
                            step_group_uids[last_obs_index] = uid
                            cnt += 1
            if cnt > 0:
                if len(l) ==1 :
                    group_size1.append(cnt)
                elif len(l) ==2 :
                    group_size2.append(cnt)
        
        # alone steps
        uid = str(uuid.uuid4())
        alone_cnt=0
        for idx in group_indices:
            if step_group_uids[idx] == None:
                step_group_uids[idx] = uid
                alone_cnt += 1
                
        print(f"group1/group2/alone group/all group steps: {group_size1}/{group_size2}/{alone_cnt}/{len(group_indices)}")
        assert sum(group_size1)+sum(group_size2)+alone_cnt == len(group_indices)
    breakpoint()
    print(f"All: considered nums of steps/all steps: {np.where(step_group_uids != None)[0].shape[0]}/{len(step_group_uids)}")
    return step_group_uids 

def count_history_obs(trajs):
    substring_counter = defaultdict(list)  # substring -> list(trajectory indices)
    for i, traj in enumerate(trajs):
        all_subs = all_substrings(traj)
        for sub in all_subs:
            substring_counter[sub].append(i)
    candidates = [s for s, idx_list in substring_counter.items()]
    if not candidates:
        return []
    return sorted(candidates, key=lambda x: len(x), reverse=True)

def get_history_obs(trajs, history_length: int = 5):
    if idx[1] < l:
        current_history_length = idx[1] if idx[1] > 0 else 0
    else:   
        current_history_length = l
    history_obs = tuple(group_traj_obs[idx[0]][(idx[1] - current_history_length):idx[1]])

def new_find_common_state_subsequences(obs: np.array, group_ids: np.array, traj_ids: np.array, step_rewards: torch.Tensor, history_length: int, summarize: bool = False):
    print("new_find_common_state_subsequences")

    # Initialize the result array with placeholder values
    step_group_uids = np.empty(len(obs), dtype=object)
    
    # Get unique indices
    unique_group_ids = np.unique(group_ids)

    # Process each unique index
    for gid in unique_group_ids:
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

        step_stats = defaultdict(list)
        for step_obs, indices in group_clusters.items():
            if step_obs == 'alone':
                uid = str(uuid.uuid4())
                cnt = 0
                for traj_idx, step_idx in indices:
                    uuid_index = group_indices[group_traj_idx[traj_idx][step_idx]]
                    step_group_uids[uuid_index] = uid
                    cnt += 1
                    #print(f"Group {gid} - step obs: {step_obs} Alone step obs size: {cnt}")
                step_stats['alone'].append(cnt)
            else:
                group_history_clusters = defaultdict(list)
                # greedy merge history
                use_flag = np.zeros(len(indices), dtype=bool)
                for l in reversed(range(1,history_length+1,1)):
                    for j, idx in enumerate(indices):
                        if idx[1] !=0 and idx[1] < l:
                            continue
                        #     current_history_length = idx[1] if idx[1] > 0 else 0
                        # else:   
                        #     current_history_length = l
                        history_obs = group_traj_obs[idx[0]][(idx[1] - l):idx[1]]
                        if not use_flag[j]:
                            group_history_clusters[tuple(history_obs)].append(j)
                            use_flag[j] = True
                    for k,v in group_history_clusters.items():
                        if len(v) < 2:
                            use_flag[v] = False
                    group_history_clusters = defaultdict(list, {k: v for k, v in group_history_clusters.items() if len(v) > 1})  # filter out clusters with only one observation

                #context-inconsistent history 
                for i in range(len(use_flag)):
                    if not use_flag[i]:
                        group_history_clusters["context-inconsistent"].append(i)

                for key, history_indices in group_history_clusters.items():
                    uid = str(uuid.uuid4())
                    cnt = 0
                    for j in history_indices:
                        traj_idx, step_idx = indices[j]
                        uuid_index = group_indices[group_traj_idx[traj_idx][step_idx]]
                        step_group_uids[uuid_index] = uid
                        cnt += 1
                    if key == "context-inconsistent":
                        step_stats[key].append(cnt)
                    else:
                        step_stats[f'context-{len(key)}'].append(cnt)

    print(f"Group {gid} - All steps: {len(group_indices)}")
    print(step_stats)
    assert np.where(step_group_uids[group_indices] == None)[0].shape[0] == 0

    return step_group_uids 

def new_find_common_state_subsequences2(obs: np.array, group_ids: np.array, traj_ids: np.array, step_rewards: torch.Tensor, history_length: int, response_mask):
    print("new_find_common_state_subsequences")
    breakpoint()
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
                # uid = str(uuid.uuid4())
                # for traj_idx, step_idx in indices:
                #     uuid_index = group_indices[group_traj_idx[traj_idx][step_idx]]
                continue
            else:
                if step_obs=="obs5":
                    debug=0
                advantages = np.zeros(len(indices), dtype=float)
                group_history_clusters = defaultdict(list)
                for l in reversed(range(0,history_length+1,1)):
                    for j, idx in enumerate(indices):
                        if idx[1] < l:
                            continue
                        history_obs = group_traj_obs[idx[0]][(idx[1] - l):idx[1]]
                        group_history_clusters[tuple(history_obs)].append(j)
                    group_history_clusters = defaultdict(list, {k: v for k, v in group_history_clusters.items() if len(v) > 1})  # filter out clusters with only one observation
                print(f"Group {group_i} - step obs: {step_i} History clusters: {[(len(k),len(v)) for k,v in group_history_clusters.items()]}")    
                cnt_group = np.zeros(len(indices), dtype=int)
                for key, history_indices in group_history_clusters.items():
                    cnt_group[history_indices] += 1
                    rewards = []
                    for j in history_indices:
                        traj_idx, step_idx = indices[j]
                        rewards.append(group_traj_step_rewards[traj_idx][step_idx])
                    mean_reward = np.mean(rewards)
                    std_reward = np.std(rewards) + 1e-6
                    scores = (rewards - mean_reward) / (std_reward + 1e-6)
                    if len(key) == 0:
                        print(f"Group {group_i} - step obs: {step_i} original advantages: {scores}")
                    #advantages[history_indices] += ((len(key)+1)/(history_length+1)) * scores
                    advantages[history_indices] += scores
                advantages /= (cnt_group + 1e-6)
                for i,idx in enumerate(indices):
                    traj_idx, step_idx = idx
                    all_step_advantages[group_indices[group_traj_idx[traj_idx][step_idx]]] = advantages[i]
            print(f"Group {group_i} - step obs: {step_i} new Advantages: {advantages}")

    scores = torch.tensor(all_step_advantages, dtype=torch.float32, device=response_mask.device)
    cnt = torch.where(scores == 0)[0].shape[0]
    print(f"step-level zero advantage: {cnt}/ {scores.shape[0]}")
    scores = scores.unsqueeze(-1).tile([1, response_length]) * response_mask

    return scores

def step_norm_reward(step_rewards: torch.Tensor,
                      response_mask: torch.Tensor,
                      index: np.array,
                      epsilon: float = 1e-6,
                      remove_std: bool = True,
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
        # ## mask
        # for i in range(bsz):
        #     if len(id2score[index[i]]) < 10:
        #         mask[i] = 0
        # ###
        for i in range(bsz):
            if remove_std:
                scores[i] = scores[i] - id2mean[index[i]]
            else:
                scores[i] = (scores[i] - id2mean[index[i]]) / (id2std[index[i]] + epsilon)
        # scores *= mask
        cnt = torch.where(scores == 0)[0].shape[0]
        print(f"step-level zero advantage: {cnt}/ {bsz}")
        step_advantages = scores.unsqueeze(-1).tile([1, response_length]) * response_mask
    
    return step_advantages


if __name__ == "__main__":
    # Example usage
    #obs = np.array(["obs1", "obs2", "obs3", "obs4", "obs5", "obs1", "obs2", "obs3", "obs4", "obs5", "obs1", "obs2", "obs3", "obs4", "obs5", 'obs6'])
    obs = np.array(["obs1", "obs2", "obs3", "obs4", "obs5", 
                    "obs1", "obs3", "obs2", "obs4", "obs5", 
                    "obs1", "obs5", "obs2", "obs4", "obs5", 
                    "obs1", "obs2", "obs3", "obs4", "obs5", 'obs6'
                    ])
    print("obs:", obs)
    index = np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0])
    traj_ids = np.array([0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 3, 3, 3, 3, 3, 3,])
    step_rewards = np.array([1.15, 2.7, 1.5, 2.55, 2.0, 1.24, 2.0, 1.5, 2.6, 2.0, 1.3, 2.2, 1.5, 2.5, 2.3, 1.0, 2.0, 1.5, 2.59, 2.8, 1.0])
    history_length = 3
    step_group_uids = new_find_common_state_subsequences2(obs, index, traj_ids, step_rewards, history_length)
    print("Step Group UIDs:", step_group_uids)
    
