import sys
import os

import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "ppo"))
from envs.walk_env import CLEARANCE_W, UPRIGHT_W


def _step(envs, agent, buffer, device, obs, totals):
    with torch.no_grad():
        actions, logprobs, _, values = agent.get_action_and_value(obs)
    next_obs, rewards, next_terms, next_truncs, infos = envs.step(actions.cpu().numpy())
    n = len(rewards)
    totals["reward"] += float(rewards.sum())
    totals["forward"] += float(infos.get("reward_forward", np.zeros(n)).sum())
    totals["clearance"] += float(infos.get("reward_clearance", np.zeros(n)).sum()) * CLEARANCE_W
    totals["upright"] += float(infos.get("reward_upright", np.zeros(n)).sum()) * UPRIGHT_W
    t = lambda arr: torch.tensor(arr, dtype=torch.float32, device=device)
    buffer.store(obs, actions, t(rewards), values.reshape(-1), t(next_terms), t(next_truncs), logprobs)
    return (torch.tensor(np.array(next_obs, dtype=np.float32), device=device),
            t(next_terms), t(next_truncs), n, int((next_terms | next_truncs).sum()))


def collect(envs, agent, buffer, device, obs, terms, truncs):
    totals = dict(reward=0.0, forward=0.0, clearance=0.0, upright=0.0)
    n_steps, n_done = 0, 0
    for _ in range(buffer.capacity):
        obs, terms, truncs, n, done = _step(envs, agent, buffer, device, obs, totals)
        n_steps += n
        n_done += done
    stats = {k: v / n_steps for k, v in totals.items()}
    stats["ep_len"] = (n_steps / n_done) if n_done > 0 else n_steps
    return obs, terms, truncs, stats
