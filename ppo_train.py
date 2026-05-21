import csv
import os
import sys

import numpy as np
import torch
import wandb

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "ppo"))
from ppo_update import ppo_update
from eval_utils import run_episode, frames_to_video

_CSV_PATH = "log/metrics.csv"
_CSV_FIELDS = ["epoch", "reward/total", "reward/forward", "reward/clearance", "reward/upright", "reward/arm",
               "train/ep_len", "ppo/policy_loss", "ppo/value_loss", "ppo/entropy", "ppo/kl", "ppo/clip_frac"]


def init_csv():
    os.makedirs(os.path.dirname(_CSV_PATH), exist_ok=True)
    with open(_CSV_PATH, "w", newline="") as f:
        csv.DictWriter(f, fieldnames=_CSV_FIELDS).writeheader()


def _log_csv(row: dict):
    with open(_CSV_PATH, "a", newline="") as f:
        csv.DictWriter(f, fieldnames=_CSV_FIELDS, extrasaction="ignore").writerow(row)


def update_iters(agent, optimizer, scaler, obs, act, lp, adv, ret, args):
    idx = np.arange(obs.shape[0])
    pl_list, vl_list, ent_list, kl_list, cf_list = [], [], [], [], []
    for _ in range(args.train_iters):
        np.random.shuffle(idx)
        for s in range(0, obs.shape[0], args.batch_size):
            b = idx[s:s + args.batch_size]
            pl, vl, ent, kl, cf = ppo_update(agent, optimizer, scaler,
                                              obs[b], act[b], ret[b], lp[b], adv[b],
                                              args.clip_eps, args.vf_coef, args.ent_coef)
            pl_list.append(pl); vl_list.append(vl); ent_list.append(ent)
            kl_list.append(kl); cf_list.append(cf)
            if abs(kl) > args.target_kl:
                return pl_list, vl_list, ent_list, kl_list, cf_list
    return pl_list, vl_list, ent_list, kl_list, cf_list


def ppo_epoch(agent, optimizer, scaler, buffer, args, obs, terms, truncs):
    with torch.no_grad():
        next_vals = agent.get_value(obs).reshape(1, -1)
    adv, ret = buffer.calculate_advantages(next_vals, terms.reshape(1, -1), truncs.reshape(1, -1))
    traj_obs, traj_act, traj_lp = buffer.get()
    flat = lambda t: t.view(-1, *t.shape[2:])
    traj_obs, traj_act, traj_lp = flat(traj_obs), flat(traj_act), flat(traj_lp)
    traj_adv = adv.view(-1)
    traj_adv = (traj_adv - traj_adv.mean()) / (traj_adv.std() + 1e-8)
    return update_iters(agent, optimizer, scaler, traj_obs, traj_act, traj_lp, traj_adv, ret.view(-1), args)


def build_log(stats, ppo_lists, epoch, args, eval_env, agent, device):
    pl_list, vl_list, ent_list, kl_list, cf_list = ppo_lists
    log = {
        "reward/total":     stats["reward"],
        "reward/forward":   stats["forward"],
        "reward/clearance": stats["clearance"],
        "reward/upright":   stats["upright"],
        "reward/arm":       stats["arm"],
        "train/ep_len":     stats["ep_len"],
        "ppo/policy_loss":  np.mean(pl_list),
        "ppo/value_loss":   np.mean(vl_list),
        "ppo/entropy":      np.mean(ent_list),
        "ppo/kl":           np.mean(kl_list),
        "ppo/clip_frac":    np.mean(cf_list),
    }
    if epoch % args.video_every == 0:
        frames, ep_reward, ep_len = run_episode(eval_env, agent, device)
        log["eval/video"]  = wandb.Video(frames_to_video(frames), fps=30, format="mp4")
        log["eval/reward"] = ep_reward
        log["eval/ep_len"] = ep_len
    _log_csv({"epoch": epoch, **log})
    return log
