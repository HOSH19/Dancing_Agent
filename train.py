import argparse
import json
import sys
import os
import time

import gymnasium as gym
import numpy as np
import torch
import wandb

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "ppo"))
from lib.agent_ppo import PPOAgent
from lib.buffer_ppo import PPOBuffer
from envs.walk_env import WalkEnv
from envs.arm_env import ArmEnv
from envs.reward_norm import RewardNorm
from audio.beat_track import constant_beats, load_beats
from collect import collect
from ppo_train import ppo_epoch, build_log, init_csv


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--config", default="configs/walk_finetune.json")
    p.add_argument("--wandb", action="store_true", default=None)
    for name, t in [("checkpoint", str), ("run_name", str), ("epochs", int),
                    ("n_envs", int), ("n_steps", int), ("batch_size", int),
                    ("train_iters", int), ("video_every", int)]:
        p.add_argument(f"--{name}", type=t, default=None)
    for name in ("lr", "gamma", "gae_lambda", "clip_eps", "ent_coef", "vf_coef", "target_kl"):
        p.add_argument(f"--{name}", type=float, default=None)
    cli = p.parse_args()
    with open(cli.config) as f:
        cfg = json.load(f)
    for k, v in vars(cli).items():
        if k != "config" and v is not None:
            cfg[k] = v
    return argparse.Namespace(**cfg)


def _beat_times(args):
    audio = getattr(args, "audio", None)
    return load_beats(audio) if audio else constant_beats(getattr(args, "bpm", 120.0))


def make_env(render=False, beat_times=None):
    env = gym.make("Humanoid-v5", render_mode="rgb_array" if render else None)
    return RewardNorm(ArmEnv(WalkEnv(env), beat_times))


def setup(args, device):
    bt = _beat_times(args)
    envs = gym.vector.AsyncVectorEnv([lambda: make_env(render=False, beat_times=bt)] * args.n_envs)
    eval_env = make_env(render=True, beat_times=bt)
    obs_dim = envs.single_observation_space.shape
    act_dim = envs.single_action_space.shape
    agent = PPOAgent(obs_dim[0], act_dim[0]).to(device)
    agent.load_state_dict(torch.load(args.checkpoint, map_location=device))
    optimizer = torch.optim.Adam(agent.parameters(), lr=args.lr, eps=1e-5)
    scaler = torch.amp.GradScaler("cpu")
    buffer = PPOBuffer(obs_dim, act_dim, args.n_steps, args.n_envs, device, args.gamma, args.gae_lambda)
    return envs, eval_env, agent, optimizer, scaler, buffer


def train_loop(envs, eval_env, agent, optimizer, scaler, buffer, args, device):
    obs = torch.tensor(np.array(envs.reset()[0], dtype=np.float32), device=device)
    terms = torch.zeros(args.n_envs, device=device)
    truncs = torch.zeros(args.n_envs, device=device)
    for epoch in range(1, args.epochs + 1):
        t0 = time.time()
        obs, terms, truncs, stats = collect(envs, agent, buffer, device, obs, terms, truncs)
        ppo_lists = ppo_epoch(agent, optimizer, scaler, buffer, args, obs, terms, truncs)
        log = build_log(stats, ppo_lists, epoch, args, eval_env, agent, device)
        if args.wandb:
            wandb.log(log)
        kl, cf = np.mean(ppo_lists[3]), np.mean(ppo_lists[4])
        print(f"epoch {epoch:4d} | reward {stats['reward']:.3f} | fwd {stats['forward']:.3f} | "
              f"arm {stats['arm']:.3f} | upright {stats['upright']:.3f} | "
              f"ep_len {stats['ep_len']:.0f} | kl {kl:.4f} | clip {cf:.3f} | {time.time()-t0:.1f}s", flush=True)


def main():
    args = parse_args()
    device = torch.device("cpu")
    envs, eval_env, agent, optimizer, scaler, buffer = setup(args, device)
    init_csv()
    if args.wandb:
        wandb.init(project="dancing-agent", name=args.run_name, config=vars(args))
    train_loop(envs, eval_env, agent, optimizer, scaler, buffer, args, device)
    os.makedirs("checkpoints", exist_ok=True)
    torch.save(agent.state_dict(), "checkpoints/walk_finetune.pt")
    envs.close()
    eval_env.close()
    if args.wandb:
        wandb.finish()


if __name__ == "__main__":
    main()
