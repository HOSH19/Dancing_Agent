"""
Usage:
    python train.py --config configs/control.json
    python train.py --config configs/beat_heavy.json --timesteps 1_000_000
"""
import os
os.environ.setdefault("MUJOCO_GL", "osmesa")
import json
import argparse
import wandb
from wandb.integration.sb3 import WandbCallback
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import SubprocVecEnv, VecNormalize
from stable_baselines3.common.callbacks import CheckpointCallback
from envs.reward_env import make_env
from envs.obs_space import GENRES
from callbacks.video_callback import VideoCallback
from callbacks.reward_log_callback import RewardLogCallback
from callbacks.wandb_eval_callback import WandbEvalCallback


def build_envs(features_dir, genre, n_envs, reward_weights):
    env = SubprocVecEnv([make_env(features_dir, genre, reward_weights) for _ in range(n_envs)])
    return VecNormalize(env, norm_obs=True, norm_reward=True, clip_obs=10.0)


def build_model(env):
    return PPO("MlpPolicy", env, n_steps=8192, batch_size=256, n_epochs=5,
               learning_rate=1e-4, ent_coef=0.02, gamma=0.99, gae_lambda=0.95,
               clip_range=0.2, target_kl=0.05, verbose=1, device="cpu")


def build_callbacks(features_dir, genre, n_envs, eval_freq, save_dir, reward_weights):
    eval_env = SubprocVecEnv([make_env(features_dir, genre, reward_weights)])
    eval_env = VecNormalize(eval_env, norm_obs=True, norm_reward=False, training=False)
    return [
        CheckpointCallback(save_freq=max(500_000 // n_envs, 1), save_path=save_dir, name_prefix="dance_agent"),
        WandbEvalCallback(eval_env, eval_freq=max(eval_freq // n_envs, 1), n_eval_episodes=3,
                          best_model_save_path=f"{save_dir}/best", verbose=1),
    ]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--features_dir", default="features")
    p.add_argument("--genre", default=None, choices=GENRES + [None])
    p.add_argument("--timesteps", type=int, default=3_000_000)
    p.add_argument("--n_envs", type=int, default=4)
    p.add_argument("--eval_freq", type=int, default=50_000)
    p.add_argument("--save_dir", default="checkpoints")
    p.add_argument("--video_dir", default="videos/training")
    p.add_argument("--video_freq", type=int, default=500_000)
    p.add_argument("--run_name", type=str, default=None)
    p.add_argument("--config", type=str, default=None,
                   help="Path to reward weights JSON (e.g. configs/walk_phase1.json)")
    p.add_argument("--load_checkpoint", type=str, default=None,
                   help="Path to a saved PPO model to continue training from (e.g. for phase 2)")
    args = p.parse_args()

    reward_weights = json.loads(open(args.config).read()) if args.config else {}
    if args.run_name is None and args.config:
        args.run_name = os.path.splitext(os.path.basename(args.config))[0]

    run = wandb.init(
        project="dancing-agent",
        name=args.run_name,
        config=vars(args),
        sync_tensorboard=False,
        monitor_gym=False,
    )

    env = build_envs(args.features_dir, args.genre, args.n_envs, reward_weights)
    if args.load_checkpoint:
        model = PPO.load(args.load_checkpoint, env=env)
    else:
        model = build_model(env)
    genres = [args.genre] if args.genre else GENRES
    video_cb = VideoCallback(
        features_dir=args.features_dir,
        out_dir=args.video_dir,
        genres=genres,
        record_freq=args.video_freq,
        verbose=1,
    )
    wandb_cb = WandbCallback(
        gradient_save_freq=10_000,
        model_save_path=f"{args.save_dir}/wandb",
        verbose=1,
    )
    callbacks = build_callbacks(args.features_dir, args.genre, args.n_envs, args.eval_freq, args.save_dir, reward_weights)
    callbacks += [video_cb, wandb_cb, RewardLogCallback()]
    model.learn(total_timesteps=args.timesteps, callback=callbacks)
    run.finish()
    model.save(f"{args.save_dir}/dance_agent_final")
    env.save(f"{args.save_dir}/vec_normalize.pkl")


if __name__ == "__main__":
    main()
