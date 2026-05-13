"""
Usage:
    python train.py                        # all 3 genres, 3M steps
    python train.py --genre hiphop         # single genre sanity check
    python train.py --timesteps 1_000_000
"""
import argparse
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import SubprocVecEnv, VecNormalize
from stable_baselines3.common.callbacks import CheckpointCallback, EvalCallback
from envs.reward_env import make_env
from envs.obs_space import GENRES


def build_envs(features_dir, genre, n_envs):
    env = SubprocVecEnv([make_env(features_dir, genre) for _ in range(n_envs)])
    return VecNormalize(env, norm_obs=True, norm_reward=True, clip_obs=10.0)


def build_model(env):
    return PPO("MlpPolicy", env, n_steps=2048, batch_size=64, n_epochs=10,
               learning_rate=3e-4, ent_coef=0.01, gamma=0.99, gae_lambda=0.95,
               clip_range=0.2, verbose=1, tensorboard_log="runs/")


def build_callbacks(features_dir, genre, n_envs, save_dir):
    eval_env = SubprocVecEnv([make_env(features_dir, genre)])
    eval_env = VecNormalize(eval_env, norm_obs=True, norm_reward=False, training=False)
    return [
        CheckpointCallback(save_freq=max(500_000 // n_envs, 1), save_path=save_dir, name_prefix="dance_agent"),
        EvalCallback(eval_env, eval_freq=max(100_000 // n_envs, 1), n_eval_episodes=3,
                     best_model_save_path=f"{save_dir}/best", verbose=1),
    ]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--features_dir", default="features")
    p.add_argument("--genre", default=None, choices=GENRES + [None])
    p.add_argument("--timesteps", type=int, default=3_000_000)
    p.add_argument("--n_envs", type=int, default=4)
    p.add_argument("--save_dir", default="checkpoints")
    args = p.parse_args()

    env = build_envs(args.features_dir, args.genre, args.n_envs)
    model = build_model(env)
    callbacks = build_callbacks(args.features_dir, args.genre, args.n_envs, args.save_dir)
    model.learn(total_timesteps=args.timesteps, callback=callbacks)
    model.save(f"{args.save_dir}/dance_agent_final")
    env.save(f"{args.save_dir}/vec_normalize.pkl")


if __name__ == "__main__":
    main()
