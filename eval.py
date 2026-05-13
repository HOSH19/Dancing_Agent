"""
Usage:
    python eval.py --checkpoint checkpoints/dance_agent_final
    python eval.py --checkpoint checkpoints/best/best_model --genre hiphop
"""
import argparse
import os
import numpy as np
import imageio
import mujoco
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from envs.reward_env import make_env
from envs.obs_space import GENRES

VIDEO_FPS = 30
RENDER_HW = 480
MAX_STEPS = 1500


def record(model, vec_normalize_path, features_dir, genre, out_path):
    env = VecNormalize.load(vec_normalize_path, DummyVecEnv([make_env(features_dir, genre)]))
    env.training = False
    env.norm_reward = False
    obs, frames, total_reward = env.reset(), [], 0.0
    renderer = mujoco.Renderer(env.envs[0].unwrapped.model, height=RENDER_HW, width=RENDER_HW)
    for _ in range(MAX_STEPS):
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, done, _ = env.step(action)
        total_reward += float(reward[0])
        renderer.update_scene(env.envs[0].unwrapped.data, camera="track")
        frames.append(renderer.render())
        if done[0]:
            break
    imageio.mimsave(out_path, frames, fps=VIDEO_FPS)
    print(f"  [{genre}] {len(frames)} frames, reward={total_reward:.1f} → {out_path}")
    env.close()
    return frames


def save_comparison(all_frames, out_dir):
    min_len = min(len(f) for f in all_frames)
    combined = [np.concatenate([f[i] for f in all_frames], axis=1) for i in range(min_len)]
    path = f"{out_dir}/all_genres_comparison.mp4"
    imageio.mimsave(path, combined, fps=VIDEO_FPS)
    print(f"\nSide-by-side → {path}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--vec_normalize", default="checkpoints/vec_normalize.pkl")
    p.add_argument("--features_dir", default="features")
    p.add_argument("--genre", default=None, choices=GENRES + [None])
    p.add_argument("--out_dir", default="videos")
    args = p.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    model = PPO.load(args.checkpoint)
    genres = [args.genre] if args.genre else GENRES
    all_frames = [record(model, args.vec_normalize, args.features_dir, g, f"{args.out_dir}/{g}.mp4") for g in genres]
    if len(all_frames) == 3:
        save_comparison(all_frames, args.out_dir)


if __name__ == "__main__":
    main()
