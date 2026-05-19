import os
import imageio
import mujoco
import numpy as np
import wandb
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from envs.reward_env import make_env
from envs.obs_space import GENRES

VIDEO_FPS = 30
RENDER_HW = 480
MAX_STEPS = 600


class VideoCallback(BaseCallback):
    def __init__(self, features_dir: str, out_dir: str, genres: list,
                 record_freq: int, vec_normalize_path: str = None, verbose: int = 0):
        super().__init__(verbose)
        self._features_dir = features_dir
        self._out_dir = out_dir
        self._genres = genres
        self._record_freq = record_freq
        self._vec_normalize_path = vec_normalize_path
        self._last_recorded = -1

    def _on_step(self) -> bool:
        if self.num_timesteps - self._last_recorded < self._record_freq:
            return True
        self._last_recorded = self.num_timesteps
        step_dir = os.path.join(self._out_dir, f"step_{self.num_timesteps:09d}")
        os.makedirs(step_dir, exist_ok=True)
        all_frames = []
        for genre in self._genres:
            frames = self._record_genre(genre, step_dir)
            all_frames.append(frames)
        if len(all_frames) > 1:
            self._save_comparison(all_frames, step_dir)
        return True

    def _record_genre(self, genre: str, out_dir: str) -> list:
        env = DummyVecEnv([make_env(self._features_dir, genre)])
        if self._vec_normalize_path and os.path.exists(self._vec_normalize_path):
            env = VecNormalize.load(self._vec_normalize_path, env)
        else:
            env = VecNormalize(env, norm_obs=True, norm_reward=False, training=False)
        env.training = False
        env.norm_reward = False

        obs = env.reset()
        renderer = mujoco.Renderer(env.envs[0].unwrapped.model, height=RENDER_HW, width=RENDER_HW)
        frames = []
        for _ in range(MAX_STEPS):
            action, _ = self.model.predict(obs, deterministic=True)
            obs, _, done, _ = env.step(action)
            renderer.update_scene(env.envs[0].unwrapped.data, camera="track")
            frames.append(renderer.render())
            if done[0]:
                break
        env.close()
        out_path = os.path.join(out_dir, f"{genre}.mp4")
        imageio.mimsave(out_path, frames, fps=VIDEO_FPS)
        wandb.log({f"video/{genre}": wandb.Video(out_path, fps=VIDEO_FPS, format="mp4")},
                  step=self.num_timesteps)
        if self.verbose:
            print(f"  [video] {genre} → {out_path}")
        return frames

    def _save_comparison(self, all_frames: list, out_dir: str):
        min_len = min(len(f) for f in all_frames)
        combined = [np.concatenate([f[i] for f in all_frames], axis=1) for i in range(min_len)]
        path = os.path.join(out_dir, "all_genres.mp4")
        imageio.mimsave(path, combined, fps=VIDEO_FPS)
        wandb.log({"video/all_genres": wandb.Video(path, fps=VIDEO_FPS, format="mp4")},
                  step=self.num_timesteps)
        if self.verbose:
            print(f"  [video] side-by-side → {path}")
