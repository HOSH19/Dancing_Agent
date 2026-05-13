import numpy as np
from pathlib import Path
from gymnasium.envs.mujoco.humanoid_v5 import HumanoidEnv
from envs.obs_space import GENRES, extend_obs_space, audio_obs
from envs.contacts import get_foot_contacts


class DanceEnv(HumanoidEnv):
    def __init__(self, features_dir: str = "features", genre: str = None, **kwargs):
        self._features_dir = Path(features_dir)
        self._fixed_genre = genre
        self._cache: dict = {}
        self._feat = None
        self._current_genre = None
        self._audio_step = 0
        super().__init__(**kwargs)
        self.observation_space = extend_obs_space(self.observation_space)

    def _load(self, genre: str) -> dict | None:
        if genre not in self._cache:
            path = self._features_dir / f"{genre}.npy"
            if not path.exists():
                return None
            self._cache[genre] = np.load(path, allow_pickle=True).item()
        return self._cache[genre]

    def _full_obs(self, base_obs: np.ndarray) -> np.ndarray:
        return np.concatenate([base_obs, audio_obs(self._feat, self._audio_step, self._current_genre)]).astype(np.float32)

    def reset(self, *, seed=None, options=None):
        genre = self._fixed_genre or np.random.choice(GENRES)
        self._current_genre = genre
        self._feat = self._load(genre)
        self._audio_step = 0
        obs, info = super().reset(seed=seed, options=options)
        info["genre"] = genre
        return self._full_obs(obs), info

    def step(self, action):
        obs, reward, terminated, truncated, info = super().step(action)
        self._audio_step += 1
        t = min(self._audio_step, self._feat["n_rl_steps"] - 1) if self._feat else 0
        info["genre"] = self._current_genre
        info["beat_indicator"] = float(self._feat["beat_indicator"][t]) if self._feat else 0.0
        info["rms_energy"] = float(self._feat["rms_energy"][t]) if self._feat else 0.0
        info["audio_step"] = self._audio_step
        if self._feat and self._audio_step >= self._feat["n_rl_steps"]:
            truncated = True
        return self._full_obs(obs), reward, terminated, truncated, info

    def foot_contacts(self) -> dict:
        return get_foot_contacts(self.model, self.data)
