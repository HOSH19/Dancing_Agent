import numpy as np
import gymnasium as gym

_CLIP = 10.0


class RewardNorm(gym.Wrapper):
    def __init__(self, env, gamma=0.99):
        super().__init__(env)
        self._ret = 0.0
        self._mean = 0.0
        self._M2 = 0.0
        self._n = 0
        self._gamma = gamma

    def step(self, action):
        obs, reward, term, trunc, info = self.env.step(action)
        self._ret = self._ret * self._gamma + reward
        self._n += 1
        delta = self._ret - self._mean
        self._mean += delta / self._n
        self._M2 += delta * (self._ret - self._mean)
        std = max(np.sqrt(self._M2 / self._n), 1e-8) if self._n > 1 else 1.0
        normed = float(np.clip(reward / std, -_CLIP, _CLIP))
        if term or trunc:
            self._ret = 0.0
        return obs, normed, term, trunc, info
