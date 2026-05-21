import numpy as np
import gymnasium as gym
from audio.beat_track import beat_phase

SWING_AMP  = 0.5    # radians — natural shoulder swing range
BEAT_K     = 8.0    # Gaussian sharpness; higher = tighter sync required
ARM_W      = 4.0
STEP_DT    = 0.015  # Humanoid-v5 control dt (5 substeps × 0.003s)


class ArmEnv(gym.Wrapper):
    def __init__(self, env, beat_times: np.ndarray):
        super().__init__(env)
        m = env.unwrapped.model
        self._rs = int(m.joint("right_shoulder1").qposadr[0])
        self._ls = int(m.joint("left_shoulder1").qposadr[0])
        self._beat_times = beat_times
        self._t = 0.0

    def reset(self, **kwargs):
        self._t = 0.0
        return self.env.reset(**kwargs)

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        self._t += STEP_DT
        arm = self._arm_reward()
        info["reward_arm"] = arm
        return obs, reward + ARM_W * arm, terminated, truncated, info

    def _arm_reward(self):
        phase  = beat_phase(self._beat_times, self._t)
        qpos   = self.env.unwrapped.data.qpos
        r_target = SWING_AMP * np.sin(2 * np.pi * phase)
        l_target = SWING_AMP * np.sin(2 * np.pi * (phase + 0.5))
        r_err = (float(qpos[self._rs]) - r_target) ** 2
        l_err = (float(qpos[self._ls]) - l_target) ** 2
        return float(np.exp(-BEAT_K * r_err) + np.exp(-BEAT_K * l_err))
