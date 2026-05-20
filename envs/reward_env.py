import numpy as np
from stable_baselines3.common.monitor import Monitor
from envs.dance_env import DanceEnv
from rewards.reward_fn import WalkRewardTracker


class DanceEnvWithReward(DanceEnv):
    def __init__(self, *args, reward_weights: dict = None, **kwargs):
        super().__init__(*args, **kwargs)
        self._tracker = WalkRewardTracker(**(reward_weights or {}))

    def reset(self, **kwargs):
        self._tracker.reset_episode()
        return super().reset(**kwargs)

    def step(self, action):
        obs, base_reward, terminated, truncated, info = super().step(action)
        reward, reward_info = self._tracker.update(
            beat_phase=info.get("beat_phase", 0.0),
            com_height=float(self.data.qpos[2]),
            forward_vel=float(self.data.qvel[0]),
            action_sq_norm=float(np.mean(action ** 2)),
            foot_contacts=self.foot_contacts(),
        )
        info.update(reward_info)
        return obs, reward, terminated, truncated, info


def make_env(features_dir: str, genre: str = None, reward_weights: dict = None):
    def _init():
        return Monitor(DanceEnvWithReward(features_dir=features_dir, genre=genre,
                                         reward_weights=reward_weights))
    return _init
