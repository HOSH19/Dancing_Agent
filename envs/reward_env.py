import numpy as np
from stable_baselines3.common.monitor import Monitor
from envs.dance_env import DanceEnv
from rewards.reward_fn import DanceRewardTracker


class DanceEnvWithReward(DanceEnv):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._tracker = DanceRewardTracker()

    def reset(self, **kwargs):
        self._tracker.reset_episode()
        return super().reset(**kwargs)

    def step(self, action):
        obs, base_reward, terminated, truncated, info = super().step(action)
        reward, reward_info = self._tracker.update(
            genre=info.get("genre", "hiphop"),
            base_reward=base_reward,
            beat_phase=info.get("beat_phase", 0.0),
            rms_energy=info.get("rms_energy", 0.0),
            com_velocity=float(np.linalg.norm(self.data.qvel[:3])),
            com_height=float(self.data.qpos[2]),
            foot_contacts=self.foot_contacts(),
            step=info.get("audio_step", 0),
        )
        info.update(reward_info)
        return obs, reward, terminated, truncated, info


def make_env(features_dir: str, genre: str = None):
    def _init():
        return Monitor(DanceEnvWithReward(features_dir=features_dir, genre=genre))
    return _init
