import copy
import numpy as np
import wandb
from stable_baselines3.common.callbacks import EvalCallback
from stable_baselines3.common.vec_env import VecNormalize


class WandbEvalCallback(EvalCallback):
    def _on_step(self) -> bool:
        if self.eval_freq > 0 and self.n_calls % self.eval_freq == 0:
            self._sync_normalize()
        result = super()._on_step()
        if self.eval_freq > 0 and self.n_calls % self.eval_freq == 0:
            wandb.log({
                "eval/mean_reward": self.last_mean_reward,
                "eval/mean_ep_length": np.mean(self.evaluations_length[-1]) if self.evaluations_length else 0,
            })
        return result

    def _sync_normalize(self):
        train_env = self.model.get_env()
        if isinstance(train_env, VecNormalize) and isinstance(self.eval_env, VecNormalize):
            self.eval_env.obs_rms = copy.deepcopy(train_env.obs_rms)
            self.eval_env.ret_rms = copy.deepcopy(train_env.ret_rms)
