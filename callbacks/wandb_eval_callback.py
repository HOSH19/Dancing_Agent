import numpy as np
import wandb
from stable_baselines3.common.callbacks import EvalCallback


class WandbEvalCallback(EvalCallback):
    def _on_step(self) -> bool:
        result = super()._on_step()
        if self.eval_freq > 0 and self.n_calls % self.eval_freq == 0:
            wandb.log({
                "eval/mean_reward": self.last_mean_reward,
                "eval/mean_ep_length": np.mean(self.evaluations_length[-1]) if self.evaluations_length else 0,
            })
        return result
