import numpy as np
import wandb
from stable_baselines3.common.callbacks import BaseCallback

_KEYS = ("r_beat", "r_energy", "r_alive", "r_diversity")
_SHORT = {"r_beat": "beat", "r_energy": "energy", "r_alive": "alive", "r_diversity": "diversity"}


class RewardLogCallback(BaseCallback):
    def __init__(self):
        super().__init__()
        self._accum = {k: [] for k in _KEYS}
        self._genre_accum = {}

    def _on_step(self) -> bool:
        for info in self.locals.get("infos", []):
            genre = info.get("genre")
            for key in _KEYS:
                if key in info:
                    self._accum[key].append(info[key])
                    if genre:
                        self._genre_accum.setdefault(genre, {k: [] for k in _KEYS})[key].append(info[key])
        return True

    def _on_rollout_end(self) -> None:
        log = {}
        # overall mean per metric → "reward" section
        for key, vals in self._accum.items():
            if vals:
                log[f"reward/{_SHORT[key]}"] = np.mean(vals)
        # per-genre per-metric → one section per metric, 3 genre lines each
        for genre, accum in self._genre_accum.items():
            for key, vals in accum.items():
                if vals:
                    log[f"{_SHORT[key]}/{genre}"] = np.mean(vals)
        if log:
            wandb.log(log)
        self._accum = {k: [] for k in _KEYS}
        self._genre_accum = {}
