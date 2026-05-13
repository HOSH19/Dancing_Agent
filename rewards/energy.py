import numpy as np
from collections import deque

ENERGY_WINDOW = 40


class EnergyCorrelation:
    def __init__(self):
        self._speeds: deque = deque(maxlen=ENERGY_WINDOW)
        self._audios: deque = deque(maxlen=ENERGY_WINDOW)

    def reset(self):
        self._speeds.clear()
        self._audios.clear()

    def update(self, com_velocity: float, rms_energy: float) -> float:
        self._speeds.append(com_velocity)
        self._audios.append(rms_energy)
        if len(self._speeds) < 4:
            return 0.0
        corr = np.corrcoef(list(self._speeds), list(self._audios))[0, 1]
        return float(np.clip(corr, 0.0, 1.0)) if np.isfinite(corr) else 0.0
