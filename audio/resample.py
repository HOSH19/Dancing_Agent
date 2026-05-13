import numpy as np
import librosa

MUJOCO_DT = 0.05
SR = 22050
HOP_LENGTH = 512


def librosa_times(n_frames: int) -> np.ndarray:
    return librosa.frames_to_time(np.arange(n_frames), sr=SR, hop_length=HOP_LENGTH)


def to_rl_grid(feat: np.ndarray, n_frames: int) -> np.ndarray:
    n_rl_steps = int(len(feat) * (HOP_LENGTH / SR) / MUJOCO_DT)
    rl_times = np.arange(n_rl_steps) * MUJOCO_DT
    indices = np.searchsorted(librosa_times(n_frames), rl_times, side="right") - 1
    return feat[np.clip(indices, 0, len(feat) - 1)]


def norm(x: np.ndarray) -> np.ndarray:
    lo, hi = x.min(), x.max()
    return (x - lo) / (hi - lo + 1e-8)
