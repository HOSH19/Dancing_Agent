import numpy as np
from gymnasium import spaces

GENRES = ["hiphop", "waltz", "edm"]
GENRE_ONEHOT = {g: np.eye(3)[i] for i, g in enumerate(GENRES)}
AUDIO_DIM = 8  # beat_phase, onset, rms, centroid, tempo_norm, genre(3)


def extend_obs_space(base_space: spaces.Box) -> spaces.Box:
    return spaces.Box(
        low=np.concatenate([base_space.low, np.zeros(AUDIO_DIM, dtype=np.float32)]),
        high=np.concatenate([base_space.high, np.ones(AUDIO_DIM, dtype=np.float32)]),
        dtype=np.float32,
    )


def audio_obs(feat: dict | None, audio_step: int, genre: str) -> np.ndarray:
    if feat is None:
        return np.zeros(AUDIO_DIM, dtype=np.float32)
    t = min(audio_step, feat["n_rl_steps"] - 1)
    return np.array([
        feat["beat_phase"][t],
        feat["onset_strength"][t],
        feat["rms_energy"][t],
        feat["spectral_centroid"][t],
        feat["tempo_normalized"],
        *GENRE_ONEHOT[genre],
    ], dtype=np.float32)
