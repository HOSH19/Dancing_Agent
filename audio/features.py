import numpy as np
import librosa
from audio.resample import SR, HOP_LENGTH, norm, to_rl_grid


def _beat_phase(beat_times: np.ndarray, frame_times: np.ndarray, tempo: float) -> np.ndarray:
    beat_period = 60.0 / tempo
    phase = np.zeros(len(frame_times))
    for i, t in enumerate(frame_times):
        prev = beat_times[beat_times <= t]
        last = prev[-1] if len(prev) else 0.0
        phase[i] = min((t - last) / beat_period, 1.0)
    return phase


def _beat_indicator(beat_frames: np.ndarray, n_frames: int) -> np.ndarray:
    indicator = np.zeros(n_frames)
    for bf in beat_frames:
        if bf < n_frames:
            indicator[bf] = 1.0
    return indicator


def extract_raw(y: np.ndarray) -> dict:
    n_frames = int(len(y) / HOP_LENGTH)
    tempo, beat_frames = librosa.beat.beat_track(y=y, sr=SR, hop_length=HOP_LENGTH)
    tempo = float(np.atleast_1d(tempo)[0])
    beat_times = librosa.frames_to_time(beat_frames, sr=SR, hop_length=HOP_LENGTH)
    frame_times = librosa.frames_to_time(np.arange(n_frames), sr=SR, hop_length=HOP_LENGTH)
    return {
        "n_frames": n_frames,
        "tempo": tempo,
        "beat_phase": _beat_phase(beat_times, frame_times, tempo),
        "beat_indicator": _beat_indicator(beat_frames, n_frames),
        "onset_strength": norm(librosa.onset.onset_strength(y=y, sr=SR, hop_length=HOP_LENGTH)),
        "rms_energy": norm(librosa.feature.rms(y=y, hop_length=HOP_LENGTH)[0]),
        "spectral_centroid": norm(librosa.feature.spectral_centroid(y=y, sr=SR, hop_length=HOP_LENGTH)[0]),
    }


def build_feature_dict(audio_path: str, genre: str) -> dict:
    y, _ = librosa.load(audio_path, sr=SR)
    raw = extract_raw(y)
    n = raw["n_frames"]

    def r(feat): return to_rl_grid(feat, n)

    n_rl_steps = len(r(raw["beat_phase"]))
    return {
        "genre": genre,
        "tempo": raw["tempo"],
        "tempo_normalized": raw["tempo"] / 200.0,
        "n_rl_steps": n_rl_steps,
        "beat_phase": r(raw["beat_phase"]),
        "beat_indicator": r(raw["beat_indicator"]),
        "onset_strength": r(raw["onset_strength"]),
        "rms_energy": r(raw["rms_energy"]),
        "spectral_centroid": r(raw["spectral_centroid"]),
    }
