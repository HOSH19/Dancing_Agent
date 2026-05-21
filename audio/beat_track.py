import numpy as np


def load_beats(audio_path: str) -> np.ndarray:
    import librosa
    _, beat_frames = librosa.beat.beat_track(y=librosa.load(audio_path)[0],
                                             sr=librosa.load(audio_path)[1])
    return librosa.frames_to_time(beat_frames, sr=librosa.load(audio_path)[1])


def constant_beats(bpm: float, duration_s: float = 600.0) -> np.ndarray:
    interval = 60.0 / bpm
    return np.arange(0.0, duration_s, interval)


def beat_phase(beat_times: np.ndarray, t: float) -> float:
    if t <= beat_times[0]:
        return 0.0
    idx = np.searchsorted(beat_times, t, side="right") - 1
    if idx >= len(beat_times) - 1:
        interval = beat_times[-1] - beat_times[-2]
        return ((t - beat_times[-1]) % interval) / interval
    return (t - beat_times[idx]) / (beat_times[idx + 1] - beat_times[idx])
