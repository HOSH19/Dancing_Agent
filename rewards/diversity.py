import numpy as np

GENRES = ["hiphop", "waltz", "edm"]
EMA_ALPHA = 0.01


def gait_vector(step_freqs: list, com_heights: list) -> np.ndarray:
    return np.array([
        np.mean(step_freqs) if step_freqs else 0.0,
        np.std(com_heights) if com_heights else 0.0,
    ], dtype=np.float32)


class DiversityTracker:
    def __init__(self):
        self._means = {g: np.zeros(2) for g in GENRES}
        self._counts = {g: 0 for g in GENRES}

    def update(self, genre: str, vec: np.ndarray) -> float:
        others = [g for g in GENRES if g != genre and self._counts[g] > 0]
        r = 0.0
        for other in others:
            n1, n2 = np.linalg.norm(vec) + 1e-8, np.linalg.norm(self._means[other]) + 1e-8
            r -= float(np.dot(vec / n1, self._means[other] / n2))
        if others:
            r /= len(others)
        self._means[genre] = (1 - EMA_ALPHA) * self._means[genre] + EMA_ALPHA * vec
        self._counts[genre] += 1
        return r
