from rewards.beat import beat_reward
from rewards.energy import EnergyCorrelation
from rewards.genre_reward import genre_reward


class DanceRewardTracker:
    def __init__(self, w_beat=0.5, w_energy=0.35, w_genre=0.15):
        self._w = (w_beat, w_energy, w_genre)
        self._energy = EnergyCorrelation()
        self._prev_contacts: dict = {}
        self._com_heights: list = []

    def reset_episode(self):
        self._energy.reset()
        self._prev_contacts = {}
        self._com_heights = []

    def update(self, *, genre, beat_phase, rms_energy,
               com_velocity, com_height, foot_contacts) -> tuple:
        r_beat, events = beat_reward(foot_contacts, self._prev_contacts, beat_phase)
        self._com_heights.append(com_height)
        r_energy = self._energy.update(com_velocity, rms_energy)
        r_genre = genre_reward(genre, foot_contacts, events, self._com_heights)
        wb, we, wg = self._w
        total = wb * r_beat + we * r_energy + wg * r_genre
        return float(total), {"r_beat": r_beat, "r_energy": r_energy, "r_genre": r_genre}
