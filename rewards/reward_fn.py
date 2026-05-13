from rewards.beat import beat_reward
from rewards.energy import EnergyCorrelation
from rewards.diversity import DiversityTracker, gait_vector

W_BEAT, W_ENERGY, W_ALIVE, W_DIVERSITY = 0.4, 0.2, 0.3, 0.1


class DanceRewardTracker:
    def __init__(self):
        self._energy = EnergyCorrelation()
        self._diversity = DiversityTracker()
        self._prev_contacts: dict = {}
        self._step_freqs: list = []
        self._com_heights: list = []
        self._last_contact_step: dict = {}

    def reset_episode(self):
        self._energy.reset()
        self._prev_contacts = {}
        self._step_freqs = []
        self._com_heights = []
        self._last_contact_step = {}

    def update(self, *, genre, base_reward, beat_indicator, rms_energy,
               com_velocity, com_height, foot_contacts, step) -> tuple:
        r_beat, events = beat_reward(foot_contacts, self._prev_contacts, beat_indicator)
        for foot in events:
            if foot in self._last_contact_step:
                dt = (step - self._last_contact_step[foot]) * 0.05
                if dt > 0:
                    self._step_freqs.append(1.0 / dt)
            self._last_contact_step[foot] = step

        self._com_heights.append(com_height)
        r_energy = self._energy.update(com_velocity, rms_energy)
        r_alive = base_reward
        r_diversity = self._diversity.update(genre, gait_vector(self._step_freqs, self._com_heights))

        total = W_BEAT * r_beat + W_ENERGY * r_energy + W_ALIVE * r_alive + W_DIVERSITY * r_diversity
        return float(total), {"r_beat": r_beat, "r_energy": r_energy, "r_alive": r_alive, "r_diversity": r_diversity}
