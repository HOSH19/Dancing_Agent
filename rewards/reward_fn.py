from rewards.beat import beat_reward
from rewards.walk import walk_reward


class WalkRewardTracker:
    def __init__(self, w_forward=0.6, w_alive=0.3, w_ctrl=0.1, w_beat=0.0, **_):
        self._w = (w_forward, w_alive, w_ctrl, w_beat)
        self._prev_contacts: dict = {}

    def reset_episode(self):
        self._prev_contacts = {}

    def update(self, *, beat_phase, com_height, forward_vel, action_sq_norm, foot_contacts) -> tuple:
        w_fwd, wa, wc, wb = self._w
        r_forward, r_alive, r_ctrl = walk_reward(forward_vel, com_height, action_sq_norm)
        r_beat = 0.0
        if wb > 0.0:
            r_beat, _ = beat_reward(foot_contacts, self._prev_contacts, beat_phase)
        total = w_fwd * r_forward + wa * r_alive - wc * r_ctrl + wb * r_beat
        return float(total), {"r_forward": r_forward, "r_alive": r_alive, "r_ctrl": r_ctrl, "r_beat": r_beat}
