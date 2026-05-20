import math

_TARGET_SPEED = 1.25  # m/s — comfortable human walking speed
_ALIVE_HEIGHT_MIN = 1.0
_ALIVE_HEIGHT_GOOD = 1.3


def walk_reward(forward_vel: float, com_height: float, action_sq_norm: float) -> tuple[float, float, float]:
    r_forward = float(math.tanh(forward_vel / _TARGET_SPEED))
    r_alive = float(min(1.0, max(0.0, (com_height - _ALIVE_HEIGHT_MIN) / (_ALIVE_HEIGHT_GOOD - _ALIVE_HEIGHT_MIN))))
    r_ctrl = float(action_sq_norm)
    return r_forward, r_alive, r_ctrl
