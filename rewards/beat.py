import math

FOOT_FORCE_THRESHOLD = 1.0  # Newtons
LIFT_BONUS = 0.05  # small dense reward for any foot lift, breaks no-step attractor


def beat_reward(foot_contacts: dict, prev_contacts: dict, beat_phase: float) -> tuple[float, dict]:
    r = 0.0
    step_events = {}
    phase_bonus = float(math.exp(-8.0 * (1.0 - beat_phase) ** 2))
    for foot, force in foot_contacts.items():
        is_contact = force > FOOT_FORCE_THRESHOLD
        was_contact = prev_contacts.get(foot, False)
        if is_contact and not was_contact:
            r += phase_bonus
            step_events[foot] = True
        elif was_contact and not is_contact:
            r += LIFT_BONUS
        prev_contacts[foot] = is_contact
    return r, step_events
