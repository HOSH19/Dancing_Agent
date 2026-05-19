FOOT_FORCE_THRESHOLD = 1.0  # Newtons


def beat_reward(foot_contacts: dict, prev_contacts: dict, beat_phase: float) -> tuple[float, dict]:
    r = 0.0
    step_events = {}
    # phase-based bonus: peaks at beat_phase=1 (just before beat), decays quickly
    phase_bonus = float(__import__("math").exp(-8.0 * (1.0 - beat_phase) ** 2))
    for foot, force in foot_contacts.items():
        is_contact = force > FOOT_FORCE_THRESHOLD
        if is_contact and not prev_contacts.get(foot, False):
            r += phase_bonus
            step_events[foot] = True
        prev_contacts[foot] = is_contact
    return r, step_events
