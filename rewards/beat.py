FOOT_FORCE_THRESHOLD = 0.01


def beat_reward(foot_contacts: dict, prev_contacts: dict, beat_indicator: float) -> tuple[float, dict]:
    r = 0.0
    step_events = {}
    for foot, force in foot_contacts.items():
        is_contact = force > FOOT_FORCE_THRESHOLD
        if is_contact and not prev_contacts.get(foot, False):
            if beat_indicator > 0.5:
                r += 1.0
            step_events[foot] = True
        prev_contacts[foot] = is_contact
    return r, step_events
