import numpy as np

_FEET = ("right_foot", "left_foot")
_FOOT_FORCE_THRESHOLD = 1.0
_WALTZ_HEIGHT_WINDOW = 20  # ~1 second of history to measure rise-and-fall


def genre_reward(genre: str, foot_contacts: dict, step_events: dict, com_heights: list) -> float:
    if genre == "waltz":
        return _waltz(com_heights)
    if genre == "hiphop":
        return _hiphop(foot_contacts, step_events)
    if genre == "edm":
        return _edm(foot_contacts)
    return 0.0


def _waltz(com_heights: list) -> float:
    # Waltz: characteristic rise-and-fall motion. Reward CoM height variance over recent steps.
    # Typical variation during dancing: ±2–5 cm → std ≈ 0.01–0.03 m → scaled to 0.2–0.6.
    if len(com_heights) < 6:
        return 0.0
    return float(np.std(com_heights[-_WALTZ_HEIGHT_WINDOW:]) * 20.0)


def _hiphop(foot_contacts: dict, step_events: dict) -> float:
    # HipHop: reward heavy landings (stomp). Normalize against ~200 N peak impact force.
    return sum(min(foot_contacts.get(foot, 0.0) / 200.0, 1.0) for foot in step_events)


def _edm(foot_contacts: dict) -> float:
    # EDM: reward both feet leaving the ground simultaneously (jump/bounce).
    both_airborne = all(foot_contacts.get(f, 0.0) < _FOOT_FORCE_THRESHOLD for f in _FEET)
    return 1.0 if both_airborne else 0.0
