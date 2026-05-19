# CLAUDE.md

## Code Style

- **Functions must be short** — if a function exceeds ~20 lines, split it.
- **Files must be under 100 lines** — if a file grows past 100 lines, split it into focused modules.
- **Aggressive separation of concerns** — one file = one responsibility. No file should do two things.
- **No comments that describe what the code does** — only comment the non-obvious *why*.
- **No docstrings** — clear naming is sufficient.
- **No unnecessary abstractions** — don't generalize until there are 3+ concrete use cases.

## Project Structure

Keep modules flat and focused:
- `envs/` — environment wrappers only, no reward logic
- `rewards/` — reward computation only, no env logic
- Top-level scripts (`train.py`, `eval.py`, `extract_features.py`) stay thin — just arg parsing and orchestration

## Dependencies

Use the `dancing-agent` conda environment:
```bash
conda activate dancing-agent
```

## Reward Design — Lessons Learned

### Bugs fixed (do not reintroduce)

**Contact detection** (`envs/contacts.py`): originally used `abs(contact.dist)` (penetration depth ≈ 0.002) against a threshold of 0.01 — contacts were *never* detected. Must use `mujoco.mj_contactForce()` with a threshold of ~1 N. Standing humanoid produces ~80 N ground reaction force.

**Alive reward domination** (`rewards/reward_fn.py`): `base_reward` from `HumanoidEnv` includes a hardcoded `healthy_reward=5.0` constant. Even at `W_ALIVE=0.05`, this contributes 0.25/step vs beat's 0.012/step. Fix: set `r_alive = 1.0` (constant survival flag) so alive contributes 0.05/step and beat can dominate.

**Beat reward sparsity** (`rewards/beat.py`): original binary `beat_indicator > 0.5` fired only on the exact beat frame (~4.4% of steps). Replaced with a Gaussian over `beat_phase`: `exp(-8 * (1 - beat_phase)^2)`, giving partial credit across the whole beat cycle.

### Current weights
```python
W_BEAT, W_ENERGY, W_ALIVE, W_DIVERSITY = 0.5, 0.3, 0.05, 0.15
```
Approximate per-step contribution when dancing well: beat ≈ 0.25, energy ≈ 0.09, alive = 0.05, diversity ≈ 0.

### Diagnosing future regressions
- If `reward/alive` dominates WandB charts → alive signal is too strong again
- If `beat/edm = 0` throughout → contact detection is broken (check force threshold)
- If `ep_len_mean` plateaus below 200 steps → agent found standing-still local optimum
- If beat reward declines after initial spike → agent settling into no-movement policy; increase `W_BEAT` or add a per-step stepping incentive
