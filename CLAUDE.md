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

## Project Goal

Teach a MuJoCo humanoid to **walk in time with music** — footsteps synchronized to the beat. This is a two-phase curriculum:

- **Phase 1** (`configs/walk_phase1.json`): Pure locomotion. No music. Rewards forward velocity, staying upright, and penalizes control cost. Goal: stable walking before any music signal is introduced.
- **Phase 2** (`configs/walk_phase2.json`): Locomotion + beat sync. Builds on a Phase 1 checkpoint. Adds `w_beat` to reward footstrikes timed to the beat phase.

Training commands:
```bash
# Phase 1
python train.py --config configs/walk_phase1.json --timesteps 2000000 --run_name walk_phase1

# Phase 2 (after Phase 1 checkpoint saved)
python train.py --config configs/walk_phase2.json --timesteps 2000000 \
  --load_checkpoint checkpoints/dance_agent_final --run_name walk_phase2
```

Phase 1 success criteria before moving to Phase 2: `ep_len_mean > 300`, `reward/forward` rising steadily, video shows continuous forward walking.

## Reward Design

### Weights

**Phase 1** (`walk_phase1.json`):
```json
{"w_forward": 0.60, "w_alive": 0.30, "w_ctrl": 0.10, "w_beat": 0.00}
```

**Phase 2** (`walk_phase2.json`):
```json
{"w_forward": 0.40, "w_alive": 0.25, "w_ctrl": 0.10, "w_beat": 0.25}
```

All weights must sum to 1.0 (note: `w_ctrl` is subtracted, so `w_forward + w_alive + w_beat = 0.90` and `w_ctrl = 0.10`).

### Component signals

| Signal | Formula | Notes |
|---|---|---|
| `r_forward` | `tanh(qvel[0] / 1.25)` | Peaks at 1.25 m/s walking speed |
| `r_alive` | `clamp((com_height - 1.0) / 0.3, 0, 1)` | Shaped survival; 1.3 m = full reward |
| `r_ctrl` | `mean(action²)` | Subtracted — penalizes flailing |
| `r_beat` | Gaussian phase bonus on landing + LIFT_BONUS + BEAT_PULSE | See beat.py |

### Beat reward details (`rewards/beat.py`)

- **Gaussian phase bonus**: `exp(-8 * (1 - beat_phase)²)` — fires on each foot landing, rewarding steps near the beat peak.
- **LIFT_BONUS = 0.15**: fires on each foot lift (contact falling edge). Breaks the no-step attractor where the agent avoids stepping to avoid the risk of falling.
- **BEAT_PULSE = 0.05**: dense per-step signal scaled by `beat_phase`. Provides a continuous gradient even before footstrike synchronization emerges.

## Bugs Fixed (do not reintroduce)

**Contact detection** (`envs/contacts.py`): originally used `abs(contact.dist)` (penetration depth ≈ 0.002) against a threshold of 0.01 — contacts were *never* detected. Must use `mujoco.mj_contactForce()` with a threshold of ~1 N. Standing humanoid produces ~80 N ground reaction force.

**No-step attractor** (`rewards/beat.py`): beat reward only fired on foot landing (contact rising edge). Agent learned to never lift feet — landing never happens, beat gradient is zero. Fix: `LIFT_BONUS = 0.15` per foot-lift.

**Eval VecNormalize never synced** (`callbacks/wandb_eval_callback.py`): eval env diverges from training obs_rms after 500K+ steps → model acts on mis-scaled obs → agent falls in 1–2 steps → `eval/mean_ep_length ≈ 0`. Fix: deep-copy obs_rms/ret_rms from training env before each eval.

**Standing-still local optimum**: when only `r_alive` provides positive reward and there's no forward velocity signal, the agent learns to stand motionless — maximizes alive, zero beat/energy. Fix: `r_forward = tanh(qvel[0] / 1.25)` eliminates this loophole by requiring actual walking.

**PPO instability** (`train.py`): with `n_steps=2048` and `lr=3e-4`, `clip_fraction` reached 0.81 and `approx_kl` reached 1.7–2.3. Fix: `n_steps=8192`, `batch_size=256`, `n_epochs=5`, `lr=1e-4`, `target_kl=0.05`. Verified at 400k steps: `clip_fraction=0.10`, `approx_kl=0.0096`.

## PPO Health Targets

| Metric | Target | Danger zone |
|---|---|---|
| `clip_fraction` | 0.05–0.20 | > 0.40 → reduce LR or increase n_steps |
| `approx_kl` | 0.005–0.05 | > 0.10 → policy diverging |
| `ep_len_mean` | > 300 by 1M steps (Phase 1) | < 150 at 1M → survival reward not working |

### Diagnosing regressions
- `ep_len_mean` plateaus below 150 → agent not learning to walk; check `reward/forward` is rising
- `reward/forward ≈ 0` with high `reward/alive` → standing-still local optimum; `r_forward` should be pulling the agent forward
- `reward/beat ≈ 0` in Phase 2 → beat signal not reaching the agent; check `BEAT_PULSE` and contact detection
- `eval/mean_ep_length ≈ 0` while training is long → eval VecNormalize out of sync
- `clip_fraction > 0.40` → reduce LR or lower `target_kl`

## Related Work / Novelty

**This project is novel as of May 2026.** The closest published work:

- **DFM: Deep Fourier Mimic** (Watanabe et al., Feb 2025, [arxiv:2502.10980](https://arxiv.org/abs/2502.10980)): humanoid RL for dance, but imitates *pre-recorded motion capture data* using Fourier representations. No music input at inference time.
- **Music-synchronized robot papers**: scripted choreography with beat detection, not RL.

**Key differentiator**: this project teaches a humanoid to walk in time with music using only audio-derived reward signals (beat phase, RMS energy) with no reference motion data and no pre-recorded choreography. The question — *can a humanoid learn rhythmically timed locomotion purely from music structure?* — is unaddressed in published work.
