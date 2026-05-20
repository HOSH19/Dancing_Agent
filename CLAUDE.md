# CLAUDE.md

## Code Style

- **Functions must be short** — if a function exceeds ~20 lines, split it.
- **Files must be under 100 lines** — if a file grows past 100 lines, split it into focused modules.
- **Aggressive separation of concerns** — one file = one responsibility. No file should do two things.
- **No comments that describe what the code does** — only comment the non-obvious *why*.
- **No docstrings** — clear naming is sufficient.
- **No unnecessary abstractions** — don't generalize until there are 3+ concrete use cases.

## Dependencies

Use the miniforge3 `dancing-agent` env — it's the arm64 build required by MuJoCo on Apple Silicon:
```bash
/Users/hoshuhan/miniforge3/envs/dancing-agent/bin/python
```
The anaconda3 `dancing-agent` env is x86_64 and will fail with MuJoCo.

## Project Structure

```
ppo/              # upstream PPO-Humanoid repo (do not modify)
  model.pt        # pre-trained Humanoid-v5 checkpoint
  lib/            # PPOAgent, buffer, utils
eval.py           # run pre-trained model, log videos + metrics to W&B
eval_utils.py     # frame capture and episode rollout helpers
```

Planned additions:
```
rewards/          # beat reward computation only, no env logic
envs/             # environment wrappers only, no reward logic
train.py          # thin orchestration script
```

Top-level scripts stay thin — just arg parsing and orchestration.

## Project Goal

Teach a MuJoCo humanoid to **walk in time with music** — footsteps synchronized to the beat.

**Current status**: pre-trained locomotion model evaluated and logged to W&B. The agent walks stably (ep_len up to 1000 steps, reward ~2k–6k).

**Next phase**: add beat-sync reward on top of the existing locomotion policy. Two-phase curriculum:

- **Phase 1**: verify stable walking from the pre-trained checkpoint (done).
- **Phase 2**: fine-tune with beat reward — footstrikes timed to beat phase.

## Reward Design (Phase 2 plan)

Base locomotion reward matches Gymnasium HumanoidV5 defaults. Beat sync added on top:

```
total = r_locomotion + w_beat * r_beat
```

### Beat reward components

| Signal | Formula | Notes |
|---|---|---|
| Gaussian landing bonus | `exp(-8 * (1 - beat_phase)²)` | fires on each foot landing near beat peak |
| `LIFT_BONUS` | `0.15` per foot lift | breaks no-step attractor |
| `BEAT_PULSE` | `0.05 * beat_phase` | dense gradient before sync emerges |

### Known failure modes to avoid

**No-step attractor**: beat reward only on landing → agent learns to never lift feet → beat gradient is zero. Fix: `LIFT_BONUS` per foot-lift.

**Contact detection**: `abs(contact.dist)` (penetration depth ~0.002) is always below any reasonable threshold. Must use `mujoco.mj_contactForce()` with threshold ~1 N (standing humanoid produces ~80 N).

**Standing-still local optimum**: no forward velocity signal → agent maximizes alive bonus by standing still. Fix: `r_forward = 1.25 * qvel[0]` (or tanh variant).

**Eval obs normalization drift**: if using VecNormalize, deep-copy obs_rms/ret_rms from training env before each eval, otherwise eval diverges after 500K+ steps.

## PPO Health Targets

| Metric | Target | Danger zone |
|---|---|---|
| `clip_fraction` | 0.05–0.20 | > 0.40 → reduce LR or increase n_steps |
| `approx_kl` | 0.005–0.05 | > 0.10 → policy diverging |
| `ep_len_mean` | > 300 by 1M steps | < 150 at 1M → survival reward not working |

Stable PPO config (validated): `n_steps=8192`, `batch_size=256`, `n_epochs=5`, `lr=1e-4`, `target_kl=0.05`.

## W&B Logging

Project: `dancing-agent` under account `hoshuhan`.

Run eval:
```bash
/Users/hoshuhan/miniforge3/envs/dancing-agent/bin/python eval.py --episodes 3
```
