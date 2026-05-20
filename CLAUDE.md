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

**No-step attractor** (`rewards/beat.py`): beat reward only fires on foot *landing* (contact rising edge). Agent discovers it can avoid the termination risk of stepping by never lifting feet — landing never happens, beat gradient is zero. Fix: add `LIFT_BONUS = 0.05` per foot-lift event (contact falling edge) to give a dense gradient signal for any stepping behavior.

**Eval VecNormalize never synced** (`callbacks/wandb_eval_callback.py`): eval env created once at callback construction with `training=False`. After 500K+ training steps, obs_rms diverges from default → model acts on mis-scaled obs → agent falls in 1–2 steps → `eval/mean_ep_length ≈ 0`. Fix: deep-copy obs_rms/ret_rms from training env at the start of each `_on_step` eval trigger.

### Genre-specific rewards (`rewards/genre_reward.py`)

Replaced diversity (always −0.5, useless penalty) with direct movement-quality rewards per genre:

| Genre | Reward | Signal |
|---|---|---|
| waltz | CoM height std over last 20 steps × 20 | Rise-and-fall oscillation |
| hiphop | Landing force / 200 N (per foot, capped 1.0) | Stomp intensity |
| edm | 1.0 if both feet airborne, else 0 | Jump / bounce |

These fire per step (waltz, edm) or per landing event (hiphop). Weight controlled by `w_genre` in configs.

### Current weights (r_alive removed — constant 1.0 contributes nothing to gradient)
```python
W_BEAT, W_ENERGY, W_GENRE = 0.5, 0.35, 0.15  # control config
```
Approximate per-step contribution when dancing well: beat ≈ 0.15–0.25 (includes lift bonus), energy ≈ 0.10, genre ≈ 0.05–0.15.

### Diagnosing future regressions
- If `reward/alive` dominates WandB charts → alive signal is too strong again
- If `beat/edm = 0` throughout → contact detection is broken (check force threshold)
- If `ep_len_mean` plateaus below 200 steps → agent found standing-still local optimum
- If beat reward declines after initial spike → agent settling into no-movement policy; increase `W_BEAT` or add a per-step stepping incentive
- If `eval/mean_ep_length ≈ 0` while training episodes are long → eval VecNormalize is out of sync (check `_sync_normalize` in `WandbEvalCallback`)
- If diversity reward is consistently −0.3 to −0.7 → diversity is acting as pure penalty (gait hasn't differentiated per genre); run `no_diversity` ablation to isolate
