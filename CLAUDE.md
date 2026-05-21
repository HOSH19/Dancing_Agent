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
envs/
  walk_env.py     # reward shaping wrapper (foot clearance + upright)
  arm_env.py      # arm swing reward synced to beat phase
  reward_norm.py  # running mean/std reward normalizer (per-env)
audio/
  beat_track.py   # beat_phase(), constant_beats(), load_beats()
collect.py        # rollout collection
ppo_train.py      # PPO update loop + CSV/W&B logging
ppo_update.py     # single gradient step
train.py          # orchestration: args, env setup, train loop
eval.py           # run model, save video locally or log to W&B
eval_utils.py     # frame capture and episode rollout helpers
configs/          # JSON run configs (wandb flag, hyperparams)
log/              # train.log + metrics.csv (wiped on each new run)
checkpoints/      # saved model weights
```

Top-level scripts stay thin — just arg parsing and orchestration.

## Running

```bash
# training run (no wandb)
nohup python train.py --config configs/arm_v2.json > log/train.log 2>&1 &

# evaluation — save video locally
python eval.py --model checkpoints/walk_finetune.pt --no-wandb

# evaluation — log to W&B
python eval.py --model checkpoints/walk_finetune.pt
```

W&B is controlled by the `"wandb"` flag in the config file — never pass CLI args.

## Project Goal

Teach a MuJoCo humanoid to **swing its arms in sync with a music beat** while walking.

The base gait (forward walking) comes from the pre-trained `ppo/model.pt`. We finetune it to add beat-synchronized arm swing without disrupting the walking behavior.

**Longer-term**: sync the stepping gait itself to the beat (currently too hard to break the shuffling attractor).

## Env Stack

```
RewardNorm(ArmEnv(WalkEnv(gym.make("Humanoid-v5"))))
```

- `WalkEnv` — adds foot clearance (alternating-gait) + upright reward on top of base
- `ArmEnv` — adds Gaussian arm-sync reward keyed to beat phase
- `RewardNorm` — divides each step reward by running std of discounted returns; prevents value function collapse

## Arm Sync Reward (ArmEnv)

- Shoulder joints: `right_shoulder1`, `left_shoulder1` (accessed via `qposadr` — must cast to `int`)
- Beat phase φ ∈ [0,1]: computed from sim time against `beat_times` array each step
- Target: `r_target = SWING_AMP * sin(2π φ)`, `l_target = SWING_AMP * sin(2π(φ + 0.5))`
- Reward: `exp(-BEAT_K * err²)` per shoulder, bounded [0, 1] each, sum max = 2.0
- `STEP_DT = 0.015s` (5 substeps × 0.003s) — used to advance sim time `_t`

### Key: `qposadr` returns a numpy array — must use `int(qposadr[0])` for scalar indexing into `qpos`

## Arm Sync Runs

| Config | BEAT_K | ARM_W | LR | Checkpoint | Outcome |
|---|---|---|---|---|---|
| arm-v1 (walk_v9.json) | 5.0 | 2.0 | 3e-5 | ppo/model.pt | Plateaued at ~1.43/2.0 (~70%) after ~85 epochs; noisy ±0.15 oscillation |
| arm-v2 (arm_v2.json) | 8.0 | 4.0 | 1e-5 | walk_finetune_v1.pt | In progress |

Plateau diagnosis: soft Gaussian (BEAT_K=5) gave partial credit too easily; ARM_W=2 too weak relative to base reward (~2.6 combined forward+upright). Sharper kernel + higher weight + lower LR to refine.

## Gait Reward (WalkEnv) — Historical

Spent significant effort trying to teach high-stepping gait. All attempts failed to break the shuffling attractor in the pre-trained model:

| Run | Signal | Weight | Outcome |
|---|---|---|---|
| v6 | `max(0, xpos - 0.04)` | 8 | Zero gradient — threshold below resting foot height (~0.05m) |
| v7 | `min(max(0, xpos-0.10), 0.10)` | 3 | Zero gradient — threshold too high for shuffler to reach |
| v8 | `min(max(0, xpos-0.065), 0.085)` | 20 | Flat at ~0.01 — shuffling attractor too strong |
| v9 | `max(0, cvel[fid,5])` foot above floor | 15 | Unbounded — agent flaps feet wildly, ep_len→136, falls |
| v10 | alternating-gait contact constraint | 20 | Implemented but pivoted to arm sync before full test |

**Decision**: pivoted to arm sync since gait change requires training from scratch or much larger reward signal.

## MuJoCo Body References

- Foot body names: `"right_foot"`, `"left_foot"`
- Resting foot height: ~0.05m (`data.xpos[fid, 2]`)
- Upright signal: `data.xmat[torso_id, 8]` = R[2,2] = 1.0 when perfectly upright
- Foot vertical velocity: `data.cvel[fid, 5]` (spatial vel layout: `[wx,wy,wz, vx,vy,vz]`)
- Contact force: must use `mujoco.mj_contactForce()` with threshold ~1 N; `contact.dist` is always near zero and unusable

## PPO Health Targets

| Metric | Target | Danger zone |
|---|---|---|
| `clip_fraction` | 0.05–0.20 | > 0.40 → reduce LR or increase n_steps |
| `approx_kl` | 0.005–0.05 | > 0.10 → policy diverging |
| `ep_len_mean` | > 500 | < 200 → reward destabilizing balance |
| `value_loss` | < 1.0 with RewardNorm | growing → RewardNorm not working |

Validated PPO config: `n_steps=2048`, `batch_size=256`, `train_iters=10`, `target_kl=0.02`, `gamma=0.99`.
LR: 3e-5 for fresh finetuning; 1e-5 for refinement from a partially-trained checkpoint.

## W&B Logging

Project: `dancing-agent` under account `hoshuhan`.
Enable by setting `"wandb": true` in the config JSON. Off by default to avoid polluting the dashboard during debug runs.
