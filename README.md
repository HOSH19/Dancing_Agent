# Dancing Agent

Teaching a MuJoCo humanoid to walk in time with music — footsteps synchronized to the beat using reinforcement learning.

No reference motion data. No pre-recorded choreography. Just audio-derived reward signals.

## Status

- [x] Baseline: pre-trained locomotion policy evaluated and logged to W&B
- [ ] Phase 2: fine-tune with beat-sync reward

## Quick start

```bash
/Users/hoshuhan/miniforge3/envs/dancing-agent/bin/python eval.py --episodes 3
```

Logs episode rewards, lengths, and video to W&B project `dancing-agent`.

## Project structure

```
ppo/          # upstream PPO-Humanoid (pre-trained model + agent code)
eval.py       # evaluate pre-trained model, log to W&B
eval_utils.py # episode rollout and frame capture
```

## Approach

**Phase 1** (current): verify the pre-trained humanoid walks stably.

**Phase 2** (planned): add a beat-sync reward on top:
- Gaussian bonus for footstrikes landing near the beat peak
- Lift bonus to break the no-step attractor
- Dense beat-pulse signal for a continuous gradient

The core question — *can a humanoid learn rhythmically timed locomotion purely from music structure?* — is unaddressed in published work as of May 2026. The closest work ([DFM, Watanabe et al. 2025](https://arxiv.org/abs/2502.10980)) imitates pre-recorded motion capture data and has no music input at inference time.

## Credits

Locomotion baseline from [ProfessorNova/PPO-Humanoid](https://github.com/ProfessorNova/PPO-Humanoid) — a clean PPO implementation for Gymnasium's `Humanoid-v5`. The pre-trained `model.pt` and agent architecture (`ppo/lib/agent_ppo.py`) are used unchanged.
