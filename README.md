# Dancing Agent

A MuJoCo humanoid trained with PPO to tap its feet to music — and move differently depending on the genre (hip-hop, waltz, EDM).

## How it works

The humanoid's observation is extended with audio features extracted offline via librosa:

| Feature | Description |
|---|---|
| `beat_phase` | Sawtooth 0→1, resets on each beat |
| `onset_strength` | Percussive attack intensity |
| `rms_energy` | Loudness |
| `spectral_centroid` | Brightness (bass vs treble) |
| `tempo_normalized` | BPM / 200 |
| `genre_onehot` | 3-dim one-hot: hiphop / waltz / edm |

The reward has three components plus a genre-specific shaping term:

```
R = w_beat   * R_beat    # foot strikes near beats (Gaussian phase bonus + lift bonus)
  + w_energy * R_energy  # CoM velocity correlates with audio RMS energy
  + w_genre  * R_genre   # genre-specific movement quality (see below)
```

**Genre rewards:**
- **Waltz** — CoM height variance (rise-and-fall motion)
- **HipHop** — landing force per foot (stomp intensity)
- **EDM** — both feet simultaneously airborne (jumping)

Reward weights are set via JSON config (`configs/`). Default: `w_beat=0.50, w_energy=0.35, w_genre=0.15`.

## Project structure

```
├── audio/                    # royalty-free MP3s, one per genre
├── configs/                  # reward weight ablations (JSON)
├── envs/
│   ├── dance_env.py          # MuJoCo humanoid + music observation
│   ├── obs_space.py          # observation space helpers
│   ├── contacts.py           # foot contact force (mj_contactForce)
│   └── reward_env.py         # reward wrapper + env factory
├── rewards/
│   ├── beat.py               # beat sync + foot-lift reward
│   ├── energy.py             # energy correlation (rolling window)
│   ├── genre_reward.py       # waltz / hiphop / edm shaping
│   └── reward_fn.py          # composite tracker
├── callbacks/
│   ├── video_callback.py     # records per-genre MP4s during training
│   ├── reward_log_callback.py# logs reward components to WandB
│   └── wandb_eval_callback.py# eval with VecNormalize sync
├── extract_features.py       # audio → features/<genre>.npy
├── train.py                  # PPO training entry point
└── eval.py                   # checkpoint → MP4 export
```

## Setup

```bash
conda create -n dancing-agent python=3.11
conda activate dancing-agent
pip install "mujoco>=3.0" "gymnasium[mujoco]>=1.0" "stable-baselines3>=2.3" \
            "imageio[ffmpeg]" librosa wandb numpy
```

## Usage

**1. Add audio files** (one per genre):
```
audio/hiphop.mp3  audio/waltz.mp3  audio/edm.mp3
```

**2. Extract features:**
```bash
python extract_features.py --audio audio/hiphop.mp3 --genre hiphop
python extract_features.py --audio audio/waltz.mp3  --genre waltz
python extract_features.py --audio audio/edm.mp3    --genre edm
```

**3. Train with a reward config:**
```bash
python train.py --config configs/control.json --timesteps 3_000_000
```

Available configs: `control`, `beat_heavy`, `energy_heavy`, `no_diversity`.

**4. Export videos:**
```bash
python eval.py --checkpoint checkpoints/dance_agent_final
```

## Related work

- [DeepMimic](https://xbpeng.github.io/projects/DeepMimic/) — motion imitation from mocap (requires reference clips; this project does not)
- [Music-Driven Legged Robots (arXiv 2503.04063)](https://arxiv.org/abs/2503.04063) — beat sync for quadrupeds
