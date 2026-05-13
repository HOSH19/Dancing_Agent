# Dancing Agent 💃🕺

A reinforcement learning agent that learns to dance to music. A MuJoCo humanoid is trained with PPO to synchronize its gait with audio features — and crucially, to move **differently** depending on the genre (hip-hop, waltz, EDM).

## How it works

The humanoid's observation is extended with 8 music-derived features extracted offline via librosa:

| Feature | Description |
|---|---|
| `beat_phase` | Sawtooth 0→1, resets on each beat |
| `onset_strength` | Percussive attack intensity |
| `rms_energy` | Loudness |
| `spectral_centroid` | Brightness (bass vs treble) |
| `tempo_normalized` | BPM / 200 |
| `genre_onehot` | 3-dim one-hot: hiphop / waltz / edm |

The reward has four components:

```
R = 0.4 * R_beat      # foot strikes aligned with beats
  + 0.2 * R_energy    # movement speed correlates with loudness
  + 0.3 * R_alive     # standard humanoid survival bonus
  + 0.1 * R_diversity # gait must look different across genres
```

The diversity reward penalizes cosine similarity between the current gait and the running mean gait of other genres — forcing distinct movement styles to emerge.

## Project structure

```
dancing-agent/
├── audio/
│   ├── features.py       # librosa feature extraction
│   └── resample.py       # resampling to MuJoCo timestep grid
├── envs/
│   ├── dance_env.py      # MuJoCo humanoid + music observation
│   ├── obs_space.py      # observation space extension helpers
│   ├── contacts.py       # foot contact force extraction
│   └── reward_env.py     # reward wrapper + env factory
├── rewards/
│   ├── beat.py           # beat synchronization reward
│   ├── energy.py         # energy correlation reward
│   ├── diversity.py      # genre diversity reward
│   └── reward_fn.py      # composite tracker
├── extract_features.py   # CLI: audio → features/<genre>.npy
├── train.py              # PPO training
└── eval.py               # rollout → .mp4 video export
```

## Setup

```bash
conda create -n dancing-agent python=3.11
conda activate dancing-agent
pip install "mujoco>=3.0" "gymnasium[mujoco]>=1.0" "stable-baselines3>=2.3" \
            "imageio[ffmpeg]" librosa numpy
```

## Usage

**1. Add audio files** (royalty-free MP3s, one per genre):
```
audio/hiphop.mp3
audio/waltz.mp3
audio/edm.mp3
```

**2. Extract features:**
```bash
python extract_features.py --audio audio/hiphop.mp3 --genre hiphop
python extract_features.py --audio audio/waltz.mp3  --genre waltz
python extract_features.py --audio audio/edm.mp3    --genre edm
```

**3. Train** (~5–8 hours on M2):
```bash
python train.py --timesteps 3_000_000
```

**4. Export videos:**
```bash
python eval.py --checkpoint checkpoints/dance_agent_final
# → videos/hiphop.mp4, waltz.mp4, edm.mp4, all_genres_comparison.mp4
```

## What to expect

| Training steps | Behavior |
|---|---|
| 0–100k | Falls immediately |
| 500k | Starts standing/shuffling |
| 1.5M | Walks clumsily |
| 3M+ | Stable gait with genre-dependent rhythm |

## Inspiration & related work

- [DeepMimic](https://xbpeng.github.io/projects/DeepMimic/) — motion imitation from mocap (requires reference clips; this project does not)
- [Music-Driven Legged Robots (arXiv 2503.04063)](https://arxiv.org/abs/2503.04063) — beat sync for quadrupeds (no style variation)
- [Karl Sims, Evolving Virtual Creatures (1994)](https://www.karlsims.com/papers/siggraph94.pdf)
