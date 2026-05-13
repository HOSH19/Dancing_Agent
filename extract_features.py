"""
Usage: python extract_features.py --audio audio/hiphop.mp3 --genre hiphop
Saves features/<genre>.npy
"""
import argparse
import numpy as np
from pathlib import Path
from audio.features import build_feature_dict

GENRES = ["hiphop", "waltz", "edm"]

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--audio", required=True)
    parser.add_argument("--genre", required=True, choices=GENRES)
    parser.add_argument("--out_dir", default="features")
    args = parser.parse_args()

    feat = build_feature_dict(args.audio, args.genre)
    Path(args.out_dir).mkdir(exist_ok=True)
    out = f"{args.out_dir}/{args.genre}.npy"
    np.save(out, feat)
    print(f"Saved {feat['n_rl_steps']} steps → {out}  (tempo={feat['tempo']:.1f} BPM)")
