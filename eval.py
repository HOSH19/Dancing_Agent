import argparse
import sys
import os

import gymnasium as gym
import torch
import wandb

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "ppo"))
from lib.agent_ppo import PPOAgent
from eval_utils import run_episode, frames_to_video


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="ppo/model.pt")
    p.add_argument("--episodes", type=int, default=3)
    p.add_argument("--wandb_project", default="dancing-agent")
    p.add_argument("--run_name", default="eval-pretrained")
    return p.parse_args()


def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    env = gym.make("Humanoid-v5", render_mode="rgb_array")
    obs_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]

    agent = PPOAgent(obs_dim, action_dim).to(device)
    agent.load_state_dict(torch.load(args.model, map_location=device))
    agent.eval()

    wandb.init(project=args.wandb_project, name=args.run_name, config=vars(args))

    for ep in range(args.episodes):
        frames, total_reward, steps = run_episode(env, agent, device)
        video_path = frames_to_video(frames)

        wandb.log({
            "episode": ep,
            "reward": total_reward,
            "episode_length": steps,
            "video": wandb.Video(video_path, fps=30, format="mp4"),
        })
        print(f"Episode {ep+1}: reward={total_reward:.1f}, steps={steps}")

    env.close()
    wandb.finish()


if __name__ == "__main__":
    main()
