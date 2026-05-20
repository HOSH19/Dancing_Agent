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


def load_agent(model_path, device):
    env = gym.make("Humanoid-v5", render_mode="rgb_array")
    agent = PPOAgent(env.observation_space.shape[0], env.action_space.shape[0]).to(device)
    agent.load_state_dict(torch.load(model_path, map_location=device))
    agent.eval()
    return env, agent


def eval_loop(env, agent, device, n_episodes):
    for ep in range(n_episodes):
        frames, total_reward, steps = run_episode(env, agent, device)
        wandb.log({
            "episode": ep,
            "reward": total_reward,
            "episode_length": steps,
            "video": wandb.Video(frames_to_video(frames), fps=30, format="mp4"),
        })
        print(f"Episode {ep+1}: reward={total_reward:.1f}, steps={steps}")


def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    env, agent = load_agent(args.model, device)
    wandb.init(project=args.wandb_project, name=args.run_name, config=vars(args))
    eval_loop(env, agent, device, args.episodes)
    env.close()
    wandb.finish()


if __name__ == "__main__":
    main()
