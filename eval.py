import argparse
import sys
import os

import gymnasium as gym
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "ppo"))
from lib.agent_ppo import PPOAgent
from eval_utils import run_episode, frames_to_video


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="ppo/model.pt")
    p.add_argument("--episodes", type=int, default=3)
    p.add_argument("--wandb_project", default="dancing-agent")
    p.add_argument("--run_name", default="eval-pretrained")
    p.add_argument("--no-wandb", action="store_true")
    p.add_argument("--out", default="eval_output", help="directory for local video output")
    return p.parse_args()


def load_agent(model_path, device):
    env = gym.make("Humanoid-v5", render_mode="rgb_array")
    agent = PPOAgent(env.observation_space.shape[0], env.action_space.shape[0]).to(device)
    agent.load_state_dict(torch.load(model_path, map_location=device))
    agent.eval()
    return env, agent


def eval_loop(env, agent, device, n_episodes, args):
    use_wandb = not args.no_wandb
    if use_wandb:
        import wandb
        wandb.init(project=args.wandb_project, name=args.run_name, config=vars(args))

    if args.no_wandb:
        os.makedirs(args.out, exist_ok=True)

    for ep in range(n_episodes):
        frames, total_reward, steps = run_episode(env, agent, device)
        video_path = frames_to_video(frames)
        print(f"Episode {ep+1}: reward={total_reward:.1f}, steps={steps}")
        if use_wandb:
            import wandb
            wandb.log({
                "episode": ep,
                "reward": total_reward,
                "episode_length": steps,
                "video": wandb.Video(video_path, fps=30, format="mp4"),
            })
        else:
            dest = os.path.join(args.out, f"ep{ep+1}.mp4")
            os.replace(video_path, dest)
            print(f"  saved → {dest}")

    if use_wandb:
        import wandb
        wandb.finish()


def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    env, agent = load_agent(args.model, device)
    eval_loop(env, agent, device, args.episodes, args)
    env.close()


if __name__ == "__main__":
    main()
