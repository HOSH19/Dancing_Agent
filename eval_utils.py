import numpy as np
import imageio
import tempfile
import os


def frames_to_video(frames: list[np.ndarray], fps: int = 30) -> str:
    path = os.path.join(tempfile.mkdtemp(), "eval.mp4")
    imageio.mimwrite(path, frames, fps=fps, codec="libx264")
    return path


def run_episode(env, agent, device) -> tuple[list[np.ndarray], float, int]:
    import torch

    obs, _ = env.reset()
    frames, total_reward, steps = [], 0.0, 0
    done = False

    while not done:
        frame = env.render()
        if frame is not None:
            frames.append(frame)

        with torch.no_grad():
            obs_t = torch.tensor(np.array([obs], dtype=np.float32), device=device)
            action, _, _, _ = agent.get_action_and_value(obs_t)

        obs, reward, terminated, truncated, _ = env.step(action.squeeze(0).cpu().numpy())
        total_reward += reward
        steps += 1
        done = terminated or truncated

    return frames, total_reward, steps
