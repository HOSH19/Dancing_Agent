import torch
import torch.nn as nn


def _compute_losses(agent, obs, actions, returns, old_log_probs, advantages, clip_eps, vf_coef, ent_coef):
    _, new_log_probs, entropies, new_values = agent.get_action_and_value(obs, actions)
    ratio = torch.exp(new_log_probs - old_log_probs)
    kl = ((old_log_probs - new_log_probs) / actions.size(-1)).mean()
    surr1 = ratio * advantages
    surr2 = torch.clamp(ratio, 1 - clip_eps, 1 + clip_eps) * advantages
    clip_fraction = ((ratio < 1 - clip_eps) | (ratio > 1 + clip_eps)).float().mean()
    policy_loss = -torch.min(surr1, surr2).mean()
    value_loss = nn.MSELoss()(new_values.squeeze(1), returns)
    entropy = entropies.mean()
    loss = policy_loss + vf_coef * value_loss - ent_coef * entropy
    return loss, policy_loss, value_loss, entropy, kl, clip_fraction


def ppo_update(agent, optimizer, scaler, obs, actions, returns, old_log_probs, advantages, clip_eps, vf_coef, ent_coef):
    agent.train()
    optimizer.zero_grad()
    with torch.amp.autocast("cpu"):
        loss, pl, vl, ent, kl, cf = _compute_losses(
            agent, obs, actions, returns, old_log_probs, advantages, clip_eps, vf_coef, ent_coef)
    scaler.scale(loss).backward()
    scaler.unscale_(optimizer)
    nn.utils.clip_grad_norm_(agent.parameters(), 1.0)
    scaler.step(optimizer)
    scaler.update()
    return pl.item(), vl.item(), ent.item(), kl.item(), cf.item()
