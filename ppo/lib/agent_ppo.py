import torch
import torch.nn as nn
from torch.distributions import Normal


def _mlp(in_dim, out_dim, hidden=512):
    return nn.Sequential(
        nn.Linear(in_dim, hidden), nn.Tanh(),
        nn.Linear(hidden, hidden), nn.Tanh(),
        nn.Linear(hidden, hidden), nn.Tanh(),
        nn.Linear(hidden, out_dim),
    )


class PPOAgent(nn.Module):
    def __init__(self, num_inputs: int, num_actions: int):
        super().__init__()
        self.actor_mu = nn.Sequential(*_mlp(num_inputs, num_actions), nn.Tanh())
        self.actor_logstd = nn.Parameter(torch.ones(1, num_actions) * -0.5)
        self.critic = _mlp(num_inputs, 1)

    def forward(self, x):
        mu = self.actor_mu(x)
        return mu, torch.exp(self.actor_logstd).expand_as(mu)

    def get_value(self, x):
        return self.critic(x)

    def get_action_and_value(self, x, action=None):
        mu, std = self.forward(x)
        dist = Normal(mu, std)
        if action is None:
            action = dist.rsample()
        return action, dist.log_prob(action).sum(-1), dist.entropy().mean(-1), self.get_value(x)
