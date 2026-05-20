import torch


class PPOBuffer:
    def __init__(self, obs_dim, act_dim, size, num_envs, device, gamma=0.99, gae_lambda=0.95):
        self.capacity = size
        self.gamma, self.gae_lambda = gamma, gae_lambda
        self.ptr = 0
        z = lambda *shape: torch.zeros(shape, dtype=torch.float32, device=device)
        self.obs_buf     = z(size, num_envs, *obs_dim)
        self.act_buf     = z(size, num_envs, *act_dim)
        self.rew_buf     = z(size, num_envs)
        self.val_buf     = z(size, num_envs)
        self.term_buf    = z(size, num_envs)
        self.trunc_buf   = z(size, num_envs)
        self.logprob_buf = z(size, num_envs)

    def store(self, obs, act, rew, val, term, trunc, logprob):
        self.obs_buf[self.ptr] = obs
        self.act_buf[self.ptr] = act
        self.rew_buf[self.ptr] = rew
        self.val_buf[self.ptr] = val
        self.term_buf[self.ptr] = term
        self.trunc_buf[self.ptr] = trunc
        self.logprob_buf[self.ptr] = logprob
        self.ptr += 1

    def calculate_advantages(self, last_vals, last_terminateds, last_truncateds):
        assert self.ptr == self.capacity
        with torch.no_grad():
            adv_buf = torch.zeros_like(self.rew_buf)
            last_gae = 0.0
            for t in reversed(range(self.capacity)):
                next_vals  = last_vals         if t == self.capacity - 1 else self.val_buf[t + 1]
                term_mask  = 1.0 - last_terminateds if t == self.capacity - 1 else 1.0 - self.term_buf[t + 1]
                trunc_mask = 1.0 - last_truncateds  if t == self.capacity - 1 else 1.0 - self.trunc_buf[t + 1]
                delta = self.rew_buf[t] + self.gamma * next_vals * term_mask - self.val_buf[t]
                last_gae = delta + self.gamma * self.gae_lambda * term_mask * trunc_mask * last_gae
                adv_buf[t] = last_gae
            return adv_buf, adv_buf + self.val_buf

    def get(self):
        assert self.ptr == self.capacity
        self.ptr = 0
        return self.obs_buf, self.act_buf, self.logprob_buf
