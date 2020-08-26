import torch
import torch.nn as nn


class Prior(nn.Module):
    def log_prob(self, w):
        pass


class StandardNormal(Prior):
    def __init__(self):
        super().__init__()
        self.mu = nn.Parameter(torch.tensor(0., dtype=torch.float32), requires_grad=False)
        self.scale = nn.Parameter(torch.tensor(1., dtype=torch.float32), requires_grad=False)

    def log_prob(self, w):
        return torch.distributions.Normal(loc=self.mu, scale=self.scale).log_prob(w).sum()
