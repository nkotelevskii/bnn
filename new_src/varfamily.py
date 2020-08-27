import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from pyro.distributions.transforms import AffineAutoregressive
from pyro.nn import AutoRegressiveNN
import pdb



def make_positive(x):
    # return torch.exp(x)
    return F.softplus(x)


class VF(nn.Module):
    def __init__(self, shape, use_bias):
        super().__init__()
        self.shape = shape
        self.use_bias = use_bias

        self.weight = None
        if self.use_bias:
            self.bias = None

    def reset_parameters(self):
        pass

    def get_weight_sample(self):
        pass

    def get_bias_sample(self):
        pass

    def get_weight_mean(self):
        pass

    def get_bias_mean(self):
        pass

    def log_prob(self):
        pass

    def forward(self, sample=True):
        if self.training or sample:
            self.weight = self.get_weight_sample()
            if self.use_bias:
                self.bias = self.get_bias_sample()
        else:
            self.weight = self.get_weight_mean()
            self.bias = self.get_bias_mean() if self.use_bias else None

        return self.weight, self.bias


class MeanField(VF):
    def __init__(self, shape, use_bias):
        super().__init__(shape, use_bias)

        # define raw parameters
        self.weight_mu = nn.Parameter(torch.empty(shape))
        self.weight_rho = nn.Parameter(torch.empty(shape))

        if self.use_bias:
            self.bias_mu = nn.Parameter(torch.empty(shape[0]))
            self.bias_rho = nn.Parameter(torch.empty(shape[0]))
        else:
            self.register_parameter('bias_mu', None)
            self.register_parameter('bias_rho', None)
        self.reset_parameters()

    def reset_parameters(self, ):
        torch.nn.init.xavier_uniform_(self.weight_mu)
        torch.nn.init.xavier_uniform_(self.weight_rho)
        if self.use_bias:
            self.bias_mu.data.fill_(0.0)
            self.bias_rho.data.fill_(0.0)

    def get_weight_sigma(self):
        return make_positive(self.weight_rho)

    def get_bias_sigma(self):
        return make_positive(self.bias_rho)

    def get_weight_sample(self):
        return torch.distributions.Normal(self.weight_mu, self.get_weight_sigma()).rsample()

    def get_bias_sample(self):
        return torch.distributions.Normal(self.bias_mu, self.get_bias_sigma()).rsample()

    def get_weight_mean(self):
        return self.weight_mu

    def get_bias_mean(self):
        return self.bias_mu

    def log_prob(self):
        '''
        Note, that here we are using last sampled weights and biases
        :return: logprob
        '''
        log_prob = torch.distributions.Normal(self.weight_mu, self.get_weight_sigma()).log_prob(self.weight).sum()
        if self.use_bias:
            log_prob += torch.distributions.Normal(self.bias_mu, self.get_bias_sigma()).log_prob(self.bias).sum()
        return log_prob


class NF(MeanField):
    def __init__(self, shape, use_bias, t=1):
        super().__init__(shape, use_bias)
        self.jac_w = 0.
        self.jac_b = 0.

        self.t = t

        self.transform_w = self.get_transform_w()
        self.transform_b = self.get_transform_b()

    def get_weight_sample(self):
        pdb.set_trace()
        w_0 = torch.distributions.Normal(self.weight_mu, self.get_weight_sigma()).rsample().view(1, -1)
        self.jac_w = 0.
        for t in self.transform_w:
            w = t(w_0)
            self.jac_w += t.log_abs_det_jacobian(w_0, w)
            w_0 = w
        return w.view(*self.shape)

    def get_bias_sample(self):
        b_0 = torch.distributions.Normal(self.bias_mu, self.get_bias_sigma()).rsample()[None]
        self.jac_b = 0.
        for t in self.transform_b:
            b = t(b_0)
            self.jac_b += t.log_abs_det_jacobian(b_0, b)
            b_0 = b
        return b[0]

    def get_weight_mean(self):
        w_0 = self.weight_mu.view(1, -1)
        for t in self.transform_w:
            w = t(w_0)
            w_0 = w
        return w.view(*self.shape)

    def get_bias_mean(self):
        b_0 = self.bias_mu[None]
        for t in self.transform_b:
            b = t(b_0)
            b_0 = b
        return b[0]

    def log_prob(self):
        '''
        Note, that here we are using last sampled weights and biases
        :return: logprob
        '''
        log_prob = torch.distributions.Normal(self.weight_mu, self.get_weight_sigma()).log_prob(
            self.weight).sum() - self.jac_w
        if self.use_bias:
            log_prob += torch.distributions.Normal(self.bias_mu, self.get_bias_sigma()).log_prob(
                self.bias).sum() - self.jac_b
        return log_prob


class IAFMixin:
    def get_transform_w(self):
        prod = np.prod(self.shape)
        transform_w = nn.ModuleList(
            [AffineAutoregressive(AutoRegressiveNN(prod, [3 * prod]), stable=True) for _ in range(self.t)])
        return transform_w

    def get_transform_b(self):
        transform_b = nn.ModuleList(
            [AffineAutoregressive(AutoRegressiveNN(self.shape[0], [3 * self.shape[0]]), stable=True) for _ in
             range(self.t)])
        return transform_b


class IAF(NF, IAFMixin):
    pass
