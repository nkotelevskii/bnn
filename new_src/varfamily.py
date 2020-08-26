import torch
import torch.nn as nn
import torch.nn.functional as F

from pyro.nn import AutoRegressiveNN, ConditionalAutoRegressiveNN
from pyro.distributions.transforms import AffineAutoregressive, NeuralAutoregressive, BlockAutoregressive


def make_positive(x):
    return torch.exp(x)
    # return F.softplus(x)


# ### Then define transform functions (separately for weights and for biases)
# self.weight_transform = self.weight_transform(args)  ## identity, nn.Modulelist or nn.Sequential
# self.bias_transform = self.bias_transform(args)  ## identity, nn.Modulelist or nn.Sequential

# def weight_transform(self, args):
#     '''
#
#     :param args: input arguments
#     :return: transform function for weights
#     '''
#     pass
#
# def bias_transform(self, args):
#     '''
#
#     :param args: input arguments
#     :return: transform function for biases
#     '''
#     pass


class VF(nn.Module):
    def __init__(self, shape, use_bias):
        super().__init__()
        self.shape = shape
        self.use_bias = use_bias

        self.weight = None
        if self.use_bias:
            self.bias = None
        self.log_prob = None

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



class NF(MeanField):
    def __init__(self, shape, use_bias, t=1):
        super().__init__(shape, use_bias)

        self.t = t

        self.transform_w = None
        self.transform_b = None

    def get_weight_sample(self):
        w_0 = torch.distributions.Normal(self.weight_mu, self.get_weight_sigma()).rsample()
        jac = 0.
        for t in self.transform_w:
            w = t(w_0)
            jac += t.log_abs_det_jacobian(w_0, w)
            w_0 = w
        return w, jac

    def get_bias_sample(self):
        b_0 = torch.distributions.Normal(self.bias_mu, self.get_bias_sigma()).rsample()
        jac = 0.
        for t in self.transform_b:
            b = t(b_0)
            jac += t.log_abs_det_jacobian(b_0, b)
            b_0 = b
        return b, jac

    def get_weight_mean(self):
        return self.weight_mu

    def get_bias_mean(self):
        return self.bias_mu


class IAFMixin:
    def __init__(self):
        self.transform_w = nn.ModuleList([AffineAutoregressive(AutoRegressiveNN(self.shape[1], [3 * self.shape[1]]), stable=True) for _ in range(self.t)])
        self.transform_b = nn.ModuleList([AffineAutoregressive(AutoRegressiveNN(self.shape[0], [3 * self.shape[0]]), stable=True) for _ in range(self.t)])


class IAF(NF, IAFMixin):
    pass
