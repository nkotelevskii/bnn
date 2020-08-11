import torch
import torch.nn as nn
import torch.nn.functional as F
import pdb

from pyro.nn import AutoRegressiveNN, ConditionalAutoRegressiveNN
from pyro.distributions.transforms import AffineAutoregressive, NeuralAutoregressive, BlockAutoregressive



def make_positive(x):
    return F.softplus(x)

class Prior(nn.Module):
    def __init__(self, aux={}):
        super().__init__()
        if aux['prior_type'] == 'standard':
            self.prior_mu = aux['prior_mu']
            self.prior_sigma = aux['prior_sigma']

            self.prior_bias_mu = aux['prior_bias_mu']
            self.prior_bias_sigma = aux['prior_bias_sigma']
            
            self.prior_weight = torch.distributions.Normal(loc=torch.tensor(self.prior_mu, device=aux['device']),
                                                            scale=torch.tensor(self.prior_sigma, device=aux['device']))
            self.prior_bias = torch.distributions.Normal(loc=torch.tensor(self.prior_bias_mu, device=aux['device']),
                                                            scale=torch.tensor(self.prior_bias_sigma, device=aux['device']))
            self.prior_weight_sampler = lambda x: self.prior_weight.sample(x)
            self.prior_bias_sampler = lambda x: self.prior_bias.sample(x)
            
            if aux['var_family_type'] == 'standard':
                self.kl_weights = lambda w, w_0, w_mu, w_sigma: self.calculate_kl_normal(w, w_0, w_mu, w_sigma, self.prior_mu, self.prior_sigma)
                self.kl_bias = lambda b, b_0, b_mu, b_sigma: self.calculate_kl_normal(b, b_0, b_mu, b_sigma, self.prior_bias_mu, self.prior_bias_sigma)
            else:
                self.kl_weights = lambda w, w_0, w_mu, w_sigma: self.calculate_kl_general(w, w_0, w_mu, w_sigma, self.prior_mu, self.prior_sigma)
                self.kl_bias = lambda b, b_0, b_mu, b_sigma: self.calculate_kl_general(b, b_0, b_mu, b_sigma, self.prior_bias_mu, self.prior_bias_sigma)
        else:
            raise NotImplementedError('Prior type is not implemented yet') 
        
    def calculate_kl_normal(self, z, z_0, mu_q, sig_q, mu_prior, sig_prior):
        kl = 0.5 * (2 * torch.log(sig_prior / sig_q) - 1 + (sig_q / sig_prior).pow(2) + ((mu_prior - mu_q) / sig_prior).pow(2)).mean()
        return kl
    
    def calculate_kl_general(self, z, z_0, mu_q, sig_q, mu_prior, sig_prior):
        kl = (torch.distributions.Normal(mu_q, sig_q).log_prob(z_0) - torch.distributions.Normal(mu_prior, sig_prior).log_prob(z)).mean()
        return kl


class DiagPriorModule(nn.Module):
    def __init__(self, shape, bias=True, aux={}):
        super().__init__()

        aux_def = {
            'prior_type': 'standard',
            'var_family_type': 'standard',
            'device': "cpu",
            'prior_mu': 0.,
            'prior_sigma': 1.,
            'prior_bias_mu': 0.,
            'prior_bias_sigma': 1.,
        }

        aux_def.update(aux)
        aux = aux_def

        self.weight_mu = nn.Parameter(torch.empty(shape))
        self.weight_rho = nn.Parameter(torch.empty(shape))

        self.use_bias = bias
        

        if self.use_bias:
            self.bias_mu = nn.Parameter(torch.empty(shape[0]))
            self.bias_rho = nn.Parameter(torch.empty(shape[0]))
        else:
            self.register_parameter('bias_mu', None)
            self.register_parameter('bias_rho', None)

        self.prior = Prior(aux)
        self.reset_parameters()
        
        if aux['var_family_type'] == 'standard':
            self.variational_transform_w = lambda x: x
            self.variational_transform_b = lambda x: x
            self.compute_logjacobian_w = lambda x,y: 0
            self.compute_logjacobian_b = lambda x,y: 0
        elif aux['var_family_type'] == 'IAF':
#             pdb.set_trace()
            self.w_flow = AffineAutoregressive(AutoRegressiveNN(shape[1], [3 * shape[1]]), stable=True)
            self.variational_transform_w = self.w_flow
            self.b_flow = AffineAutoregressive(AutoRegressiveNN(shape[0], [3 * shape[0]]), stable=True)
            self.variational_transform_b = self.b_flow
            
            self.compute_logjacobian_w = lambda x,y: self.w_flow.log_abs_det_jacobian(x, y)
            self.compute_logjacobian_b = lambda x,y: self.b_flow.log_abs_det_jacobian(x, y)

    def reset_parameters(self):
        self.weight_mu.data = self.prior.prior_weight_sampler(self.weight_mu.shape)
        self.weight_rho.data = self.prior.prior_weight_sampler(self.weight_rho.shape)

        if self.use_bias:
            self.bias_mu.data = self.prior.prior_bias_sampler(self.bias_mu.shape)
            self.bias_rho.data = self.prior.prior_bias_sampler(self.bias_rho.shape)

    def get_weight_sigma(self):
        return make_positive(self.weight_rho)

    def get_bias_sigma(self):
        return make_positive(self.bias_rho)

    def get_kl(self):
#         pdb.set_trace()
        kl = self.prior.kl_weights(self.w, self.w_0, self.weight_mu, self.get_weight_sigma())
        if self.use_bias:
            kl += self.prior.kl_bias(self.b, self.b_0, self.bias_mu, self.get_bias_sigma())
        return (kl -self.log_jac_w - self.log_jac_b).mean()

    def func(self, input, weight, bias):
        pass

    def forward(self, input, sample=True):
        if self.training or sample:
            self.w_0 = torch.distributions.Normal(self.weight_mu, self.get_weight_sigma()).rsample()
            self.w = self.variational_transform_w(self.w_0)
            self.log_jac_w = self.compute_logjacobian_w(self.w_0, self.w)
            if self.use_bias:
                self.b_0 = torch.distributions.Normal(self.bias_mu, self.get_bias_sigma()).rsample()
                self.b = self.variational_transform_b(self.b_0)
                self.log_jac_b = self.compute_logjacobian_b(self.b_0, self.b)
            else:
                self.b = None
                self.log_jac_b = 0.
        else:
            self.w = self.variational_transform(self.weight_mu)
            self.b = self.variational_transform(self.bias_mu) if self.use_bias else None

        output = self.func(input, self.w, self.b)

        return output
    
    
    
class BBBLinear(DiagPriorModule):
    """
    Linear layer with a diagonal Normal prior on its weights. To be used with Bayes by BackProp
    (Stochastic Variational Inference). Applying the layer to an input returns a pair: output and KL-term for the loss.
    """
    def __init__(self, in_features, out_features, bias=True, aux={}):
        """
        Parameters
        ----------
        in_features
        out_features
        bias
        priors
        """

        self.in_features = in_features
        self.out_features = out_features

        shape = (out_features, in_features)

        super().__init__(shape, bias=bias, aux=aux)

    def func(self, input, weight, bias):
        return F.linear(input, weight, bias)