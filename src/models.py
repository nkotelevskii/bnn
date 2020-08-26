import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pdb


class Net_classification(nn.Module):
    def __init__(self, args):
        super(Net_classification, self).__init__()
        last_features = args.last_features
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=16, kernel_size=5, stride=2, padding=2)
        self.conv2 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=5, stride=2, padding=2)
        self.conv3 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=5, stride=2, padding=2)
        self.linear1 = nn.Linear(in_features=1024, out_features=256)
        self.linear2 = nn.Linear(in_features=256, out_features=last_features)
        self.activation = nn.Softplus()
    def forward(self, x):
        h1 = torch.relu(self.conv1(x))
        h2 = torch.relu(self.conv2(h1))
        h3 = torch.relu(self.conv3(h2))
        h3_flat = h3.view(h3.shape[0], -1)
        h4 = torch.relu(self.linear1(h3_flat))
        h5 = self.activation(self.linear2(h4))
        return h5
    
    
    
class Net_regression(nn.Module):
    def __init__(self, args):
        super(Net_regression, self).__init__()
        in_features = args.in_features
        last_features = args.last_features
        self.linear1 = nn.Linear(in_features=in_features, out_features=4*in_features)
        self.linear2 = nn.Linear(in_features=4*in_features, out_features=last_features)
        self.activation = nn.Softplus()
    def forward(self, x):
        h3_flat = x
        h4 = torch.relu(self.linear1(h3_flat))
        h5 = self.activation(self.linear2(h4))
        return h5
    
    
    
class LastLayerBNN(nn.Module):
    def __init__(self, in_features, last_features=10, aux={}):
        super().__init__()

        self.layer1 = nn.Linear(in_features, 4*in_features)
        self.layer2 = nn.Linear(4*in_features, last_features)
        self.layer3 = BBBLinear(last_features, 1, aux=aux)
        self.device = aux.device
        self.std_normal = torch.distributions.Normal(loc=torch.tensor(0., dtype=torch.float32, device=self.device),
                                             scale=torch.tensor(1., dtype=torch.float32, device=self.device))

    def get_kl(self):
        return self.layer3.get_kl()

    def forward(self, input, sample=True):
        h = F.leakyrelu(self.layer1(input))
        h = F.leakyrelu(self.layer2(h))
        h = self.layer3(h, sample=sample)
        
        return h
    
class BNN(nn.Module):
    def __init__(self, in_features, last_features=10, aux={}):
        super().__init__()

        self.layer1 = BBBLinear(in_features, 4*in_features, aux=aux)
        self.layer2 = BBBLinear(4*in_features, last_features, aux=aux)
        self.layer3 = BBBLinear(last_features, 1, aux=aux)

    def get_kl(self):
        return self.layer1.get_kl() + self.layer2.get_kl() + self.layer3.get_kl()

    def forward(self, input, sample=True):
#         pdb.set_trace()
        h = F.leaky_relu(self.layer1(input, sample=sample))
        h = F.leaky_relu(self.layer2(h, sample=sample))
        h = self.layer3(h, sample=sample)
        
        return h
    
    
    
    
    
class Dropout_layer(nn.Module):
    def __init__(self,):
        super(Dropout_layer, self).__init__()
        self.dropout = nn.Dropout(p=0.5)
    def forward(self, x):
        return self.dropout(x)
    
    
class HMC_vanilla(nn.Module):
    def __init__(self, kwargs):
        super(HMC_vanilla, self).__init__()
        self.device = kwargs.device
        self.N = kwargs.N

        self.alpha_logit = torch.tensor(np.log(kwargs.alpha) - np.log(1. - kwargs.alpha), device=self.device)
        self.gamma = torch.tensor(np.log(kwargs.gamma), device=self.device)
        self.use_partialref = kwargs.use_partialref  # If false, we are using full momentum refresh
        self.use_barker = kwargs.use_barker  # If false, we are using standard MH ration, otherwise Barker ratio
        self.device_zero = torch.tensor(0., dtype=kwargs.torchType, device=self.device)
        self.device_one = torch.tensor(1., dtype=kwargs.torchType, device=self.device)
        self.uniform = torch.distributions.Uniform(low=self.device_zero,
                                                   high=self.device_one)  # distribution for transition making
        self.std_normal = torch.distributions.Normal(loc=self.device_zero, scale=self.device_one)

    def _forward_step(self, q_old, x=None, k=None, target=None, p_old=None):
        """
        The function makes forward step
        Also, this function computes log_jacobian of the transformation
        Input:
        q_old - current position
        x - data object (optional)
        k - auxiliary variable
        target - target class (optional)
        p_old - auxilary variables for some types of transitions (like momentum for HMC)
        Output:
        q_new - new position
        p_new - new momentum
        """
        gamma = torch.exp(self.gamma)
        p_flipped = -p_old
        q_old.requires_grad_(True)
        p_ = p_flipped + gamma / 2. * self.get_grad(q=q_old, target=target, x=x)  # NOTE that we are using log-density, not energy!
        q_ = q_old
        for l in range(self.N):
            q_ = q_ + gamma * p_
            if (l != self.N - 1):
                p_ = p_ + gamma * self.get_grad(q=q_, target=target, x=x)  # NOTE that we are using log-density, not energy!
        p_ = p_ + gamma / 2. * self.get_grad(q=q_, target=target, x=x)  # NOTE that we are using log-density, not energy!

        p_ = p_.detach()
        q_ = q_.detach()
        q_old.requires_grad_(False)

        return q_, p_

    def make_transition(self, q_old, p_old, target_distr, k=None, x=None, args=None, get_prior=None,
                        prior_flow=None):
        """
        Input:
        q_old - current position
        p_old - current momentum
        target_distr - target distribution
        x - data object (optional)
        k - vector for partial momentum refreshment
        args - dict of arguments
        scales - if we train scales for momentum or not
        Output:
        q_new - new position
        p_new - new momentum
        log_jac - log jacobians of transformations
        current_log_alphas - current log_alphas, corresponding to sampled decision variables
        a - decision variables (0 or +1)
        q_upd - proposal states
        """
        ### Partial momentum refresh
        alpha = torch.sigmoid(self.alpha_logit)
        if self.use_partialref:
            p_ref = p_old * alpha + torch.sqrt(1. - alpha ** 2) * self.std_normal.sample(p_old.shape)
        else:
            p_ref = self.std_normal.sample(q_old.shape)

        ############ Then we compute new points and densities ############
        q_upd, p_upd = self._forward_step(q_old=q_old, p_old=p_ref, k=k, target=target_distr, x=x)

#         pdb.set_trace()
        target_log_density_f = target_distr.log_prob(z=q_upd, x=x) + self.std_normal.log_prob(p_upd).sum(list(np.arange(1, len(p_upd.shape[1:]) + 1)))
        target_log_density_old = target_distr.log_prob(z=q_old, x=x) + self.std_normal.log_prob(p_ref).sum(list(np.arange(1, len(p_upd.shape[1:]) + 1)))

        log_t = target_log_density_f - target_log_density_old
        log_1_t = torch.logsumexp(torch.cat([torch.zeros_like(log_t).view(-1, 1),
                                             log_t.view(-1, 1)], dim=-1), dim=-1)  # log(1+t)
        if self.use_barker:
            current_log_alphas_pre = log_t - log_1_t
        else:
            current_log_alphas_pre = torch.min(self.device_zero, log_t)

        log_probs = torch.log(self.uniform.sample((q_upd.shape[0],)))
        a = torch.where(log_probs < current_log_alphas_pre, self.device_one, self.device_zero)
        
        q_new = torch.where((a == self.device_zero)[:, None], q_old, q_upd)
        p_new = torch.where((a == self.device_zero)[:, None], p_ref, p_upd)
            
        return q_new, p_new, a

    def get_grad(self, q, target, x=None):
        q_init = q.detach().requires_grad_(True)
        s = target.log_prob(x=x, z=q_init)
        grad = torch.autograd.grad(s.sum(), q_init)[0]
        return grad