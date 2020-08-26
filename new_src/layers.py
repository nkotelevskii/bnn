import torch.nn as nn
import torch.nn.functional as F
from varfamily import MeanField


class BaseLayer(nn.Module):
    """
    Linear layer with a diagonal Normal prior on its weights. To be used with Bayes by BackProp
    (Stochastic Variational Inference). Applying the layer to an input returns a pair: output and KL-term for the loss.
    """

    def __init__(self, shape, vf_class=MeanField, bias=True, **kwargs):
        """
        Parameters
        ----------
        in_features
        out_features
        bias
        priors
        """
        super().__init__()
        self.shape = shape
        self.vf = vf_class(shape, bias, **kwargs)

    def func(self, input, weight, bias):
        pass

    def log_prob(self,):
        return self.vf.log_prob()

    def forward(self, x, sample=True):
        weight, bias = self.vf(sample)
        output = self.func(x, weight, bias)
        return output



class BLinear(BaseLayer):
    def __init__(self, in_features, out_features, vf_class=MeanField, bias=True, **kwargs):
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

        super().__init__(shape, bias=bias, vf_class=vf_class, **kwargs)


    def func(self, input, weight, bias):
        return F.linear(input, weight, bias)
