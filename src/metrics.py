import numpy as np
import torch
import pdb
from tqdm import tqdm
from models import Dropout_layer


def log_likelihood(model, params, dataset, args):
    '''
    The function computes marginal log-likelihood via importance sampling, using args.n_IS samples.
    '''
    device = args.device
    torchType = args.torchType
    std_normal = torch.distributions.Normal(loc=torch.tensor(0., device=device, dtype=torchType),
                                   scale=torch.tensor(1., device=device, dtype=torchType),)
    log_n_IS = torch.log(torch.tensor(args.n_IS, device=device, dtype=torchType))
    nll_list = []
    with torch.no_grad():
        if args.problem == 'classification':
            if args.model_type == 'mfg':
                last_weight_mu, last_weight_logvar, last_bias_mu, last_bias_logvar = params
                for test_batch, test_label in tqdm(dataset.next_test_batch()):
                    nll_samples = torch.empty((args.n_IS, test_batch.shape[0]))
                    emb = model(test_batch)
                    for i in range(args.n_IS):
                        last_weight = last_weight_mu + std_normal.sample(last_weight_mu.shape) * torch.exp(0.5 * last_weight_logvar)
                        last_bias = last_bias_mu + std_normal.sample(last_bias_mu.shape) * torch.exp(0.5 * last_bias_logvar)
                        preds = emb @ last_weight + last_bias
                        log_likelihood = torch.distributions.Categorical(logits=preds).log_prob(test_label)
                        nll_samples[i, :] = log_likelihood + std_normal.log_prob(last_weight).sum() + std_normal.log_prob(last_bias).sum() - std_normal.log_prob((last_weight - last_weight_mu) / torch.exp(0.5 * last_weight_logvar)).sum() - std_normal.log_prob((last_bias - last_bias_mu) / torch.exp(0.5 * last_bias_logvar)).sum()
                    nll_lse = torch.logsumexp(nll_samples, dim=0)
                    nll = -log_n_IS + torch.mean(nll_lse)
                    nll_list.append(nll.cpu().detach().item())
                    
            elif args.model_type == 'mcdo':
                dropout = Dropout_layer()
                last_weight_mu, last_bias_mu = params
                for test_batch, test_label in tqdm(dataset.next_test_batch()):
                    nll_samples = torch.empty((args.n_IS, test_batch.shape[0]))
                    emb = model(test_batch)
                    for i in range(args.n_IS):
                        last_weight = last_weight_mu
                        last_bias = last_bias_mu
                        preds = dropout(emb) @ last_weight + last_bias
                        log_likelihood = torch.distributions.Categorical(logits=preds).log_prob(test_label)
                        nll_samples[i, :] = log_likelihood
                    nll_lse = torch.logsumexp(nll_samples, dim=0)
                    nll = -log_n_IS + torch.mean(nll_lse)
                    nll_list.append(nll.cpu().detach().item())
        else:
            if args.model_type == 'mfg':
                last_weight_mu, last_weight_logvar, last_bias_mu, last_bias_logvar = params
                for i in range(n_samples):
                    pass
            elif args.model_type == 'mcdo':
                last_weight_mu, last_bias_mu = params
                for i in range(n_samples):
                    pass
    return nll_list

def log_likelihood_errors(model, params, dataset, args):
    '''
    The function computes log-likelihood of errors, using normal distribution assumption
    and args.n_IS samples.
    '''
    if args.problem != 'regression':
        raise NotImplementedError
    device = args.device
    torchType = args.torchType
    std_normal = torch.distributions.Normal(loc=torch.tensor(0., device=device, dtype=torchType),
                                   scale=torch.tensor(1., device=device, dtype=torchType),)
    dropout = Dropout_layer()
    nll_list = []
    with torch.no_grad():
        if args.model_type == 'mcdo':
            last_weight_mu, last_bias_mu = params
        elif args.model_type == 'mfg':
            last_weight_mu, last_weight_logvar, last_bias_mu, last_bias_logvar = params
        else:
            raise NotImplementedError
            
        for val_batch, val_label in tqdm(dataset.next_val_batch()):
            pred_matrix = torch.empty((args.n_IS, val_batch.shape[0]), device=device, dtype=torchType)
            emb = model(val_batch)
            for i in range(args.n_IS):
                if args.model_type == 'mfg':
                    last_weight = last_weight_mu + std_normal.sample(last_weight_mu.shape) * torch.exp(0.5 * last_weight_logvar)
                    last_bias = last_bias_mu + std_normal.sample(last_bias_mu.shape) * torch.exp(0.5 * last_bias_logvar)
                    preds = emb @ last_weight + last_bias
                elif args.model_type == 'mcdo':
                    last_weight = last_weight_mu
                    last_bias = last_bias_mu
                    preds = dropout(emb) @ last_weight + last_bias
                else:
                    raise NotImplementedError
                pred_matrix[i, :] = preds.view(-1)
            means = pred_matrix.mean(0)
            stds = pred_matrix.std(0)
            log_likelihood = std_normal.log_prob((val_label.view(-1) - means) / stds)
            nll_list.append(log_likelihood.mean().cpu().numpy())
        return np.mean(nll_list)