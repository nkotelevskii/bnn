import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

import pdb
from tqdm import tqdm
from torchsummary import summary

from models import Net_classification, Net_regression, Dropout_layer



def train(args, dataset):
    
    args.num_classes = len(np.unique(dataset.train_ans))
    problem = 'classification' if args.num_classes < 20 else 'regression'
    if problem == 'regression':
        args.num_classes = 1
    args['problem'] = problem
    args['in_features'] = dataset.in_features[0]
    args['last_features'] = 10
    
    if args['problem'] == 'classification':
        model = Net_classification(args).to(args['device'])
        summary(model, input_size=(1, 28, 28))
    else:
        model = Net_regression(args).to(args['device'])
        summary(model, input_size=(1, args['in_features']))
        
    if args.model_type == 'mfg':
        return train_model_mfg(args, dataset, model)
    elif args.model_type == 'mcdo':
        return train_model_mcdo(args, dataset, model)
    

def train_model_mfg(args, dataset, model):
    
    device = args.device
    torchType = args.torchType
    num_epoches = args.num_epoches
    print_info = args.print_info
            
    ## Init parameters
    last_weight_mu = nn.Parameter(torch.randn((args['last_features'], args.num_classes), device=device, dtype=torchType))
    last_weight_logvar = nn.Parameter(torch.randn((args['last_features'], args.num_classes), device=device, dtype=torchType))

    last_bias_mu = nn.Parameter(torch.randn((1, args.num_classes), device=device, dtype=torchType))
    last_bias_logvar = nn.Parameter(torch.randn((1, args.num_classes), device=device, dtype=torchType))
    
    
    ## Best parameters
    best_last_weight_mu = nn.Parameter(torch.randn((args['last_features'], args.num_classes), device=device, dtype=torchType))
    best_last_weight_logvar = nn.Parameter(torch.randn((args['last_features'], args.num_classes), device=device, dtype=torchType))
    best_last_bias_mu = nn.Parameter(torch.randn((1, args.num_classes), device=device, dtype=torchType))
    best_last_bias_logvar = nn.Parameter(torch.randn((1, args.num_classes), device=device, dtype=torchType))
    best_model = type(model)(args).to(args.device) # get a new instance
    
    params = list(model.parameters()) + [last_weight_mu, last_weight_logvar] + [last_bias_mu, last_bias_logvar]
    optimizer = torch.optim.Adam(params, lr=1e-3)
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, np.linspace(start=10, stop=num_epoches, num=50), gamma=0.9)
    
    
    std_normal = torch.distributions.Normal(loc=torch.tensor(0., device=device, dtype=torchType),
                                       scale=torch.tensor(1., device=device, dtype=torchType),)
    best_mse = float('inf')
    best_KL = float('inf')
    current_tol = 0
    for ep in tqdm(range(num_epoches)):
        for x_train, y_train_labels in dataset.next_train_batch():
            emb = model(x_train)
            last_weight = last_weight_mu + std_normal.sample(last_weight_mu.shape) * torch.exp(0.5 * last_weight_logvar)
            last_bias = last_bias_mu + std_normal.sample(last_bias_mu.shape) * torch.exp(0.5 * last_bias_logvar)
            preds = emb @ last_weight + last_bias

            if args['problem'] == 'classification':
                log_likelihood = torch.distributions.Categorical(logits=preds).log_prob(y_train_labels).sum()
            else:
                log_likelihood = torch.distributions.Normal(loc=preds, scale=torch.tensor(1., device=device,
                                                                                          dtype=torchType)).log_prob(y_train_labels).sum()

            KL = (0.5 * (-last_weight_logvar + torch.exp(last_weight_logvar) + last_weight_mu ** 2 - 1.)).mean() \
                            + (0.5 * (-last_bias_logvar + torch.exp(last_bias_logvar) + last_bias_mu ** 2 - 1.)).mean()

            elbo = log_likelihood - KL
            (-elbo).backward()

            optimizer.step()
            optimizer.zero_grad()
        scheduler.step()

        if ep % print_info == 0:
            print(f'ELBO value is {elbo.cpu().detach().numpy()} on epoch number {ep}')
            score_total = []
            with torch.no_grad():
                for x_val, y_val_labels in dataset.next_val_batch():
                    emb = model(x_val)
                    last_weight = last_weight_mu
                    last_bias = last_bias_mu
                    logits = emb @ last_weight + last_bias
                    if args['problem'] == 'classification':
                        probs = torch.softmax(logits, dim=-1)
                        y_pred = torch.argmax(probs, dim=-1)
                        score = (y_pred==y_val_labels).to(torchType).cpu().mean().numpy()
                        score_total.append(score)
                    else:
                        score = ((logits - y_val_labels)**2).mean().cpu().numpy()
                        score_total.append(score)

            if args['problem'] == 'classification':
                print(f"Mean validation accuracy at epoch number {ep} is {np.array(score_total).mean()}")
            else:
                print(f"Mean validation MSE at epoch number {ep} is {np.array(score_total).mean()}")
            print(f'Current KL is {KL.cpu().detach().numpy()}')

            if (np.array(score_total).mean() < best_mse):
                best_mse = np.array(score_total).mean()
                current_tol = 0
                
                best_model.load_state_dict(model.state_dict())
                best_last_weight_mu = last_weight_mu.clone()
                best_last_weight_logvar = last_weight_logvar.clone()
                best_last_bias_mu = last_bias_mu.clone()
                best_last_bias_logvar = last_bias_logvar.clone()
                best_KL = KL.cpu().detach().numpy()
            else:
                current_tol += 1
                if (print_info * current_tol > args['early_stopping_tol']):
                    break
    
    
    return best_model, [best_last_weight_mu, best_last_weight_logvar, best_last_bias_mu, best_last_bias_logvar], [best_mse, best_KL]
    
    
def train_model_mcdo(args, dataset, model):
    
    device = args.device
    torchType = args.torchType
    num_epoches = args.num_epoches
    print_info = args.print_info
    
    ## Init parameters
    last_weight_mu = nn.Parameter(torch.randn((args['last_features'], args.num_classes), device=device, dtype=torchType))
    last_bias_mu = nn.Parameter(torch.randn((1, args.num_classes), device=device, dtype=torchType))
    dropout = Dropout_layer()
    
    ## Best parameters
    best_last_weight_mu = nn.Parameter(torch.randn((args['last_features'], args.num_classes), device=device, dtype=torchType))
    best_last_bias_mu = nn.Parameter(torch.randn((1, args.num_classes), device=device, dtype=torchType))
    best_model = type(model)(args).to(args.device) # get a new instance
    
    params = list(model.parameters()) + [last_weight_mu, last_bias_mu]
    optimizer = torch.optim.Adam(params, lr=1e-3)
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, np.linspace(start=10, stop=num_epoches, num=50), gamma=0.9)
    
    best_mse = float('inf')
    current_tol = 0
    for ep in tqdm(range(num_epoches)):
        dropout.train()
        for x_train, y_train_labels in dataset.next_train_batch():
            #####
            emb = model(x_train)
            last_weight = last_weight_mu
            last_bias = last_bias_mu
            #####
            preds = dropout(emb) @ last_weight + last_bias

            if args['problem'] == 'classification':
                log_likelihood = torch.distributions.Categorical(logits=preds).log_prob(y_train_labels).sum()
            else:
                log_likelihood = torch.distributions.Normal(loc=preds, scale=torch.tensor(1., device=device,
                                                                                          dtype=torchType)).log_prob(y_train_labels).sum()

            (-log_likelihood).backward()

            optimizer.step()
            optimizer.zero_grad()
        scheduler.step()

        if ep % print_info == 0:
            print(f'Log likelihood value is {log_likelihood.cpu().detach().numpy()} on epoch number {ep}')
            score_total = []
            dropout.eval()
            with torch.no_grad():
                for x_val, y_val_labels in dataset.next_val_batch():
                    emb = model(x_val)
                    last_weight = last_weight_mu
                    last_bias = last_bias_mu
                    logits = dropout(emb) @ last_weight + last_bias
                    if args['problem'] == 'classification':
                        probs = torch.softmax(logits, dim=-1)
                        y_pred = torch.argmax(probs, dim=-1)
                        score = (y_pred==y_val_labels).to(torchType).cpu().mean().numpy()
                        score_total.append(score)
                    else:
                        score = ((logits - y_val_labels)**2).mean().cpu().numpy()
                        score_total.append(score)

            if args['problem'] == 'classification':
                print(f"Mean validation accuracy at epoch number {ep} is {np.array(score_total).mean()}")
            else:
                print(f"Mean validation MSE at epoch number {ep} is {np.array(score_total).mean()}")
            if (np.array(score_total).mean() < best_mse):
                best_mse = np.array(score_total).mean()
                current_tol = 0
                best_model.load_state_dict(model.state_dict())
                best_last_weight_mu = last_weight_mu.clone()
                best_last_bias_mu = last_bias_mu.clone()
            else:
                current_tol += 1
                if (print_info * current_tol > args['early_stopping_tol']):
                    break
    
    return best_model, [best_last_weight_mu, best_last_bias_mu], [best_mse]
    
    