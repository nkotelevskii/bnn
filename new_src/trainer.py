from tqdm.auto import tqdm

import numpy as np
import math

import torch
import torch.nn as nn


def train_vi_regression_kl_explicit(model, prior, dataset, device, num_epochs=100, num_samples=10, report_freq=50,):
    """
    Train Pytorch model using VI with closed-form KL term.
    Parameters
    ----------
    model
    dataset
    num_epochs
    num_samples
    report_freq
    """

    params = list(model.parameters())
    optimizer = torch.optim.Adam(params, lr=1e-3)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1000, gamma=0.75)

    loss_fn = nn.MSELoss(reduction='none')
    model.train(True)

    for ep in tqdm(range(1, num_epochs + 1)):
        for batch_idx, (x, y) in enumerate(dataset.train_dataloader):
            optimizer.zero_grad()

            model_out = model(x, sample=True)
            varfamily_log_prob = model.log_prob()
            log_likelihood = torch.distributions.Normal(loc=model_out, scale=1.).log_prob(y).sum()
            prior_log_prob = 0.
            for p in model.parameters():
                prior_log_prob += prior.log_prob(p)
            kl = varfamily_log_prob - prior_log_prob
            elbo = log_likelihood - kl / len(dataset.train_dataloader)
            (-elbo).backward()
            optimizer.step()
        # scheduler.step()

        if ep % report_freq == 0:
            tqdm.write(f'Epoch {ep} | '
                       f'ELBO {elbo.cpu().detach().numpy()} | '
                       f'LL {log_likelihood.cpu().detach().numpy()} | '
                       f'KL {kl.cpu().detach().numpy() if isinstance(kl, torch.Tensor) else kl}')

            with torch.no_grad():
                se = []
                for batch_idx, (x_val_batch, y_val_batch) in enumerate(dataset.val_dataloader):
                    model_out = model(x_val_batch, sample=False)
                    y_pred = model_out
                    se.append(loss_fn(y_pred, y_val_batch).cpu().detach().numpy())
            mse = np.vstack(se).mean()
            tqdm.write(f"MSE {mse}")