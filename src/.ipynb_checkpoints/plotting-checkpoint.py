import matplotlib.pyplot as plt
from models import Dropout_layer
import torch
import numpy as np
import pdb
from tqdm import tqdm

def plot_mfg_classification(args, model, dataset, params):
    val_image_id = 9
    std_normal = torch.distributions.Normal(loc=torch.tensor(0., device=args.device, dtype=args.torchType),
                                   scale=torch.tensor(1., device=args.device, dtype=args.torchType),)
    last_weight_mu, last_weight_logvar, last_bias_mu, last_bias_logvar = params

    for batch in dataset.next_val_batch():
        test_image = batch[0][val_image_id].squeeze()
        test_label = batch[1][val_image_id]

    plt.title(f"{test_label.cpu().numpy()}")
    plt.imshow(test_image.cpu().numpy());
    
    n_samples = 100

    results = []
    with torch.no_grad():
        for _ in range(n_samples):
            emb = model(test_image[None, None, ...])
            last_weight = last_weight_mu + std_normal.sample(last_weight_mu.shape) * torch.exp(0.5 * last_weight_logvar)
            last_bias = last_bias_mu + std_normal.sample(last_bias_mu.shape) * torch.exp(0.5 * last_bias_logvar)

            logits = emb @ last_weight + last_bias
            probs = torch.softmax(logits, dim=-1)
            y_pred = torch.argmax(probs, dim=-1)
            results.append(y_pred.cpu().item())


    labels, counts = np.unique(results, return_counts=True)
    plt.bar(labels, counts, align='center')
    plt.xticks(ticks=np.arange(10))
    plt.xlim((-1, 10));
    
    n_samples = 100

    for val_batch in dataset.next_val_batch():
        val_images = val_batch[0]
        val_labels = val_batch[1]
        for i in range(val_images.shape[0]):
            test_image = val_images[i].squeeze()
            test_label = val_labels[i].squeeze()
            plt.close()
            results = []
            with torch.no_grad():
                for _ in range(n_samples):
                    emb = model(test_image[None, None, ...])
                    last_weight = last_weight_mu + std_normal.sample(last_weight_mu.shape) * torch.exp(0.5 * last_weight_logvar)
                    last_bias = last_bias_mu + std_normal.sample(last_bias_mu.shape) * torch.exp(0.5 * last_bias_logvar)

                    logits = emb @ last_weight + last_bias
                    probs = torch.softmax(logits, dim=-1)
                    y_pred = torch.argmax(probs, dim=-1)
                    results.append(y_pred.cpu().item())
            if np.unique(results).shape[0] > 1: # or np.unique(results)[0] != test_label:
                print('-' * 100)
                plt.title(f"{test_label.cpu().numpy()}")
                plt.imshow(test_image.cpu().numpy());
                plt.show()

                labels, counts = np.unique(results, return_counts=True)
                plt.bar(labels, counts, align='center')
                plt.xticks(ticks=np.arange(10))
                plt.xlim((-1, 10));
                plt.show();

def plot_mcdo_classification(args, model, dataset, params):
    val_image_id = 9
    std_normal = torch.distributions.Normal(loc=torch.tensor(0., device=args.device, dtype=args.torchType),
                                   scale=torch.tensor(1., device=args.device, dtype=args.torchType),)
    last_weight_mu, last_bias_mu = params

    for batch in dataset.next_val_batch():
        test_image = batch[0][val_image_id].squeeze()
        test_label = batch[1][val_image_id]

    plt.title(f"{test_label.cpu().numpy()}")
    plt.imshow(test_image.cpu().numpy());
    
    n_samples = 100
    dropout = Dropout_layer()
    results = []
    with torch.no_grad():
        for _ in range(n_samples):
            emb = model(test_image[None, None, ...])
            last_weight = last_weight_mu
            last_bias = last_bias_mu

            logits = dropout(emb) @ last_weight + last_bias
            probs = torch.softmax(logits, dim=-1)
            y_pred = torch.argmax(probs, dim=-1)
            results.append(y_pred.cpu().item())


    labels, counts = np.unique(results, return_counts=True)
    plt.bar(labels, counts, align='center')
    plt.xticks(ticks=np.arange(10))
    plt.xlim((-1, 10));
    
    n_samples = 100

    for val_batch in dataset.next_val_batch():
        val_images = val_batch[0]
        val_labels = val_batch[1]
        for i in range(val_images.shape[0]):
            test_image = val_images[i].squeeze()
            test_label = val_labels[i].squeeze()
            plt.close()
            results = []
            with torch.no_grad():
                for _ in range(n_samples):
                    emb = model(test_image[None, None, ...])
                    last_weight = last_weight_mu
                    last_bias = last_bias_mu

                    logits = dropout(emb) @ last_weight + last_bias
                    probs = torch.softmax(logits, dim=-1)
                    y_pred = torch.argmax(probs, dim=-1)
                    results.append(y_pred.cpu().item())
            if np.unique(results).shape[0] > 1: # or np.unique(results)[0] != test_label:
                print('-' * 100)
                plt.title(f"{test_label.cpu().numpy()}")
                plt.imshow(test_image.cpu().numpy());
                plt.show()

                labels, counts = np.unique(results, return_counts=True)
                plt.bar(labels, counts, align='center')
                plt.xticks(ticks=np.arange(10))
                plt.xlim((-1, 10));
                plt.show();
                
                
def mfg_regression_inference(args, model, dataset, params, val_id=0):

    last_weight_mu, last_weight_logvar, last_bias_mu, last_bias_logvar = params
    std_normal = torch.distributions.Normal(loc=torch.tensor(0., device=args.device, dtype=args.torchType),
                               scale=torch.tensor(1., device=args.device, dtype=args.torchType),)
    for batch in dataset.next_val_batch():
        test_image = batch[0][val_id].squeeze()
        test_label = batch[1][val_id]

    print(test_label)
    
    n_samples = 100

    results = []
    with torch.no_grad():
        for _ in range(n_samples):
            if args.problem == 'classification':
                emb = model(test_image[None, None, ...])
            else:
                emb = model(test_image[None,  ...])
            last_weight = last_weight_mu + std_normal.sample(last_weight_mu.shape) * torch.exp(0.5 * last_weight_logvar)
            last_bias = last_bias_mu + std_normal.sample(last_bias_mu.shape) * torch.exp(0.5 * last_bias_logvar)

            logits = emb @ last_weight + last_bias
            results.append(logits.cpu().item())

    print('Marginalized answer ', np.mean(results))
    print('True answer', test_label)
    print('-' * 100)
    print(f'Parameters are:')
    print('Weight:')
    print(f'Mean is \n {last_weight_mu.cpu().detach().numpy()}')
    print(f'Variance is \n {np.exp(last_weight_logvar.cpu().detach().numpy())}')
    print('-' * 100)
    print(f'Parameters are:')
    print('Bias:')
    print(f'Mean is \n {last_bias_mu.cpu().detach().numpy()}')
    print(f'Variance is \n {np.exp(last_bias_logvar.cpu().detach().numpy())}')
    
    
def mcdo_regression_inference(args, model, dataset, params, val_id=0):
    std_normal = torch.distributions.Normal(loc=torch.tensor(0., device=args.device, dtype=args.torchType),
                                   scale=torch.tensor(1., device=args.device, dtype=args.torchType),)
    last_weight_mu, last_bias_mu = params
    for batch in dataset.next_val_batch():
        test_image = batch[0][val_id].squeeze()
        test_label = batch[1][val_id]

    print(test_label)
    dropout = Dropout_layer()
    n_samples = 100

    results = []
    with torch.no_grad():
        for _ in range(n_samples):
            if args.problem == 'classification':
                emb = model(test_image[None, None, ...])
            else:
                emb = model(test_image[None,  ...])
            last_weight = last_weight_mu
            last_bias = last_bias_mu

            logits = dropout(emb) @ last_weight + last_bias
            results.append(logits.cpu().item())

    print('Marginalized answer ', np.mean(results))
    print('True answer', test_label)
    print('-' * 100)
    print(f'Parameters are:')
    print('Weight:')
    print(f'Mean is \n {last_weight_mu.cpu().detach().numpy()}')
    print('-' * 100)
    print(f'Parameters are:')
    print('Bias:')
    print(f'Mean is \n {last_bias_mu.cpu().detach().numpy()}')
    

def plot_pred_to_true(args, model, dataset, params, name=''):
    '''
    The function plots prediction values (with predicted variance) versus true ones
    '''
    
    std_normal = torch.distributions.Normal(loc=torch.tensor(0., device=args.device, dtype=args.torchType),
                               scale=torch.tensor(1., device=args.device, dtype=args.torchType),)
    device = args.device
    torchType = args.torchType
    dropout = Dropout_layer()
    
    examples = torch.tensor([], device=device, dtype=torchType)
    true_ans = torch.tensor([], device=device, dtype=torchType)
    pred_ans = torch.tensor([], device=device, dtype=torchType)
    pred_std = torch.tensor([], device=device, dtype=torchType)
    
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
#             pdb.set_trace()
                means = pred_matrix.mean(0)
                stds = pred_matrix.std(0)
            examples = torch.cat([examples, val_batch], dim=0)
            true_ans = torch.cat([true_ans, val_label.squeeze()], dim=0)
            pred_ans = torch.cat([pred_ans, means], dim=0)
            pred_std = torch.cat([pred_std, stds], dim=0)
        idx = torch.argsort(true_ans)
        examples = examples[idx].cpu().numpy()
        true_ans = true_ans[idx].cpu().numpy()
        pred_ans = pred_ans[idx].cpu().numpy()
        pred_std = pred_std[idx].cpu().numpy()
        
        plt.figure(figsize=(15, 8), dpi=300)
        plt.title(name)
        plt.fill_between(x=true_ans, y1=pred_ans+2*pred_std, y2=pred_ans-2*pred_std, alpha=0.5, label='std', )
        plt.plot(true_ans, pred_ans, '-..')
        plt.xlabel('y_true')
        plt.ylabel('y_pred')
        plt.axis('equal')
        plt.xlim(true_ans.min(), true_ans.max())
        plt.ylim(true_ans.min(), true_ans.max())
        plt.xticks(ticks=np.linspace(true_ans.min(), true_ans.max(), 10))
        plt.yticks(ticks=np.linspace(true_ans.min(), true_ans.max(), 10))
        plt.plot([true_ans.min(), true_ans.max()], [true_ans.min(), true_ans.max()], linewidth=6, c='r', label='perfect fit')
        plt.legend()
        plt.tight_layout()
        plt.savefig(f'./figs/{name}.png', format='png')
        plt.show();
        return examples, true_ans, pred_ans, pred_std