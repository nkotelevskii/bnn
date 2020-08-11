import pandas as pd
import torch
import numpy as np
import random
import matplotlib.pyplot as plt

import sys
sys.path.insert(0, './src')
from data import Dataset
from training import train
from plotting import plot_mfg_classification, plot_mcdo_classification, mfg_regression_inference, mcdo_regression_inference, plot_pred_to_true
from metrics import log_likelihood, log_likelihood_errors

from sklearn.svm import SVR
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import mean_squared_error


def set_seeds(rand_seed):
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.manual_seed(rand_seed)
    np.random.seed(rand_seed)
    random.seed(rand_seed)
    torch.cuda.manual_seed_all(rand_seed)

set_seeds(1337)


class dotdict(dict):
    """dot.notation access to dictionary attributes"""
    __getattr__ = dict.get
    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__
    
    
args = dotdict({})
args['device'] = 'cuda:0' if torch.cuda.is_available() else 'cpu'
args['torchType'] = torch.float32

df_dict = {
    'dataset': [],
    'model': [],
    'errors_loglikelihood': [],
    'val_mse': [],
    'kl': [],
          }
# , 'protein_structure', 'yacht_hydrodynamics', 'year_prediction_msd'
for dataname in ['boston_housing', 'concrete', 'energy_efficiency', 'kin8nm', 'naval_propulsion', 'ccpp', 'protein_structure', 'yacht_hydrodynamics', 'year_prediction_msd']:
    args['dataset_name'] = dataname
    if dataname in ['boston_housing']:
        args['early_stopping_tol'] = 1000
    else:
        args['early_stopping_tol'] = 400
    if args['dataset_name'].find('mnist') > -1:
        args['num_epoches'] = 201
        args['print_info'] = 50
        args['n_IS'] = 10000

        args['train_batch_size'] = 100
        args['val_dataset'] = 10000
        args['val_batch_size'] = 100
        args['test_batch_size'] = 100
    else:
        args['n_IS'] = 5000
        args['num_epoches'] = 15001
        args['print_info'] = 50
        if dataname in ['year_prediction_msd']:
            args['train_batch_size'] = 200
        elif dataname in ['concrete']:
            args['train_batch_size'] = 50
        else:
            args['train_batch_size'] = 100
        args['val_dataset'] = 200
        args['val_batch_size'] = 50
        args['test_batch_size'] = 10
        
    dataset = Dataset(args)
    print(f'Dataset {dataname} has {dataset.in_features} input features')
    
    ## Train models
    for model_type in ['mfg', 'mcdo']: # 'mcdo', 'mfg'
        args['model_type'] = model_type
        model, params, globals()[f'list_{model_type}']  = train(args, dataset)
        globals()[f'{model_type}_ll'] = log_likelihood_errors(model, params, dataset, args)
        likelihood = globals()[f'{model_type}_ll']
        print(f'{model_type} errors loglikelihood is {likelihood}')
        examples, true_ans, pred_ans, pred_std = plot_pred_to_true(args, model, dataset, params, name=f'{dataname}_{model_type}')
        
    if args.problem == 'regression':
        svr = SVR()
        gridcv = GridSearchCV(svr,
                             param_grid={
                                'C': [0.01, 0.1, 1., 10.],
                                 'epsilon': [0.01, 0.1, 1],
                                 'gamma': ['auto', 'scale'],
                              },
                              cv=5,
                              iid=True,
                             )
        gridcv.fit(dataset.x_train, dataset.y_train.ravel())
        print(f'SVR\'s MSE validation score: {mean_squared_error(dataset.y_val, gridcv.predict(dataset.x_val))}')
        
    sk_pred = gridcv.predict(examples)

    plt.figure(figsize=(15, 8), dpi=300)
    plt.title('Sklearn model')
    plt.plot(true_ans, sk_pred, '-..')
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
    plt.show();
    plt.savefig(f'./figs/{dataname}_svr.png', format='png')
    
    df_dict['dataset'].append(dataname)
    df_dict['model'].append('mfg')
    df_dict['errors_loglikelihood'].append(mfg_ll)
    df_dict['val_mse'].append(list_mfg[0])
    df_dict['kl'].append(list_mfg[1])
    
    df_dict['dataset'].append(dataname)
    df_dict['model'].append('mcdo')
    df_dict['errors_loglikelihood'].append(mcdo_ll)
    df_dict['val_mse'].append(list_mcdo[0])
    df_dict['kl'].append("None")
    
    df_dict['dataset'].append(dataname)
    df_dict['model'].append('svr')
    df_dict['errors_loglikelihood'].append("None")
    df_dict['val_mse'].append(mean_squared_error(dataset.y_val, gridcv.predict(dataset.x_val)))
    df_dict['kl'].append("None")
    
df = pd.DataFrame(df_dict)
df.to_csv('./result_table.csv', index=0)

    
    
    
    
    
    
    
        