import torch
from torch.utils.data import TensorDataset, DataLoader
from alpaca.dataloader.builder import build_dataset
import torchvision.datasets as datasets


class Dataset():
    def __init__(self, args):
        self.dataset_name = args['dataset_name']
        
        val_dataset = args['val_dataset']
        self.n_IS = args['n_IS']
        train_batch_size = args['train_batch_size']
        val_batch_size = args['val_batch_size']
        test_batch_size = args['test_batch_size']
        
        torchType = args['torchType']
        device = args['device']

        try:
            dataset = build_dataset(self.dataset_name, val_size=val_dataset)
        except TypeError:
            dataset = build_dataset(self.dataset_name, val_split=val_dataset)
        x_train, y_train = dataset.dataset('train')
        print(f'Train data shape {x_train.shape[0]}')
        self.train_ans = y_train
        self.in_features = x_train.shape[1:]
        x_val, y_val = dataset.dataset('val')
        
        if self.dataset_name in ['mnist', 'fashion_mnist']:
            x_train /= x_train.max()
            x_val /= x_val.max()
            x_shape = (-1, 1, 28, 28)
        else:
            x_shape = (-1, *x_train.shape[1:])
            
        train = TensorDataset(torch.tensor(x_train.reshape(x_shape), dtype=torchType, device=device), torch.tensor(y_train, dtype=torchType, device=device))
        validation = TensorDataset(torch.tensor(x_val.reshape(x_shape), dtype=torchType, device=device), torch.tensor(y_val, dtype=torchType, device=device))
        self.train_dataloader = DataLoader(train, batch_size=train_batch_size)
        self.val_dataloader = DataLoader(validation, batch_size=val_batch_size)
        
        self.x_train = x_train
        self.y_train = y_train
        
        self.x_val = x_val
        self.y_val = y_val
        
        if self.dataset_name.find('mnist') > -1:
            test = datasets.MNIST(root=f'./data/{self.dataset_name}', download=True, train=False)
            data_test = test.test_data.type(torchType).to(device)
            labels_test = test.test_labels.type(torchType).to(device)

            self.test = data_test.data
            self.test_labels = labels_test.data

            test_data = []
            for i in range(self.test.shape[0]):
                test_data.append([self.test[i], self.test_labels[i]])
            self.test_dataloader = torch.utils.data.DataLoader(test_data, batch_size=test_batch_size, shuffle=False)
    
    def next_train_batch(self):
        for train_batch in self.train_dataloader:
            batch = train_batch[0]
            labels = train_batch[1]
            if self.dataset_name in ['mnist', 'fashion_mnist']:
                batch = torch.distributions.Binomial(probs=batch).sample()
            yield batch, labels

    def next_val_batch(self):
        for val_batch in self.val_dataloader:
            batch = val_batch[0]
            labels = val_batch[1]
            yield batch, labels

    def next_test_batch(self):
        for test_batch in self.test_dataloader:
            batch = test_batch[0]
            labels = test_batch[1]
            if self.dataset_name in ['mnist', 'fashion_mnist']:
                batch = torch.distributions.Binomial(probs=batch).sample()
                batch = batch.view([-1, 1, 28, 28])
#                 batch = batch.repeat(self.n_IS, 1, 1, 1)
#                 labels = labels.repeat(self.n_IS, 1)
            yield batch, labels