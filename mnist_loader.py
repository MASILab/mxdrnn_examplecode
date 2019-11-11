import torch
from torchvision import datasets, transforms
from torch.utils.data.sampler import SubsetRandomSampler
from rnn_dataset import RMNIST, MNIST, MMNIST
import numpy as np

def get_train_valid_loader(data_dir,
                           batch_size,
                           model_name,
                           time, 
                           random_seed = 1234,
                           augment=False,
                           valid_size=0.1,
                           shuffle=True,
                           show_sample=False,
                           num_workers=4,
                           pin_memory=True):
    """
    Utility function for loading and returning train and valid 
    multi-process iterators over the MNIST dataset. A sample 
    9x9 grid of the images can be optionally displayed.
    If using CUDA, num_workers should be set to 1 and pin_memory to True.
    Params
    ------
    - data_dir: path directory to the dataset.
    - batch_size: how many samples per batch to load.
    - augment: whether to apply the data augmentation scheme
      mentioned in the paper. Only applied on the train split.
    - random_seed: fix seed for reproducibility.
    - valid_size: percentage split of the training set used for
      the validation set. Should be a float in the range [0, 1].
    - shuffle: whether to shuffle the train/validation indices.
    - show_sample: plot 9x9 sample grid of the dataset.
    - num_workers: number of subprocesses to use when loading the dataset.
    - pin_memory: whether to copy tensors into CUDA pinned memory. Set it to
      True if using GPU.
    Returns
    -------
    - train_loader: training set iterator.
    - valid_loader: validation set iterator.
    """
    error_msg = "[!] valid_size should be in the range [0, 1]."
    assert ((valid_size >= 0) and (valid_size <= 1)), error_msg

    normalize = transforms.Normalize((0.1307,), (0.3081,))  # MNIST

    # define transforms
    valid_transform = transforms.Compose([
            transforms.ToTensor(),
            normalize
        ])
    if augment:
        train_transform = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize
        ])
    else:
        train_transform = transforms.Compose([
            transforms.ToTensor(),
            normalize
        ])

    # load the dataset
    if model_name == 'bl':
        print ('bl loader')
        train_dataset = datasets.MNIST(root=data_dir, train=True, 
                    download=True, transform=train_transform)

        valid_dataset = datasets.MNIST(root=data_dir, train=True, 
                    download=True, transform=valid_transform)
    if model_name == 'multi':
        print ('multi channel loader')
        train_dataset = MMNIST(root = data_dir, time = time, train=True, 
                       transform=train_transform)
        valid_dataset = MMNIST(root = data_dir, time = time, train=True, 
                       transform=valid_transform)    
        
    if model_name == 'xdrnn' or model_name == 'mxdrnn':
        print ('(m)xdrnn loader')
        train_dataset = RMNIST(root = data_dir, time = time, train=True, 
                       transform=train_transform)
        valid_dataset = RMNIST(root = data_dir, time = time, train=True, 
                       transform=valid_transform)

    num_train = len(train_dataset)
    indices = list(range(num_train))
    split = int(np.floor(valid_size * num_train))

    if shuffle == True:
        np.random.seed(random_seed)
        np.random.shuffle(indices)

    train_idx, valid_idx = indices[split:], indices[:split]

    train_sampler = SubsetRandomSampler(train_idx)
    valid_sampler = SubsetRandomSampler(valid_idx)

    train_loader = torch.utils.data.DataLoader(train_dataset, 
                    batch_size=batch_size, sampler=train_sampler, 
                    num_workers=num_workers, pin_memory=pin_memory)

    valid_loader = torch.utils.data.DataLoader(valid_dataset, 
                    batch_size=batch_size, sampler=valid_sampler, 
                    num_workers=num_workers, pin_memory=pin_memory)

    return (train_loader, valid_loader) 

def get_test_loader(data_dir, 
                    batch_size,
                    model_name, 
                    time, 
                    shuffle=False,
                    num_workers=4,
                    pin_memory=True):
    """
    Utility function for loading and returning a multi-process 
    test iterator over the MNIST dataset.
    If using CUDA, num_workers should be set to 1 and pin_memory to True.
    Params
    ------
    - data_dir: path directory to the dataset.
    - batch_size: how many samples per batch to load.
    - shuffle: whether to shuffle the dataset after every epoch.
    - num_workers: number of subprocesses to use when loading the dataset.
    - pin_memory: whether to copy tensors into CUDA pinned memory. Set it to
      True if using GPU.
    Returns
    -------
    - data_loader: test set iterator.
    """
    normalize = transforms.Normalize((0.1307,), (0.3081,))  # MNIST

    # define transform
    transform = transforms.Compose([
        transforms.ToTensor(),
        normalize
    ])
    if model_name == 'bl':
        dataset = datasets.MNIST(root=data_dir, 
                                   train=False, 
                                   download=True,
                                   transform=transform)
    if model_name == 'xdrnn' or model_name == 'mxdrnn':
        dataset = RMNIST(root = data_dir, time = time, train=False, 
                       transform=transform)
    
    if model_name == 'multi':
        dataset = MMNIST(root = data_dir, time = time, train=False, 
                       transform=transform)

    data_loader = torch.utils.data.DataLoader(dataset, 
                                              batch_size=batch_size, 
                                              shuffle=shuffle, 
                                              num_workers=num_workers,
                                              pin_memory=pin_memory)

    return data_loader
