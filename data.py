import torch
import torchvision
from torch import nn
from torchvision.models import resnet18
from torch.utils.data import DataLoader, random_split
from fedlab.utils.dataset.partition import CIFAR10Partitioner
import json
import numpy as np
from tqdm import tqdm
from io import BytesIO
from flwr.common import Parameters
import random


def femnist_data(path_to_data_folder="femnist_data", combine_clients=20, subset=50):
    """
    Input: the path to the folder of json files.
    
    Data is downloadable from: https://mega.nz/file/XYhhSRIb#PAVgu1zGUoGUU5EzF2xCOnUmGlp5nNQAF8gPdvo_m2U
    It can also be downloaded by cloning the LEAF repository, and running the following command in the femnist folder:
    ./preprocess.sh -s niid --iu 1.0 --sf 1.0 -k 0 -t sample --smplseed 42 --spltseed 42
    
    Returns: a tuple containing the training dataloaders, and test dataloaders,
             with a dataloader for each client
    """

    all_client_trainloaders = []
    all_client_testloaders = []

    if combine_clients <= 1:
        for i in tqdm(range(0, 36)): # for each json file
            with open(f"{path_to_data_folder}/all_data_{i}.json") as file:

                # load the 100 clients in each json file
                data = json.load(file)
                all_clients = data["users"]

                for client in all_clients:
                    # load the dataset from one client
                    X_data = data["user_data"][client]["x"]
                    num_samples = len(X_data)
                    X_data = np.array(X_data, dtype=np.float32).reshape(num_samples, 1, 28, 28) # reshape into BxCxHxW
                    y_data = np.array(data["user_data"][client]["y"], dtype=np.int64)

                    # split into test and train data
                    X_train, X_test = random_split(X_data, (0.9, 0.1), torch.Generator().manual_seed(42))
                    y_train, y_test = random_split(y_data, (0.9, 0.1), torch.Generator().manual_seed(42))

                    # put the dataset into dataloaders
                    torch.manual_seed(47)
                    train_loader = DataLoader(dataset=list(zip(X_train, y_train)),
                                              batch_size=32,
                                              shuffle=True,
                                              pin_memory=True)
                    torch.manual_seed(47)
                    test_loader = DataLoader(dataset=list(zip(X_test, y_test)),
                                             batch_size=32,
                                             shuffle=True,
                                             pin_memory=True)

                    # add the dataloader to the overall list
                    all_client_trainloaders.append(train_loader)
                    all_client_testloaders.append(test_loader)
    else:
        all_clients = []

        for i in tqdm(range(0, 36)): # for each json file
            with open(f"{path_to_data_folder}/all_data_{i}.json") as file:
                
                # load the 100 clients in each json file
                data = json.load(file)
                for client in data["users"]:
                    X_data = data["user_data"][client]["x"]
                    num_samples = len(X_data)
                    X_data = np.array(X_data, dtype=np.float32).reshape(num_samples, 1, 28, 28)
                    y_data = np.array(data["user_data"][client]["y"], dtype=np.int64)

                    all_clients.append((X_data, y_data))
                    
        # group the given number of clients together
        grouped_clients = zip(*[iter(all_clients)] * combine_clients)
        for group in grouped_clients:

            # merge the data arrays together
            X_data = np.concatenate([client[0] for client in group])
            y_data = np.concatenate([client[1] for client in group])

            # split into test and train data
            X_train, X_test = random_split(X_data, (0.9, 0.1), torch.Generator().manual_seed(42))
            y_train, y_test = random_split(y_data, (0.9, 0.1), torch.Generator().manual_seed(42))

            # put the dataset into dataloaders
            torch.manual_seed(47)
            train_loader = DataLoader(dataset=list(zip(X_train, y_train)),
                                      batch_size=32,
                                      shuffle=True,
                                      pin_memory=True)
            torch.manual_seed(47)
            test_loader = DataLoader(dataset=list(zip(X_test, y_test)),
                                     batch_size=32,
                                     shuffle=True,
                                     pin_memory=True)

            # add the dataloader to the overall list
            all_client_trainloaders.append(train_loader)
            all_client_testloaders.append(test_loader)
    
    # subset the data loaders to the given number
    random.seed(47)
    subset_trainloaders = random.sample(all_client_trainloaders, subset)
    random.seed(47)
    subset_testloaders = random.sample(all_client_testloaders, subset)
    return subset_trainloaders, subset_testloaders

def cifar_data(num_clients=50, balanced_data=False):
    """
    Returns: a tuple containing the training data loaders, and test data loaders,
             with a dataloader for each client
    """
    # Download and reshape the dataset
    train_data = torchvision.datasets.CIFAR10(root="cifar_data", train=True, download=True)
    test_data = torchvision.datasets.CIFAR10(root="cifar_data", train=False, download=True)
    X_train = (train_data.data / 255).astype(np.float32).transpose(0, 3, 1, 2)
    y_train = np.array(train_data.targets, dtype=np.int64)
    X_test = (test_data.data / 255).astype(np.float32).transpose(0, 3, 1, 2)
    y_test = np.array(test_data.targets, dtype=np.int64)
    
    if balanced_data:
        balance=True
        partition="iid"
        dir_alpha=None
    else: # data not balanced
        balance=None
        partition="dirichlet"
        dir_alpha=0.3
    
    # Partition the data into an imbalanced and non-iid form
    partitioned_train_data = CIFAR10Partitioner(train_data.targets,
                                                  num_clients,
                                                  balance=balance,
                                                  partition=partition,
                                                  dir_alpha=dir_alpha,
                                                  seed=42)
    partitioned_test_data = CIFAR10Partitioner(test_data.targets,
                                               num_clients,
                                               balance=True,
                                               partition="iid",
                                               seed=42)
    
    all_client_trainloaders = []
    all_client_testloaders = []

    # Put the data onto a dataloader for each client, following the partitions
    for client in range(num_clients):
        client_X_train = X_train[partitioned_train_data[client], :, :, :]
        client_y_train = y_train[partitioned_train_data[client]]
        torch.manual_seed(47)
        train_loader = DataLoader(dataset=list(zip(client_X_train, client_y_train)),
                                  batch_size=32,
                                  shuffle=True,
                                  pin_memory=True)
        client_X_test = X_test[partitioned_test_data[client], :, :, :]
        client_y_test = y_test[partitioned_test_data[client]]
        torch.manual_seed(47)
        test_loader = DataLoader(dataset=list(zip(client_X_test, client_y_test)),
                                  batch_size=32,
                                  shuffle=True,
                                  pin_memory=True)

        all_client_trainloaders.append(train_loader)
        all_client_testloaders.append(test_loader)
        
    return all_client_trainloaders, all_client_testloaders