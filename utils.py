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

device = "cuda" if torch.cuda.is_available() else "cpu"

def train(model, 
          train_loader,
          optimiser="SGD", 
          lr=0.01, 
          epochs=1,
          weight_decay=0,
          ):

    model = model.to(device)
    loss_func = nn.CrossEntropyLoss()
    optim_dict = {"SGD": torch.optim.SGD(params=model.parameters(), lr=lr, weight_decay=weight_decay),
                  "Adam": torch.optim.Adam(params=model.parameters(), lr=lr, weight_decay=weight_decay)}
    optimiser = optim_dict[optimiser]

    all_epochs_losses = []
    all_full_train_losses = []
    all_full_test_losses = []
    
    # for each epoch
    for epoch in range(1, epochs+1):
        
        # for each mini-batch
        for X_mini_train, y_mini_train in train_loader:
            
            # training
            model.train()
            
            X_mini_train = X_mini_train.to(device)
            y_mini_train = y_mini_train.to(device)
            
            y_mini_train_pred = model(X_mini_train)
            train_loss = loss_func(y_mini_train_pred, y_mini_train)
            optimiser.zero_grad() 
            train_loss.backward() 
            optimiser.step()
            
def test(model,
         test_loader):
    
    model = model.to(device)
    loss_func = nn.CrossEntropyLoss()
    
    model.eval()
    with torch.no_grad():
                
        running_test_loss = 0
        running_test_correct = 0
                
        # find loss and accuracy for each minibatch in test
        for X_mini_test, y_mini_test in test_loader:
            X_mini_test = X_mini_test.to(device)
            y_mini_test = y_mini_test.to(device)
            # calculate loss
            y_mini_test_pred = model(X_mini_test)
            test_loss = loss_func(y_mini_test_pred, y_mini_test)
            running_test_loss += test_loss.item()
            # calculate accuracy
            y_mini_test_pred = torch.softmax(y_mini_test_pred, dim=1).argmax(dim=1)
            running_test_correct += sum(y_mini_test_pred == y_mini_test).item()

        # find the loss and accuracy for the full dataset
        test_loss = running_test_loss / len(test_loader)
        test_acc = (running_test_correct / len(test_loader.dataset)) * 100
        
        return test_loss, test_acc
    
def values_to_bytes(values):
    byte = BytesIO()
    np.save(byte, values, allow_pickle=True)
    return Parameters(tensors=byte.getvalue(), tensor_type="params")

def values_to_bytes_list(values_list):
    bytes_list = []
    for value in values_list:
        byte = BytesIO()
        np.save(byte, value, allow_pickle=True)
        bytes_list.append(byte.getvalue())
        byte.close()
    return Parameters(tensors=bytes_list, tensor_type="params")

def bytes_to_values(byte):
    values = BytesIO(byte)
    values = np.load(values, allow_pickle=True)
    return values

def bytes_to_values_list(bytes_list):
    values_list = []
    for byte in bytes_list:
        value = BytesIO(byte)
        value = np.load(value, allow_pickle=True)
        values_list.append(value)
    return values_list