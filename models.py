import torch
from torch import nn
from torchvision.models import resnet18

def create_model(dataset_name, model_name):
    """
    Input: Dataset name: can be 'femnist' or 'cifar'
    Input: Model name: can be 'resnet18', 'CNNRes4M3M', 'CNNRes3M2M' or 'CNN500k'
    The number correspond roughly to the parameteters - in CNNRes4M3M, for cifar it has 4 million parameters,
    for femnist it has 3 million parameters
    """
    if dataset_name=="femnist":
        num_channels=1
        image_size=28
        num_classes=62
    elif dataset_name=="cifar":
        num_channels=3
        image_size=32
        num_classes=10
    else:
        return None
    
    if model_name=="resnet18":
        model = resnet18(num_classes=num_classes) 
        model.conv1 = nn.Conv2d(num_channels, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
        return model
        
    model_dict = {"CNNRes4M3M": CNNRes4M3M,
                  "CNNRes3M2M": CNNRes3M2M,
                  "CNN500k": CNN500k,
                 }
    if model_name not in model_dict:
        return None
    
    torch.manual_seed(47)
    return model_dict[model_name](*[num_channels, image_size, num_classes])

class CNNRes4M3M(nn.Module):
    def __init__(self, num_channels, image_size, num_classes):
        super().__init__()
        
        self.conv1 = nn.Sequential(
            nn.Conv2d(num_channels, 32, kernel_size=3, padding=1),
            nn.LeakyReLU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(),
            nn.MaxPool2d(2, 2),
        )
        self.res1 = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(),
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(),
            nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(),
            nn.MaxPool2d(2, 2),
        )
        self.res2 = nn.Sequential(
            nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(),
            nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(),
        )
        self.pool = nn.MaxPool2d(2, 2)
        
        self.classifier = nn.Sequential( 
            nn.Flatten(),
            nn.Linear(256 * int(image_size/8) * int(image_size/8), 512),
            nn.LeakyReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, num_classes)
        )

    def forward(self, x):
        x = self.conv1(x)
        x = self.res1(x) + x
        x = self.conv2(x)
        x = self.res2(x) + x
        x = self.pool(x)
        x = self.classifier(x)
        return x
    
class CNNRes3M2M(nn.Module):
    def __init__(self, num_channels, image_size, num_classes):
        super().__init__()
        
        self.conv1 = nn.Sequential(
            nn.Conv2d(num_channels, 32, kernel_size=3, padding=1),
            nn.LeakyReLU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(),
            nn.MaxPool2d(2, 2),
        )
        self.res1 = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(),
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(),
            nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(),
            nn.MaxPool2d(2, 2),
        )
        self.res2 = nn.Sequential(
            nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(),
        )
        self.pool = nn.MaxPool2d(2, 2)
        
        self.classifier = nn.Sequential( 
            nn.Flatten(),
            nn.Linear(256 * int(image_size/8) * int(image_size/8), 512),
            nn.LeakyReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, num_classes)
        )

    def forward(self, x):
        x = self.conv1(x)
        x = self.res1(x) + x
        x = self.conv2(x)
        x = self.res2(x) + x
        x = self.pool(x)
        x = self.classifier(x)
        return x
    
class CNN500k(nn.Module):
    def __init__(self, num_channels, image_size, num_classes):
        super().__init__()
        
        self.layer_stack = nn.Sequential(
            nn.Conv2d(num_channels, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            
            nn.Conv2d(64, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
        
            nn.Flatten(),
            nn.Linear(32 * int(image_size/8) * int(image_size/8), 512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(256, num_classes)
        )

    def forward(self, x):
        return self.layer_stack(x)