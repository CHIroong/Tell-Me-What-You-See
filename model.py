import torch
from torch import nn

def flatten(x):
    N = x.shape[0] # read in N, C, H, W
    return x.view(N, -1)  # "flatten" the C * H * W values into a single vector per image

class Flatten(nn.Module):
    def forward(self, x):
        return flatten(x)

def create_model():
    conv1 = nn.Conv2d(3, 32, 11, padding=5)
    #conv1_1 = nn.Conv2d(32, 32, 11, padding=5)
    pool1 = nn.MaxPool2d(2, stride=2)    

    conv2 = nn.Conv2d(32, 64, 9, padding=4)
    #conv2_1 = nn.Conv2d(64, 64, 9, padding=4)
    pool2 = nn.MaxPool2d(2, stride=2)

    conv3 = nn.Conv2d(64, 128, 7, padding=3)
    #conv3_1 = nn.Conv2d(128, 128, 7, padding=3)
    pool3 = nn.MaxPool2d(2, stride=2)

    conv4 = nn.Conv2d(128, 256, 5, padding=2)
    #conv4_1 = nn.Conv2d(256, 256, 5, padding=2)
    pool4 = nn.MaxPool2d(2, stride=2)

    conv5 = nn.Conv2d(256, 512, 3, padding=1)
    #conv5_1 = nn.Conv2d(512, 512, 3, padding=1)
    pool5 = nn.MaxPool2d(2, stride=2)

    fc1 = nn.Linear(8192, 1000)
    fc2 = nn.Linear(1000, 6)
            
    nn.init.kaiming_normal_(conv1.weight)
    nn.init.kaiming_normal_(conv2.weight)
    nn.init.kaiming_normal_(conv3.weight)
    nn.init.kaiming_normal_(conv4.weight)
    nn.init.kaiming_normal_(conv5.weight)
    #nn.init.kaiming_normal_(conv1_1.weight)
    #nn.init.kaiming_normal_(conv2_1.weight)
    #nn.init.kaiming_normal_(conv3_1.weight)
    #nn.init.kaiming_normal_(conv4_1.weight)
    #nn.init.kaiming_normal_(conv5_1.weight)

    nn.init.kaiming_normal_(fc1.weight)
    nn.init.kaiming_normal_(fc2.weight)

    model = nn.Sequential(
        nn.BatchNorm2d(3),
        conv1,
        nn.ReLU(),
        #conv1_1,
        #nn.ReLU(),
        pool1,
        nn.BatchNorm2d(32),
        conv2,
        nn.ReLU(),
        #conv2_1,
        #nn.ReLU(),
        pool2,
        nn.BatchNorm2d(64),
        conv3,
        nn.ReLU(),
        #conv3_1,
        #nn.ReLU(),
        pool3, 
        nn.BatchNorm2d(128),
        conv4,
        nn.ReLU(),
        #conv4_1,
        #nn.ReLU(),
        pool4, 
        nn.BatchNorm2d(256),
        conv5,
        nn.ReLU(),
        #conv5_1,
        #nn.ReLU(),
        pool5, 
        nn.BatchNorm2d(512),
        Flatten(),
        fc1,
        nn.ReLU(),
        fc2
    )

    return model