import torch
from torch import nn
import torch.nn.functional as F

def flatten(x):
    N = x.shape[0] # read in N, C, H, W
    return x.view(N, -1)  # "flatten" the C * H * W values into a single vector per image

class Flatten(nn.Module):
    def forward(self, x):
        return flatten(x)

class Model32(nn.Module):
    def __init__(self, num_classes):
        super().__init__()

        self.batch1 = nn.BatchNorm2d(3)
        self.conv1 = nn.Conv2d(3, 32, 5, padding=2)
        self.batch1_2 = nn.BatchNorm2d(32)
        self.conv1_2 = nn.Conv2d(32, 64, 5, padding=2)
        self.pool1 = nn.MaxPool2d(2, stride=2)

        self.batch2 = nn.BatchNorm2d(64)
        self.conv2 = nn.Conv2d(64, 128, 5, padding = 2)
        self.batch2_2 = nn.BatchNorm2d(128)
        self.conv2_2 = nn.Conv2d(128, 256, 5, padding = 2)
        self.pool2 = nn.MaxPool2d(2, stride=2)
        
        self.fc1 = nn.Linear(8 * 8 * 256 + 2, 1000)
        self.fc2 = nn.Linear(1000, num_classes)

        nn.init.kaiming_normal_(self.conv1.weight)
        nn.init.kaiming_normal_(self.conv2.weight)
        nn.init.kaiming_normal_(self.conv1_2.weight)
        nn.init.kaiming_normal_(self.conv2_2.weight)
        nn.init.kaiming_normal_(self.fc1.weight)
        nn.init.kaiming_normal_(self.fc2.weight)
    
    def forward(self, image):

        scores = image[:, :3, :, :].contiguous()
        pos = image[:, 3, 0, :2]

        scores = self.pool1(F.relu(self.conv1_2(self.batch1_2(F.relu(self.conv1(self.batch1(scores)))))))

        scores = self.pool2(F.relu(self.conv2_2(self.batch2_2(F.relu(self.conv2(self.batch2(scores)))))))
        
        scores = self.fc2(F.relu(self.fc1(torch.cat((flatten(scores), pos), 1))))

        return scores

class Model64(nn.Module):
    def __init__(self, num_classes):
        super().__init__()

        self.batch1 = nn.BatchNorm2d(3)
        self.conv1 = nn.Conv2d(3, 16, 3, padding=1)
        self.drop1 = nn.Dropout2d()

        self.batch1_2 = nn.BatchNorm2d(16)
        self.conv1_2 = nn.Conv2d(8, 8, 3, padding=1)

        self.pool1 = nn.MaxPool2d(2, stride=2)

        self.batch2 = nn.BatchNorm2d(16)
        self.conv2 = nn.Conv2d(16, 32, 3, padding = 1)
        self.drop2 = nn.Dropout2d()

        self.batch2_2 = nn.BatchNorm2d(32)
        self.conv2_2 = nn.Conv2d(32, 32, 3, padding = 1)

        self.pool2 = nn.MaxPool2d(2, stride=2)
        
        self.fc1 = nn.Linear(16 * 16 * 32 + 4, 1000)
        self.fc2 = nn.Linear(1000, num_classes)

        nn.init.kaiming_normal_(self.conv1.weight)
        nn.init.kaiming_normal_(self.conv2.weight)
        nn.init.kaiming_normal_(self.conv1_2.weight)
        nn.init.kaiming_normal_(self.conv2_2.weight)
        nn.init.kaiming_normal_(self.fc1.weight)
        nn.init.kaiming_normal_(self.fc2.weight)
    
    def forward(self, image):
        scores = image[:, :3, :, :].contiguous()
        pos = image[:, 3, 0, :4]

        #scores = self.pool1(F.relu(self.conv1_2(self.batch1_2(F.relu(self.conv1(self.batch1(scores)))))))
        #scores = self.pool2(F.relu(self.conv2_2(self.batch2_2(F.relu(self.conv2(self.batch2(scores)))))))

        scores = self.pool1(F.relu(self.drop1(self.conv1(self.batch1(scores)))))
        scores = self.pool2(F.relu(self.drop2(self.conv2(self.batch2(scores)))))

        #scores = self.pool1(F.relu(self.conv1(self.batch1(scores))))
        
        scores = self.fc2(F.relu(self.fc1(torch.cat((flatten(scores), pos), 1))))

        return scores

# Epoch 10, 76.88/70.91 8/16 conv_size=3, dropout,
# max_val=75.23 Epoch 12 77.70/73.07 8/16 conv_Size=3, dropout, left+top
# max_val=76.09 Epoch 13 84.89/73.19 16/32 conv_size=3, dropout, left+top+horizontal_from_center
# max_val=77.22 Epoch 16/32 conv_size=3, dropout, left+top+horizontal_from_center
# max_val=78.52 Epoch 21 89.91/74.72 16/32 conv_Size=3, dropout, left+top+horizontal_from_center, weight_decay=1e-3

class Model96(nn.Module):
    def __init__(self, num_classes):
        super().__init__()

        self.conv1 = nn.Conv2d(3, 16, 3, padding=1)
        self.batch1 = nn.BatchNorm2d(16)       
        self.drop1 = nn.Dropout2d()

        self.batch1_2 = nn.BatchNorm2d(16)
        self.conv1_2 = nn.Conv2d(8, 16, 3, padding=1)

        self.pool1 = nn.MaxPool2d(2, stride=2)

        self.conv2 = nn.Conv2d(16, 32, 3, padding = 1)
        self.batch2 = nn.BatchNorm2d(32)        
        self.drop2 = nn.Dropout2d()

        self.batch2_2 = nn.BatchNorm2d(32)
        self.conv2_2 = nn.Conv2d(32, 32, 3, padding = 1)

        self.pool2 = nn.MaxPool2d(2, stride=2)
        
        self.fc1 = nn.Linear(24 * 24 * 32 + 10, 1000)
        self.fc2 = nn.Linear(1000, num_classes)

        nn.init.kaiming_normal_(self.conv1.weight)
        nn.init.kaiming_normal_(self.conv2.weight)
        nn.init.kaiming_normal_(self.conv1_2.weight)
        nn.init.kaiming_normal_(self.conv2_2.weight)
        nn.init.kaiming_normal_(self.fc1.weight)
        nn.init.kaiming_normal_(self.fc2.weight)
    
    def forward(self, image):

        scores = image[:, :3, :, :].contiguous()
        pos = image[:, 3, 0, :10]

        #scores = self.pool1(F.relu(self.conv1_2(self.batch1_2(F.relu(self.conv1(self.batch1(scores)))))))
        #scores = self.pool2(F.relu(self.conv2_2(self.batch2_2(F.relu(self.conv2(self.batch2(scores)))))))

        scores = self.pool1(F.relu(self.drop1(self.batch1(self.conv1(scores)))))
        scores = self.pool2(F.relu(self.drop2(self.batch2(self.conv2(scores)))))

        #scores = self.pool1(F.relu(self.conv1(self.batch1(scores))))
        
        scores = self.fc2(F.relu(self.fc1(torch.cat((flatten(scores), pos), 1))))

        return scores

class Model128(nn.Module):
    def __init__(self, num_classes):
        super().__init__()

        self.batch1 = nn.BatchNorm2d(3)
        self.conv1 = nn.Conv2d(3, 16, 3, padding=1)
        self.drop1 = nn.Dropout2d()

        self.batch1_2 = nn.BatchNorm2d(16)
        self.conv1_2 = nn.Conv2d(8, 16, 3, padding=1)

        self.pool1 = nn.MaxPool2d(2, stride=2)

        self.batch2 = nn.BatchNorm2d(16)
        self.conv2 = nn.Conv2d(16, 32, 3, padding = 1)
        self.drop2 = nn.Dropout2d()

        self.batch2_2 = nn.BatchNorm2d(32)
        self.conv2_2 = nn.Conv2d(32, 32, 3, padding = 1)

        self.pool2 = nn.MaxPool2d(2, stride=2)
        
        self.fc1 = nn.Linear(32 * 32 * 32 + 4, 1000)
        self.fc2 = nn.Linear(1000, num_classes)

        nn.init.kaiming_normal_(self.conv1.weight)
        nn.init.kaiming_normal_(self.conv2.weight)
        nn.init.kaiming_normal_(self.conv1_2.weight)
        nn.init.kaiming_normal_(self.conv2_2.weight)
        nn.init.kaiming_normal_(self.fc1.weight)
        nn.init.kaiming_normal_(self.fc2.weight)
    
    def forward(self, image):

        scores = image[:, :3, :, :].contiguous()
        pos = image[:, 3, 0, :4]

        #scores = self.pool1(F.relu(self.conv1_2(self.batch1_2(F.relu(self.conv1(self.batch1(scores)))))))
        #scores = self.pool2(F.relu(self.conv2_2(self.batch2_2(F.relu(self.conv2(self.batch2(scores)))))))

        scores = self.pool1(F.relu(self.drop1(self.conv1(self.batch1(scores)))))
        scores = self.pool2(F.relu(self.drop2(self.conv2(self.batch2(scores)))))

        #scores = self.pool1(F.relu(self.conv1(self.batch1(scores))))
        
        scores = self.fc2(F.relu(self.fc1(torch.cat((flatten(scores), pos), 1))))

        return scores

# conv size = 5

# epoch 5, iter 200, 94.88/71.15 32/64/128/256
# epoch 5, iter 200, 94.54/75.04 32/32/64/64
# epoch 4, iter 200, 93.38/72.50 16/16/32/32
# epoch 4, iter 200, 93.83/74.29 16/32
# epoch 5, iter 200, 93.80/74.74 8/16

# conv size = 3

# epoch 5, iter 200, 94.88/75.19 8/16
# epoch 5. iter 200, 85.27/70.85 8

# epoch 4, iter 200, 73.75/70.7 8/16 (dropout)
# max_val: 74.14, epoch 10, iter 200, 92/69 16/32 (dropout) conv_size = 5
# max_val: 76.23, epoch 12, 87.44/73.69 16/32 (dropout) conv_size = 3
# max_val: 75.34, epoch 13, 95.12/73.99 8/16 (dropout) conv_size = 3