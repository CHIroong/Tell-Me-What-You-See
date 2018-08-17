import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.data import sampler
import numpy as np
import torch.nn.functional as F

from spec import SpecParser, TaggedDataset
from model import create_model

dtype = torch.float32 # we will be using float throughout this tutorial
if torch.cuda.is_available():
    device = torch.device('cuda')
else:
    device = torch.device('cpu')

def check_accuracy(loader, model):    
    num_correct = 0
    num_samples = 0
    model.eval()  # set model to evaluation mode

    num_tags = 6
    matrix = []
    for i in range(num_tags):
        matrix.append([0] * num_tags)

    with torch.no_grad():        
        for x, y in loader:
            x = x.to(device=device, dtype=dtype)  # move to device, e.g. GPU
            y = y.to(device=device, dtype=torch.long)
            scores = model(x)
            _, preds = scores.max(1)
            num_correct += (preds == y).sum()
            num_samples += preds.size(0)

            y = y.data.cpu().numpy()
            preds = preds.data.cpu().numpy()
            
            for i in range(len(y)):
                matrix[y[i]][preds[i]] += 1

        acc = float(num_correct) / num_samples
        
        for line in matrix:
            for number in line:
                print("{:3d} ".format(number), end="")
            print()
        print('Got %d / %d correct (%.2f)' % (num_correct, num_samples, 100 * acc))

def train(loader_train, loader_val, model, optimizer, epochs=1):
    """
    Train a model on CIFAR-10 using the PyTorch Module API.
    
    Inputs:
    - model: A PyTorch Module giving the model to train.
    - optimizer: An Optimizer object we will use to train the model
    - epochs: (Optional) A Python integer giving the number of epochs to train for
    
    Returns: Nothing, but prints model accuracies during training.
    """
    
    torch.cuda.empty_cache()
    model = model.to(device=device)  # move the model parameters to CPU/GPU
    print_every = 100

    for e in range(epochs):
        for t, (x, y) in enumerate(loader_train):
            model.train()  # put model to training mode
            x = x.to(device=device, dtype=dtype)  # move to device, e.g. GPU
            y = y.to(device=device, dtype=torch.long)

            scores = model(x)
            loss = F.cross_entropy(scores, y)

            # Zero out all of the gradients for the variables which the optimizer
            # will update.
            optimizer.zero_grad()

            # This is the backwards pass: compute the gradient of the loss with
            # respect to each  parameter of the model.
            loss.backward()

            # Actually update the parameters of the model using the gradients
            # computed by the backwards pass.
            optimizer.step()

            if t % print_every == 0:
                print('Iteration %d, loss = %.4f' % (t, loss.item()))
                check_accuracy(loader_train, model)
                check_accuracy(loader_val, model)
                print()                

def main():
    spec = SpecParser().parse('spec.json')
    dataset = TaggedDataset(spec)
    
    # Constant to control how frequently we print train loss
    

    print('using device:', device)    
    print("total num of training sets", len(dataset))

    num_train = 3000

    loader_train = DataLoader(dataset, batch_size=64, 
                          sampler=sampler.SubsetRandomSampler(range(num_train)))

    loader_val = DataLoader(dataset, batch_size=64, 
                          sampler=sampler.SubsetRandomSampler(range(num_train, len(dataset))))

    model = create_model()
    optimizer = optim.Adam(model.parameters(), lr=1e-4)

    train(loader_train, loader_val, model, optimizer, 50)

if __name__ == "__main__":
    main()