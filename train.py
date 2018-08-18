import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.data import sampler
import numpy as np
import torch.nn.functional as F
import os

from spec import SpecParser, TaggedDataset
from model import Model32, Model64, Model96, Model128

patch_size = 96
num_workers = 4 # workers for parallel loading

dtype = torch.float32 # we will be using float throughout this tutorial
if torch.cuda.is_available():
    device = torch.device('cuda')
else:
    device = torch.device('cpu')

def check_accuracy(spec, loader, data_name, model):    
    num_correct = 0
    num_samples = 0
    model.eval()  # set model to evaluation mode

    num_tags = spec.num_tags
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
                print("{:6d} ".format(number), end="")
            print()
        print('%s Got %d / %d correct (%.2f)' % (data_name, num_correct, num_samples, 100 * acc))
    
    return acc

def train(spec, loader_train, loader_val, model, optimizer, epochs=1):    
    torch.cuda.empty_cache()
    model = model.to(device=device)  # move the model parameters to CPU/GPU
    max_val_acc = 0.75

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
        
        print('Epoch %d, loss = %.4f' % (e, loss.item()))
        check_accuracy(spec, loader_train, 'trainig', model)
        val_acc = check_accuracy(spec, loader_val, 'validation', model)
        print()                

        if val_acc > max_val_acc:
            max_val_acc = val_acc
            torch.save(model.state_dict(), os.path.join("models", "model%d.%.2f.pt" % (patch_size, val_acc * 100)))

def main():
    spec = SpecParser().parse('spec.json')
    dataset = TaggedDataset(spec)
    
    # Constant to control how frequently we print train loss
    
    print('using device:', device)    
    print("total num of training sets", len(dataset))

    num_train = int(len(dataset) * 0.97)

    if patch_size == 32:
        batch_size = 256
        model = Model32(spec.num_tags)
    elif patch_size == 64:
        batch_size = 192
        model = Model64(spec.num_tags)
    elif patch_size == 96:
        batch_size = 192
        model = Model96(spec.num_tags)
    elif patch_size == 128:
        batch_size = 64
        model = Model128(spec.num_tags)
    else:
        raise "Unknown patch_size (%d)" % (patch_size)

    loader_train = DataLoader(dataset, batch_size=batch_size, num_workers=num_workers,
                          sampler=sampler.SubsetRandomSampler(range(num_train)))

    loader_val = DataLoader(dataset, batch_size=batch_size, num_workers=num_workers, 
                          sampler=sampler.SubsetRandomSampler(range(num_train, len(dataset))))
    
    optimizer = optim.Adam(model.parameters(), lr=5e-4, weight_decay=1e-4)

    train(spec, loader_train, loader_val, model, optimizer, 500)

if __name__ == "__main__":
    main()