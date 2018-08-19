import torch
import numpy as np
from model import Model32, Model64, Model96, Model128
import os
from skimage import io

class PatchClassifier():
    def __init__(self, patch_size=96, num_tags=6, use_gpu=True):
        self.patch_size = patch_size
        self.num_tags = num_tags        
        
        self.model = {32: Model32, 64: Model64, 96: Model96, 128: Model128}[patch_size](num_tags)
        self.use_gpu = use_gpu
        if self.use_gpu and torch.cuda.is_available():
            self.device = torch.device('cuda')
        else:
            self.device = torch.device('cpu')

    def load(self, path):
        if self.use_gpu:
            self.model.load_state_dict(torch.load(path))
        else:
            self.model.load_state_dict(torch.load(path, map_location='cpu'))
        self.model.to(self.device)
        self.model.eval()
    
    def classify_one(self, image):
        image = np.expand_dims(image, axis=0)
        image = torch.from_numpy(image).to(self.device, dtype=torch.float32)
        
        scores = self.model(image).data.cpu().numpy()
        
        return np.exp(scores) / np.sum(np.exp(scores))
        
    def classify_all(self, images):
        images = torch.from_numpy(images).to(self.device, dtype=torch.float32)

        scores = self.model(images).data.cpu().numpy()
        return np.exp(scores) / np.sum(np.exp(scores), axis=1, keepdims=True)


if __name__ == "__main__":
    classifier = PatchClassifier()

    classifier.load('models/model96.81.83.pt')

    # load an image
    image = np.moveaxis(io.imread("sample.png"), -1, 0) # pull the rgb axis to the front

    # setup extra data
    image[3, 0, 0] = 0 # normalized left (0 - 100)
    image[3, 0, 1] = 30 # normalized top (0 - 100)
    image[3, 0, 2] = 100 # normalized min distance from either left or right edges (0 - 100), e.g., abs(50 - norm_left) * 2
    image[3, 0, 3] = 40 # normalized min distance from either top or bottom edges (0 - 100), e.g., abs(50 - norm_top) * 2

    scores = classifier.classify_one(image)
    print(scores)

    image2 = np.moveaxis(io.imread("sample2.png"), -1, 0) # pull the rgb axis to the front

    # setup extra data
    image2[3, 0, 0] = 0 # normalized left (0 - 100)
    image2[3, 0, 1] = 30 # normalized top (0 - 100)
    image2[3, 0, 2] = 100 # normalized min distance from either left or right edges (0 - 100), e.g., abs(50 - norm_left) * 2
    image2[3, 0, 3] = 40 # normalized min distance from either top or bottom edges (0 - 100), e.g., abs(50 - norm_top) * 2

    images = np.stack([image, image2], axis=0)

    scores = classifier.classify_all(images)
    print(scores)
    