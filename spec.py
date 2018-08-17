import json

class SpecParser:
    def parse(self, path):
        with open(path, 'r', encoding='utf8') as inf:
            return Spec(json.load(inf))
            
class Spec:
    def __init__(self, spec):
        self.width = spec['width']
        self.height = spec['height']
        self.data = spec["data"]
        self.tags = spec["tags"]


from torch.utils.data import Dataset, DataLoader
from skimage import io, transform
import numpy as np
import os

class TaggedDataset(Dataset):
    def __init__(self, spec):
        """
        Args:
            csv_file (string): Path to the csv file with annotations.
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.spec = spec
        num_train = 0
        
        patches = []
        self.patches = []

        for image in spec.data:
            num_train += len(image["patches"])
            
            for patch in image["patches"]:
                patch["image_id"] = image["id"]
                patch["left"] = patch["left"] / image["width"]
                patch["top"] = patch["top"] / image["height"]

            self.patches += image["patches"]

        self.num_train = num_train

    def __len__(self):
        return self.num_train

    def __getitem__(self, idx):
        patch = self.patches[idx]

        img_name = os.path.join("images", str(patch["image_id"]), patch["filename"])
        image = np.moveaxis(io.imread(img_name), -1, 0)
        image[3, 0, 0] = patch["left"] * 100
        image[3, 0, 1] = patch["top"] * 100
        image[3, 0, 2] = abs(50 - patch["left"] * 100) * 2
        image[3, 0, 3] = abs(50 - patch["top"] * 100) * 2
        tags = patch["tags"]
        return [image, np.argmax(tags)]