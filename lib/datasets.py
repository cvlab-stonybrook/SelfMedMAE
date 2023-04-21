import os
import numpy as np
from PIL import Image, ImageChops

import torch
from torchvision import transforms
from torchvision.datasets import VisionDataset

from tqdm import tqdm

class ImageListDataset(VisionDataset):
    def __init__(self, data_root, listfile, transform, gray=False, nolabel=False, multiclass=False):
        self.image_list = []
        self.label_list = []
        self.nolabel = nolabel
        with open(listfile) as f:
            lines = f.readlines()
            for line in lines:
                items = line.strip().split()
                image_path = os.path.join(data_root, items[0])
                if not nolabel:
                    if not multiclass:
                        label = int(items[1])
                    elif multiclass:
                        # label = torch.tensor(list(map(int, items[1:])), dtype=torch.float64)
                        label = list(map(float, items[1:]))
                    else:
                        raise ValueError("Line format is not right")
                self.image_list.append(image_path)
                if not nolabel:
                    self.label_list.append(label)

        self.transform = transform
        self.gray = gray

    def __len__(self):
        return len(self.image_list)

    def __getitem__(self, index):
        image = Image.open(self.image_list[index])
        if self.gray:
            image = image.convert('L')
        else:
            image = image.convert('RGB')
        image = self.transform(image)

        if not self.nolabel:
            label = self.label_list[index]
            return image, torch.tensor(label)
        else:
            return image

# if __name__ == "__main__":
#     from loader import *
#     augmentation = transforms.Compose([
#         transforms.RandomResizedCrop(224, scale=(0.2, 1.)),
#         transforms.RandomApply([
#             transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)
#         ], p=0.8),
#         transforms.RandomGrayscale(p=0.2),
#         transforms.RandomApply([GaussianBlur([.1, 2.])], p=0.5),
#         transforms.RandomHorizontalFlip(),
#         transforms.ToTensor(),
#         transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
#     ])

#     dataset = ImageThumbnailDataset(
#         data_root="/data/add_disk0/leizhou/imagenet100/images",
#         listfile="/nfs/bigcornea/add_disk0/leizhou/imagenet100/val_list.txt",
#         transform=augmentation,
#         thumbnail_size=32,
#         n_M=1,
#         n_E=4,
#         nolabel=True)

#     for images, thumbnail in dataset:
#         import pdb
#         pdb.set_trace()