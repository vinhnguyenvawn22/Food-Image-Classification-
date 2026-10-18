import os
import numpy as np
import torch
from torch.utils.data import Dataset
from PIL import Image
import cv2
from collections import Counter
class FoodyDataset(Dataset):
    def __init__(self, root, train=True, transform=None):
        if train:
            mode = "train"
        else:
            mode = "test"
        root = os.path.join(root, mode)
        
        self.category = [
            'banh_cuon', 'banh_mi', 'banh_ngot', 'banh_trang_nuong',
            'bun_bo', 'bun_cha', 'bun_rieu', 'com_tam',
            'ga_ran', 'hu_tieu', 'nem_nuong', 'pho',
            'pizza', 'tra_sua', 'vit_quay'
        ]
        self.image_path = []
        self.label = []
        self.transform = transform

        for i, category in enumerate(self.category):
            data_file_path = os.path.join(root, category)
            for file_name in os.listdir(data_file_path):
                file_path = os.path.join(data_file_path, file_name)
                self.image_path.append(file_path)
                self.label.append(i)

    def __len__(self):
        return len(self.label)

    def __getitem__(self, idx):
        image_path = self.image_path[idx]
        image = Image.open(image_path).convert("RGB")
        label = self.label[idx]
        
        if self.transform:
            image = self.transform(image)
            
        return image, label


if __name__ == "__main__":
    train_dataset = FoodyDataset("foody", train=True)
    test_dataset = FoodyDataset("foody", train=False)

    train_count = Counter(train_dataset.label)
    test_count = Counter(test_dataset.label)

    categories = train_dataset.category

    print("=== TRAIN ===")
    for i, name in enumerate(categories):
        print(f"{name}: {train_count[i]}")

    print("\n=== TEST ===")
    for i, name in enumerate(categories):
        print(f"{name}: {test_count[i]}")