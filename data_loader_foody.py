from create_dataset import FoodyDataset
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.transforms import ToTensor, Resize,Compose
import numpy as np

if __name__ == "__main__":
    root = "foody"
    transform = Compose([
        Resize((200, 200)),  
        ToTensor(),          
    ])
    tranning_data = FoodyDataset(root, train=True, transform= transform)
    image, label = tranning_data.__getitem__(123)

    tranning_dataloader = DataLoader(
        dataset= tranning_data,
        batch_size= 4,
        num_workers= 4,
        shuffle= True,
        drop_last= True
    )

    for images, labels in tranning_dataloader:
        print(images.shape)
        print(labels)