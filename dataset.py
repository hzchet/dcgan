import os
import io

import numpy as np
from PIL import Image
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader


class PixelDataset(Dataset):
    def __init__(
        self,
        root: str,
        mode: str
    ):
        path_to_data = f'{root}/pixel_dataset_{mode}.npy'
        assert os.path.exists(path_to_data)
        self.items = np.load(path_to_data, allow_pickle=True)
        self.transform = transforms.Compose([
            transforms.Resize(64),
            transforms.CenterCrop(64),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ])

    def __len__(self):
        return len(self.items)
    
    def __getitem__(self, ind: int):
        image_data = self.items[ind, 0]['bytes']
        image = Image.open(io.BytesIO(image_data)).convert('RGB')
        
        return self.transform(image)
