from PIL import Image
from skimage.transform import resize
from torchvision.transforms import ToTensor
from torch.utils.data import Dataset
import torch
import os
import glob
import numpy as np

class CustomDataset(Dataset):
    def __init__(self, root_dir, transform=None): 
        self.transform = transform
        self.files = []
        for class_dir in os.listdir(root_dir):
            class_dir_path = os.path.join(root_dir, class_dir)
            if os.path.isdir(class_dir_path):
                jpg_files = sorted(glob.glob(os.path.join(class_dir_path, "*.jpg")))
                npy_files = sorted(glob.glob(os.path.join(class_dir_path, "*.npy")))
                self.files.extend(list(zip(jpg_files, npy_files, [int(class_dir)]*len(jpg_files))))

    def set_transform(self, transform):
        self.transform = transform

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        img_path, np_path, class_label = self.files[idx]
        img = Image.open(img_path)
        img = img.resize((128, 64))

        if self.transform:
            img = self.transform(img)

        vector = np.load(np_path)
       # vector = np.transpose(vector, (2, 0, 1))  # Change vector from HxWxC to CxHxW

        vector = resize(vector, (2, 64, 128))
        vector = torch.from_numpy(vector).float()

        #img_tensor = ToTensor()(img)
        data = torch.cat((img, vector), dim=0)

        return data, class_label

    
