from PIL import Image
from torchvision.transforms import ToTensor
from torch.utils.data import Dataset
import os
import glob

class CustomDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.transform = transform
        self.files = []
        for class_dir in os.listdir(root_dir):
            class_dir_path = os.path.join(root_dir, class_dir)
            if os.path.isdir(class_dir_path):
                jpg_files = sorted(glob.glob(os.path.join(class_dir_path, "*.jpg")))
                self.files.extend([(jpg_file, int(class_dir)) for jpg_file in jpg_files])

    def set_transform(self, transform):
        self.transform = transform

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        img_path, class_label = self.files[idx]

        img = Image.open(img_path)
        img = img.resize((128, 64))  # Resize the image to a fixed size (width, height)
        #img = ToTensor()(img)  

        if self.transform:
            img = self.transform(img)

        return img, class_label  # Returning data and label
