import os
from PIL import Image
from torch.utils.data import Dataset
import torchvision.transforms as T


class PairedImageDataset(Dataset):
    

    def __init__(self, input_dir, target_dir=None, resize=(256, 256), ext=(".png", ".jpg", ".jpeg")):
        self.input_dir = input_dir
        self.target_dir = target_dir
        self.resize = resize

        # Collect input images
        self.input_files = sorted(
            [f for f in os.listdir(input_dir) if f.lower().endswith(ext)]
        )
        if len(self.input_files) == 0:
            raise ValueError(f"No image files found in input directory: {input_dir}")

        # Collect target images 
        if target_dir:
            self.target_files = sorted(
                [f for f in os.listdir(target_dir) if f.lower().endswith(ext)]
            )
            if len(self.input_files) != len(self.target_files):
                print(f"Warning: Input/Target image counts differ "
                      f"({len(self.input_files)} vs {len(self.target_files)}). "
                      f"Proceeding with min count.")
                n = min(len(self.input_files), len(self.target_files))
                self.input_files = self.input_files[:n]
                self.target_files = self.target_files[:n]
        else:
            self.target_files = None

        # Define transform and color jitter
        self.transform = T.Compose([
            T.Resize(resize),
            T.ColorJitter(
              brightness=0.1,  
              contrast=0.1,
              saturation=0.1,
              hue=0.02
          ),
            T.ToTensor(),
        ])

    def __len__(self):
        return len(self.input_files)

    def __getitem__(self, idx):
        
        in_path = os.path.join(self.input_dir, self.input_files[idx])
        x = Image.open(in_path).convert("RGB")
        x = self.transform(x)

        
        if self.target_files:
            tar_path = os.path.join(self.target_dir, self.target_files[idx])
            y = Image.open(tar_path).convert("RGB")
            y = self.transform(y)
            return x, y
        else:
            return x