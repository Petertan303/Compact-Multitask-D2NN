import os
import numpy as np
import torch
from torch.utils.data import Dataset
from torch.utils.data import Subset
from torchvision import datasets
from torchvision import transforms
from PIL import Image

from d2nn import config

class SyntheticShapes(Dataset):
    CLASS_NAMES = [
        "Cross", "Triangle", "H-Triangle", "Square", "H-Square",
        "Pentagon", "H-Pentagon", "Dumbbell", "Star", "H-Star",
    ]

    def __init__(self, root: str, train: bool):
        self.train = train
        split = "train" if train else "test"
        self.data_dir = os.path.join(root, f"synthetic_shapes_{split}")
        if not os.path.exists(self.data_dir):
            raise FileNotFoundError(
                f"SyntheticShapes not found at {self.data_dir}. "
                f"Generate it with: python -m d2nn.shapes"
            )

        self.images = []
        self.targets = []
        for label in range(10):
            class_dir = os.path.join(self.data_dir, str(label))
            if not os.path.isdir(class_dir):
                continue
            for fname in sorted(os.listdir(class_dir)):
                if not fname.endswith(".png"):
                    continue
                img = self._read_png(os.path.join(class_dir, fname))
                self.images.append(img)
                self.targets.append(label)
        self.images = np.stack(self.images).astype(np.float32)
        self.targets = np.array(self.targets, dtype=np.int64)

    @staticmethod
    def _read_png(fp) -> np.ndarray:
        img = Image.open(fp).convert("L")
        return np.array(img, dtype=np.float32) / 255.0

    def __len__(self):
        return len(self.targets)

    def __getitem__(self, idx):
        img = torch.from_numpy(self.images[idx]).unsqueeze(0)
        return img, int(self.targets[idx])

def _emnist_rotate_tx():
    return transforms.Compose([
        transforms.Lambda(lambda img: img.transpose(Image.ROTATE_90)),
        transforms.ToTensor(),
    ])

def load_datasets(train: bool) -> dict:
    transform = transforms.ToTensor()
    emnist_tx = _emnist_rotate_tx()

    emnist = datasets.EMNIST(
        root=config.DATA_PATH, split="letters", train=train,
        transform=emnist_tx, download=True,
    )
    targets_list = list(emnist.targets)
    sw_indices = [i for i, lab in enumerate(targets_list) if 1 <= int(lab) <= 10]
    sw_dataset = Subset(emnist, sw_indices)

    return {
        "NW": datasets.MNIST(root=config.DATA_PATH, train=train, transform=transform, download=True),
        "NE": datasets.FashionMNIST(root=config.DATA_PATH, train=train, transform=transform, download=True),
        "SW": sw_dataset,
        "SE": SyntheticShapes(root=config.DATA_PATH, train=train),
    }
