import random
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset


class NpzDataset(torch.utils.data.Dataset):
    """NpzDataset: loads a npz file as input."""

    def __init__(self, dataset_name, partition):
        self.root_dir = Path("./datasets/")
        file_name = Path(self.root_dir, f"{dataset_name}_{partition}.npz")

        self.dataset = np.load(file_name)
        self.images = torch.Tensor(self.dataset["images"])
        self.labels = self.dataset["labels"]

    def __len__(self):
        return self.images.shape[0]

    def __getitem__(self, idx):
        images = (self.images[idx] + 1) / 2  # Normalize to [0, 1] range.
        return images, self.labels[idx]
