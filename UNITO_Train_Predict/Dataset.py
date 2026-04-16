import numpy as np
import torch
from torch.utils.data import Dataset


class dataset(Dataset):
    def __init__(self, data_dir, cache_in_memory=False):
        self.data_dir = data_dir.reset_index(drop=True)
        self.image_dir = self.data_dir.iloc[:, 0].to_list()
        self.mask_dir = self.data_dir.iloc[:, 1].to_list()
        self.cache_in_memory = cache_in_memory
        self.cached_images = []
        self.cached_masks = []

        if self.cache_in_memory:
            for img_path, mask_path in zip(self.image_dir, self.mask_dir):
                image, mask = self._load_pair(img_path, mask_path)
                self.cached_images.append(image)
                self.cached_masks.append(mask)

    def __len__(self):
        return len(self.data_dir)

    def _load_pair(self, img_path, mask_path):
        image = np.load(img_path, allow_pickle=False).astype(np.float32, copy=False)
        if image.ndim == 2:
            image = np.expand_dims(image, axis=0)

        mask = np.load(mask_path, allow_pickle=False).astype(np.float32, copy=False)

        return torch.from_numpy(image), torch.from_numpy(mask)

    def __getitem__(self, index):
        if self.cache_in_memory:
            image = self.cached_images[index]
            mask = self.cached_masks[index]
        else:
            image, mask = self._load_pair(self.image_dir[index], self.mask_dir[index])

        return image, mask, self.image_dir[index]
