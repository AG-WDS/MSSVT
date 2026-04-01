import os
import cv2
import numpy as np
from torch.utils.data import Dataset


class VRLDatset(Dataset):
    
    def __init__(self, images_dir, masks_dir, transform=None):
        self.ids = sorted(os.listdir(images_dir))
        self.npys_fps = [os.path.join(images_dir, img_id) for img_id in self.ids]
        self.masks_fps = [os.path.join(masks_dir, img_id.replace(".npy", ".png")) for img_id in self.ids]
        self.transform = transform


    def __getitem__(self, idx):
        npy = np.load(self.npys_fps[idx], mmap_mode='c')
        mask = cv2.imread(self.masks_fps[idx], 0)

        if self.transform:
            augmented = self.transform(image=npy, mask=mask)
            npy, mask = augmented['image'], augmented['mask']

        npy = npy.unsqueeze(0)
        return npy, mask.long()


    def __len__(self):
        return len(self.ids)
    

    
