import albumentations as A
from albumentations.pytorch import ToTensorV2


def get_VRLDataset_transforms(phase='train'):
    if phase == 'train':
        return A.Compose([
            ToTensorV2(),
        ])
    else:
        return A.Compose([
            ToTensorV2()
        ])
        
