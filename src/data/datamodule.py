import numpy as np
import os
import torch
from torch.utils.data import DataLoader
from torchvision import transforms
from lightning.pytorch import LightningDataModule
from src.data.augmentation import *
from src.data.dataset import RWF2000

class RWF2000DataModule(LightningDataModule):
    def __init__(self, dirpath: str, target_frames=64, batch_size=8):
        super().__init__()
        self.dirpath = dirpath
        self.target_frames = target_frames
        self.batch_size = batch_size

    def prepare_data(self):
        # Download
        self.trainpath = os.path.join(self.dirpath, 'train')
        self.valpath = os.path.join(self.dirpath, 'val')
        self.tfms = {
        "train": transforms.Compose([
                    Color_Jitter(),
                    Random_Flip(p=0.5, axis=2),
                    Normalize(),
                    ToTensor() ]),
        "val": transforms.Compose([
                    Normalize(),
                    ToTensor() ])
        }
  
    def setup(self, stage = None):
        self.trn = RWF2000(
            datapath = self.trainpath,
            tfms = self.tfms['train'],
            target_frames = self.target_frames
        )

        self.val = RWF2000(
            datapath = self.valpath,
            tfms = self.tfms['val'],
            target_frames = self.target_frames
        )

    def train_dataloader(self):
        return DataLoader(
            dataset = self.trn,
            batch_size = self.batch_size,
            num_workers = os.cpu_count(),
            shuffle = True,
            pin_memory = True,
            drop_last = True,
            collate_fn = self.trn.collate_fn
        )
    
    def val_dataloader(self):
        return DataLoader(
            dataset = self.val,
            batch_size = self.batch_size,
            num_workers = os.cpu_count(),
            shuffle = False,
            pin_memory = True,
            drop_last = False,
            collate_fn = self.val.collate_fn
        )
     
    def test_dataloader(self):
        return DataLoader(
            dataset = self.val,
            batch_size = self.batch_size,
            num_workers = os.cpu_count(),
            shuffle = True,
            pin_memory = True,
            drop_last = False,
            collate_fn = self.val.collate_fn
        )

if __name__ == '__main__':
  tfms = transforms.Compose([
                    DynamicCrop(),
                    Color_Jitter(),
                    Random_Flip(p=0.5, axis=2),
                    # Normalize(),
                    ToTensor()
                ])
  
  image = np.load('/home/pep/drive/PCLOUD/Dataset/RWF2000_Dataset/RWF2000-Build/val/Fight/39BFeYnbu-I_2.npz', mmap_mode='r')['data']
  
  result = tfms(image)
  result = result.numpy()
  for i in range(len(result)):
    im = result[i, ..., :3].astype(np.uint8)
    cv2.imshow("result", im)
    if cv2.waitKey(0) == 27: break
  
  cv2.destroyAllWindows()
  
  
