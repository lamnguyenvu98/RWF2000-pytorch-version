import numpy as np
import os
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from pytorch_lightning import LightningDataModule
from augmentation import *

class RWF2000(Dataset):
  def __init__(self, datapath, tfms, target_frames):
    super(RWF2000, self).__init__()
    self.tfms = tfms
    # Load all the save_path of files, and create a dictionary that save the pair of "data:label"
    self.X_path, self.Y_dict = self.search_data(datapath)
    self.target_frames = target_frames
    return None

  def __len__(self):
    return len(self.X_path)

  def __getitem__(self, idx):
    X = self.load_data(self.X_path[idx])
    assert X.shape == (64, 224, 224, 5), 'File "{}" has wrong shape'.format(self.X_path[idx])
    class_name = self.Y_dict[self.X_path[idx]]
    y = int(class_name == 'NonFight')
    return X, y

  def collate_fn(self, batch):
    X, y = [], []
    for Xi, yi in batch:
      Xi = self.tfms(Xi)
      X.append(Xi)
      y.append(yi)
    X = torch.stack(X).permute(0, 4, 1, 2, 3).float() # (batch, num_frames, H, W, C) => (batch, C, num_frames, H, W)
    y = torch.tensor(y).long()
    return X, y

  def uniform_sampling(self, video):
    # get total frames of input video and calculate sampling interval 
    len_frames = int(len(video))
    interval = int(np.ceil(len_frames/self.target_frames))
    # init empty list for sampled video and 
    sampled_video = []
    for i in range(0,len_frames,interval):
      sampled_video.append(video[i])     
    # calculate numer of padded frames and fix it 
    num_pad = self.target_frames - len(sampled_video)
    padding = []
    if num_pad>0:
      for i in range(-num_pad,0):
        try: 
          padding.append(video[i])
        except:
          padding.append(video[0])
      sampled_video += padding     
    # get sampled video
    return np.array(sampled_video, dtype=np.float32)

  def search_data(self, datapath):
    X_path = []
    Y_dict = {}
    # list all kinds of sub-folders
    self.dir = sorted(os.listdir(datapath)) # Fight, NonFight
    for i,folder in enumerate(self.dir):
        folder_path = os.path.join(datapath, folder)
        for file_ in os.listdir(folder_path):
            file_path = os.path.join(folder_path,file_)
            # append the each file path, and keep its label  
            X_path.append(file_path)
            Y_dict[file_path] = self.dir[i]
    return X_path, Y_dict

  def print_stats(self):
    # calculate basic information
    self.n_files = len(self.X_path)
    self.n_classes = len(self.dir)
    self.indexes = np.arange(len(self.X_path))
    np.random.shuffle(self.indexes)
    # Output states
    print("Found {} files belonging to {} classes.".format(self.n_files, self.n_classes))
    for i, label in enumerate(self.dir):
        print('%10s : '%(label), i)
    return None

  def load_data(self, path):
    # load the processed .npy files which have 5 channels (1-3 for RGB, 4-5 for optical flows)
    data = np.load(path, mmap_mode='r')
    data = np.float32(data)
    # sampling 64 frames uniformly from the entire video
    data = self.uniform_sampling(data)
    # whether to utilize the data augmentation
    return data


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
                    Color_Jitter(p=0.5),
                    Random_Flip(p=0.5, axis=2),
                    DynamicCrop(),
                    Normalize(),
                    ToTensor() ]),
        "val": transforms.Compose([
                    DynamicCrop(),
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