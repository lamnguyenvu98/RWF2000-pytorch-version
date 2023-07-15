import cv2
import numpy as np
from torch.utils.data import Dataset

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

  def dynamic_sampling(self, video):
    gradient_frames = np.zeros((video[0], video[1], video[2]))

    # convert rgb to gray and calculate gradient
    for i in range(len(video)):
      gray = cv2.cvtColor(video[i, ..., :3], cv2.COLOR_BGR2GRAY)
      # calculate gradient
      gX = cv2.Sobel(gray, ddepth=cv2.CV_32F, dx=1, dy=0, ksize=3)
      gY = cv2.Sobel(gray, ddepth=cv2.CV_32F, dx=0, dy=1, ksize=3)
      # the gradient magnitude images are now of the floating point data
      # type, so we need to take care to convert them back a to unsigned
      # 8-bit integer representation so other OpenCV functions can operate
      # on them and visualize them
      gX = cv2.convertScaleAbs(gX)
      gY = cv2.convertScaleAbs(gY)
      # combine the gradient representations into a single image
      gradient_frames[i] = cv2.addWeighted(gX, 0.5, gY, 0.5, 0)

    # calculate absolute different
    abs_diff_frames = np.zeros_like(gradient_frames)

    # p_thresh = 100

    for i in range(0, len(gradient_frames)):
      if i == 0:
          abs_diff_frames[i] = np.zeros_like(gradient_frames[i])
      else:
          abs_diff_frames[i] = np.abs(gradient_frames[i] - gradient_frames[i - 1])

    # Compute energy
    energy = list()
    energy_thresh = 50

    for i in range(len(abs_diff_frames)):
      energy_each_frame = np.sum(abs_diff_frames[i])
      energy.append(energy_each_frame)

    energy = np.array(energy)

    max_energy = -9999999
    max_id = -1

    for i in range(len(energy) - self.target_frames + 1):
      energy_cal_window = np.sum(energy[i:i+self.target_frames])
      if max_energy < energy_cal_window:
          max_energy = energy_cal_window
          max_id = i

    return video[max_id:max_id + self.target_frames, ...]

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
    data = np.load(path, mmap_mode='r')['data']
    data = np.float32(data)
    # sampling 64 frames uniformly from the entire video
    data = self.uniform_sampling(data)
    # whether to utilize the data augmentation
    return data
