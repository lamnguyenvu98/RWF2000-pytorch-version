import numpy as np
import torch
from torch.nn.functional import interpolate
import cv2

class Normalize():
  def __call__(self, data):
    assert isinstance(data, np.ndarray), 'Input data should be a numpy array'
    assert data.shape[-1] == 5, 'Wrong number of channel! Channel of data should be 5 (RGB + 2 Flow)'
    def norm(data):
      mean = np.mean(data)
      std = np.std(data)
      return (data-mean) / std
    
    data[..., :3] = norm(data[...,:3])
    data[..., 3:] = norm(data[...,3:])
    return data

class Random_Flip():
  def __init__(self, p=0.5, axis=0):
    self.p = p
    self.axis = axis
  def __call__(self, data):
    assert isinstance(data, np.ndarray), 'Input data should be a numpy array'
    s = np.random.rand()
    if s < self.p:
      data = np.flip(m=data, axis=self.axis)
    return data   

class Color_Jitter():
  def __init__(self):
    return None
  def __call__(self, data):
    assert isinstance(data, np.ndarray), 'Input data should be a numpy array'
    assert len(data.shape) == 4, 'Input data should has shape of (batch, frames, H, W, C)'
    assert data.shape[-1] == 5, 'Wrong number of channel! Channel f data should be 5 (RGB + 2 Flow)'
    # range of s-component: 0-1
    # range of v component: 0-255
    s_jitter = np.random.uniform(-0.2, 0.2)
    v_jitter = np.random.uniform(-30, 30)
    for i in range(len(data[..., :3])):
      hsv = cv2.cvtColor(data[..., :3][i], cv2.COLOR_RGB2HSV)
      s = hsv[..., 1] + s_jitter
      v = hsv[..., 2] + v_jitter
      s[s < 0] = 0
      s[s > 1] = 1
      v[v < 0] = 0
      v[v > 255] = 255
      hsv[..., 1] = s
      hsv[..., 2] = v
      data[..., :3][i] = cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)
    return data

class DynamicCrop():
  def __init__(self, size=(224, 224)):
    self.size = size

  def __call__(self, video):
      # extract layer of optical flow from video
      opt_flows = video[..., 3]
      # sum of optical flow magnitude of individual frame
      magnitude = np.sum(opt_flows, axis=0)
      # filter slight noise by threshold 
      thresh = np.mean(magnitude)
      magnitude[magnitude<thresh] = 0
      # calculate center of gravity of magnitude map and adding 0.001 to avoid empty value
      x_pdf = np.sum(magnitude, axis=1) + 0.001
      y_pdf = np.sum(magnitude, axis=0) + 0.001
      # normalize PDF of x and y so that the sum of probs = 1
      x_pdf /= np.sum(x_pdf)
      y_pdf /= np.sum(y_pdf)
      # randomly choose some candidates for x and y 
      x_points = np.random.choice(a=np.arange(224), size=10, replace=True, p=x_pdf)
      y_points = np.random.choice(a=np.arange(224), size=10, replace=True, p=y_pdf)
      # get the mean of x and y coordinates for better robustness
      x = int(np.mean(x_points))
      y = int(np.mean(y_points))
      # avoid to beyond boundaries of array
      x = max(56,min(x,167))
      y = max(56,min(y,167))
      # get cropped video
      crop = video[:,x-56:x+56,y-56:y+56,:]
      crop_result = np.zeros((crop.shape[0], *self.size, crop.shape[-1]))
      # resize back to (224, 224)
      for i in range(len(video)):
        crop_result[i] = cv2.resize(crop[i], self.size)
      return crop_result.astype(np.uint8)
      # crop_rs = interpolate(torch.tensor(crop.copy()).permute(0, 3, 1, 2), self.size)
      
      
      # return crop_rs.permute(0, 2, 3, 1).numpy()

class ToTensor():
  def __call__(self, data):
    if isinstance(data, torch.Tensor): return data
    assert isinstance(data, np.ndarray), 'Input data should be a numpy array'
    return torch.tensor(data.copy())


