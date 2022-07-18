import numpy as np
import torch
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
  def __init__(self, p=0.5):
    self.p = p
  def __call__(self, data):
    assert isinstance(data, np.ndarray), 'Input data should be a numpy array'
    assert len(data.shape) == 4, 'Input data should has shape of (batch, frames, H, W, C)'
    assert data.shape[-1] == 5, 'Wrong number of channel! Channel f data should be 5 (RGB + 2 Flow)'
    # range of s-component: 0-1
    # range of v component: 0-255
    if np.random.rand() < self.p:
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

class ToTensor():
  def __call__(self, data):
    assert isinstance(data, np.ndarray), 'Input data should be a numpy array'
    return torch.tensor(data.copy())


