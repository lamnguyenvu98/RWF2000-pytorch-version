import imageio.v3 as iio
import numpy as np
import cv2
from src.utils import getOpticalFlow, dynamic_crop

def uniform_sampling(video, target_frames=64):
    # get total frames of input video and calculate sampling interval 
    len_frames = int(len(video))
    interval = int(np.ceil(len_frames / target_frames))
    # init empty list for sampled video and 
    sampled_video = []
    for i in range(0,len_frames,interval):
      sampled_video.append(video[i])     
    # calculate numer of padded frames and fix it 
    num_pad = target_frames - len(sampled_video)
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


test_video = "/home/pep/drive/PCLOUD/Dataset/RWF2000_Dataset/RWF-2000/train/Fight/3kpviz7lAMY_4.avi"


video = iio.imread(test_video, index=None, extension=".avi")

samp_video = uniform_sampling(video)

v = []
for i in range(len(samp_video)):
    frame_d = cv2.resize(samp_video[i].copy(), (224, 224), interpolation=cv2.INTER_AREA)
    # frame_d = cv2.cvtColor(frame_d, cv2.COLOR_BGR2RGB)
    # frame_d = np.reshape(frame_d, (224,224, 3))
    v.append(frame_d)

v = np.array(v)
flows = getOpticalFlow(v)

result = np.zeros((len(flows), 224, 224, 5))
result[...,:3] = v
result[...,3:] = flows

cropped = dynamic_crop(result)

for i in range(len(cropped)):
    img = cropped[i, ..., :3].astype(np.uint8)
    cv2.imshow("Crop", img)
    if cv2.waitKey(0) == 27: break

cv2.destroyAllWindows()
