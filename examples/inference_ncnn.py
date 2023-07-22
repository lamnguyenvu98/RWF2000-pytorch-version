import argparse
import ncnn
import cv2
import torch
import numpy as np
import time
from copy import deepcopy
from collections import deque

from src.utils import preprocessing
from src.data.augmentation import Normalize

# parser  = argparse.ArgumentParser()
# parser.add_argument('--video', '-v', required=True, type=str, help='Path to video to predict')
# parser.add_argument('--savedir', '-d', default='results/result.mp4', type=str, help='Path to write result')
# parser.add_argument('--parampath', '-p', required=True, type=str, help='Path to model param')
# parser.add_argument('--binarypath', '-b', required=True, type=str, help='Path to model binary')
# args = parser.parse_args()

cap = cv2.VideoCapture("videos/_q5Nwh4Z6ao_3.avi")

# Init video writer
# fps = cap.get(cv2.CAP_PROP_FPS)
# size = (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))
# fourcc = cv2.VideoWriter_fourcc(*'mp4v')
# videoWriter = cv2.VideoWriter(args.savedir, fourcc, fps, size)

# tfms = transforms.Compose([
#     Normalize(),
#     ToTensor()
# ])

normalize = Normalize()

classnames = ['Fight', 'NonFight']

queue = deque(maxlen=65)

net = ncnn.Net()
net.load_param("ncnn_models/model_jit.ncnn.param")
net.load_model("ncnn_models/model_jit.ncnn.bin")
net.opt.num_threads = 4

print("[INFO] Load model successful")

while True:
    ret, frame = cap.read()
    if not ret: break
    if len(queue) <= 0: # At initialization, populate queue with initial frame
        for i in range(64):
            queue.append(frame)

    # Add the read frame to last and pop out the oldest one
    queue.append(frame)
    queue.popleft()

    res = deepcopy(queue)
    res = preprocessing(frames=res, dynamic_crop=False)
    res = normalize(res)
    in0 = res.transpose((3, 0, 1, 2)).astype(np.float32) # (64, 224, 224, 5) => (5, 64, 224, 224)
    print("Input shape:",in0.shape)
    mat_in = ncnn.Mat(in0).clone()
    # in0 = res.unsqueeze(0).permute(0, 4, 1, 2, 3).float()
    
    with net.create_extractor() as ex:
        ex.input("in0", mat_in)
        _, out0 = ex.extract("out0")
        out = torch.from_numpy(np.array(out0)).unsqueeze(0).softmax(-1)
    
        print("Out result:", out)
        
    # show_frame = queue[-1].copy()
    # cv2.imshow("Frame", show_frame)
    # if cv2.waitKey(1) == 27:
    #     break
    
cap.release()
# cv2.destroyAllWindows()
    