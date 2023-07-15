import argparse
import ncnn
import cv2
import time
from copy import deepcopy
from collections import deque

from utils import preprocessing
from dataset.augmentation import Normalize

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

net = ncnn.Net()
net.load_param("model_dir/fgn.param")
net.load_model("model_dir/fgn.bin")

print("[INFO] Load model successful")

normalize = Normalize()

classnames = ['Fight', 'NonFight']

queue = deque(maxlen=65)

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
    
    # mat_in = ncnn.Mat()
    ex = net.create_extractor()
    ex.input("input", res)
    _, mat_out = ex.extract("output")
    
    show_frame = queue[-1].copy()
    cv2.imshow("Frame", show_frame)
    if cv2.waitKey(1) & 0xFF == 27:
        break
    
cap.release()
cv2.destroyAllWindows()
    