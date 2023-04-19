from copy import deepcopy
import cv2
import numpy as np
from model import FGN
from utils import preprocessing
import torch
from torchvision import transforms
from dataset.augmentation import Normalize, ToTensor
import argparse
from collections import deque

parser  = argparse.ArgumentParser()
parser.add_argument('--video', '-v', required=True, type=str, help='Path to video to predict')
parser.add_argument('--dir', '-d', default='results/result.mp4', type=str, help='Path to write result')
parser.add_argument('--checkpoints', '-c', required=True, type=str, help='Path to checkpoint')
args = parser.parse_args()

cap = cv2.VideoCapture(args.video)

# Init video writer
fps = cap.get(cv2.CAP_PROP_FPS)
size = (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
videoWriter = cv2.VideoWriter(args.dir, fourcc, fps, size)

tfms = transforms.Compose([
                    Normalize(),
                    ToTensor()])

classnames = ['Fight', 'NonFight']

queue = deque(maxlen=64)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

model = FGN().load_from_checkpoint(args.checkpoints).to(device)

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
    res = preprocessing(frames=res)
    res = tfms(res)
    res = res.unsqueeze(0).permute(0, 4, 1, 2, 3).float()
    pred = model(res.to(device))
    best_idx = pred.softmax(-1).argmax(-1)

    score = pred.softmax(-1)[0][best_idx].item()
    
    label = classnames[best_idx]
    text = "{}: {:.1f}".format(label, score)
    
    show_frame = queue[-1].copy()
    cv2.putText(show_frame, text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)
    videoWriter.write(show_frame)
    
cap.release()
cv2.destroyAllWindows()
