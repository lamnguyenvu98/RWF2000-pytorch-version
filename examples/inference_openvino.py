import argparse
import cv2
import torch
import numpy as np
import time
from copy import deepcopy
from collections import deque

from src.utils import preprocessing
from src.data.augmentation import Normalize

from openvino.runtime import Core

parser  = argparse.ArgumentParser()
parser.add_argument('--video', '-v', required=True, type=str, help='Path to video to predict')
parser.add_argument('--save-dir', '-d', default='results/result.mp4', type=str, help='Path to write result')
parser.add_argument('--ir_model', '-c', required=True, type=str, help='Path to checkpoint')
args = parser.parse_args()

cap = cv2.VideoCapture(args.video)

# Init video writer
fps = cap.get(cv2.CAP_PROP_FPS)
size = (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
videoWriter = cv2.VideoWriter(args.save_dir, fourcc, fps, size)

core = Core()
model_ir = core.read_model(model=args.ir_model)
compiled_model_ir = core.compile_model(model=model_ir, device_name="CPU")

print("[INFO] Compiled model successful")

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
    in0 = res.transpose((3, 0, 1, 2)).astype(np.float32) # (64, 224, 224, 5) => (5, 64, 224, 224)
    in0 = np.expand_dims(in0, axis=0) # (5, 64, 224, 224) => (1, 5, 64, 224, 224)

    start = time.perf_counter()
    # Get input and output layers.
    output_layer_ir = compiled_model_ir.output(0)

    # Run inference on the input image.
    out = compiled_model_ir([in0])[output_layer_ir]
    print("Performance:", time.perf_counter() - start)
    # print("Result:", out)
    res  = out.flatten().argmax()
    pred_class = classnames[res]

    show_frame = queue[-1].copy()
    cv2.putText(show_frame, "Class: {}".format(pred_class), (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)
    videoWriter.write(show_frame)
    # cv2.imshow("Frame", show_frame)
    
    # if cv2.waitKey(1) & 0xFF == 27:
    #     break

cap.release()
videoWriter.release()
cv2.destroyAllWindows()
