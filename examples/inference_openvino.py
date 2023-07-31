import argparse
import cv2
import torch
import numpy as np
import time
from copy import deepcopy
from collections import deque

from src.utils import preprocessing
from src.data.augmentation import Normalize

import openvino.runtime.opset11 as ov
from openvino.runtime import Core, Type, Output, Layout, layout_helpers
from openvino.preprocess import PrePostProcessor, ColorFormat, ResizeAlgorithm
from openvino.runtime.utils.decorators import custom_preprocess_function
import numpy as np

INT_MAX = np.iinfo(np.int32).max

def mean(node, axes: list, keep_dims: bool = True):
    return ov.reduce_mean(node, reduction_axes=axes, keep_dims=keep_dims)

def slicing(node, start: list, stop: list, step: list):
    start_ind = ov.constant(start, dtype=np.int32)
    stop_ind = ov.constant(stop, dtype=np.int32)
    step_ind = ov.constant(step, dtype=np.int32)
    return ov.slice(node, start=start_ind, stop=stop_ind, step=step_ind)

def std(node, mean, axes):
    square_diff = ov.squared_difference(node, mean)
    # ax = ov.constant(axes, dtype=np.int32)
    return ov.sqrt(ov.reduce_mean(square_diff, reduction_axes=axes, keep_dims=True))

def normalize(node, mean, std):
    return ov.divide(ov.subtract(node, mean), std)

@custom_preprocess_function
def custom_normalizer(output: Output):
    rgb_slice = slicing(output, start=[0,0,0,0,0], stop=[INT_MAX,3,64,224,224], step=[1,1,1,1,1])
    flow_slice = slicing(output, start=[0,3,0,0,0], stop=[INT_MAX,5,64,224,224], step=[1,1,1,1,1])

    mean_rgb = mean(rgb_slice, axes=[1, 2, 3, 4], keep_dims=True)
    mean_flow = mean(flow_slice, axes=[1, 2, 3, 4], keep_dims=True)
    
    std_rgb = std(rgb_slice, mean_rgb, axes=[1, 2, 3, 4])
    std_flow = std(flow_slice, mean_flow, axes=[1, 2, 3, 4])


    norm_rgb = normalize(rgb_slice, mean_rgb, std_rgb)
    norm_flow = normalize(flow_slice, mean_flow, std_flow)

    return ov.concat([norm_rgb, norm_flow], axis=1)

@custom_preprocess_function
def custom_softmax(output: Output):
    return ov.softmax(output, axis=1)

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
model = core.read_model(model=args.ir_model)
model.reshape([-1, 5, 64, 224, 224])

ppp = PrePostProcessor(model)

ppp.input().tensor().set_element_type(Type.f32).set_shape([-1, 64, 224, 224, 5]).set_layout(Layout('NDHWC'))
ppp.input().preprocess().convert_layout([0, 4, 1, 2, 3])
ppp.input().preprocess().custom(custom_normalizer)
ppp.output().postprocess().custom(custom_softmax)
model = ppp.build()

model = core.compile_model(model, device_name="CPU")

print("[INFO] Compiled model successful")

# normalizer = Normalize()

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

    start = time.perf_counter()
    res = deepcopy(queue)
    in0 = preprocessing(frames=res, dynamic_crop=False)

    # in0 = normalizer(in0)
    # in0 = in0.transpose((3, 0, 1, 2)).astype(np.float32) # (64, 224, 224, 5) => (5, 64, 224, 224)
    in0 = np.expand_dims(in0, axis=0) # (5, 64, 224, 224) => (1, 5, 64, 224, 224)

    
    # Get input and output layers.
    # output_layer_ir = model.output(0)

    # Run inference on the input image.
    out = model([in0])["output"]
    
    # print("Result:", out)
    best_id  = out.flatten().argmax()
    pred_class = classnames[best_id]
    score = out.flatten()[best_id]

    print("Performance:", time.perf_counter() - start)
    print("Class: {} | Score: {:.2f}".format(pred_class, score))
    
    show_frame = queue[-1].copy()
    cv2.putText(show_frame, "Class: {} - Score: {:.2f}".format(pred_class, score), (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0,255,0), 1)
    videoWriter.write(show_frame)
    # cv2.imshow("Frame", show_frame)
    
    # if cv2.waitKey(1) & 0xFF == 27:
    #     break

cap.release()
videoWriter.release()
cv2.destroyAllWindows()
