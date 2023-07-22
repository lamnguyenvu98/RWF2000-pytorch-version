import onnx
from onnx import numpy_helper
import onnxruntime as ort
from collections import deque
import cv2
import numpy as np
from copy import deepcopy

from src.utils import preprocessing
from src.data.augmentation import Normalize

onnx_model_file_path = 'model_dir/fgn.onnx'

model_proto = onnx.load(onnx_model_file_path)
onnx.checker.check_model(model_proto)
model_proto_bytes = onnx._serialize(model_proto)

ort_sess = ort.InferenceSession(
        model_proto_bytes,
        providers=["CPUExecutionProvider", "CUDAExecutionProvider"])

input_nodes = ort_sess.get_inputs()
input_names = [node.name for node in input_nodes]
input_shapes = [node.shape for node in input_nodes]
input_types = [node.type for node in input_nodes]
output_nodes = ort_sess.get_outputs()
output_names = [node.name for node in output_nodes]
output_shapes = [node.shape for node in output_nodes]
output_types = [node.type for node in output_nodes]

normalize = Normalize()
queue = deque(maxlen=65)

classnames = ['Fight', 'NonFight']

cap = cv2.VideoCapture("videos/_q5Nwh4Z6ao_3.avi")

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
    res = res[None, ...].transpose(0, 4, 1, 2, 3).astype(np.float32)
    
    output_tensors = ort_sess.run(
        output_names=output_names,
        input_feed={input_names[0]: res},
        run_options=None)
    
    pred_id = np.argmax(output_tensors[0][0])
    print("Prediction: ", classnames[pred_id])
    show_frame = queue[-1].copy()
    cv2.imshow("Frame", show_frame)
    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()
