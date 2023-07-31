import ray
from ray import serve
from ray.serve import Application
from src.utils import preprocessing
from src.data.augmentation import Normalize

import openvino.runtime.opset11 as ov
from openvino.runtime import Core, Type, Output, Layout, layout_helpers
from openvino.preprocess import PrePostProcessor, ColorFormat, ResizeAlgorithm
from openvino.runtime.utils.decorators import custom_preprocess_function
import numpy as np

# from fastapi import FastAPI

import numpy as np
import pickle
from starlette.requests import Request

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

@serve.deployment(num_replicas=1, ray_actor_options={"num_cpus": 4, "num_gpus": 0}, route_prefix='/predict')
class RWF2000_Deployment:
    def __init__(self, model_ir_path: str, device: str, num_threads: int):        
        core = Core()
        model = core.read_model(model=model_ir_path)

        model.reshape([-1, 5, 64, 224, 224])

        ppp = PrePostProcessor(model)

        ppp.input().tensor().set_element_type(Type.f32).set_shape([-1, 64, 224, 224, 5]).set_layout(Layout('NDHWC'))
        ppp.input().preprocess().convert_layout([0, 4, 1, 2, 3])
        ppp.input().preprocess().custom(custom_normalizer)
        ppp.output().postprocess().custom(custom_softmax)
        model = ppp.build()

        self.model = core.compile_model(model, device_name="CPU", config={"INFERENCE_NUM_THREADS": num_threads})
        
        self.classnames = ['Fight', 'NonFight']
    
    def predict(self, frames: np.ndarray) -> dict:
        in0 = preprocessing(frames=frames, dynamic_crop=False)
        in0 = np.expand_dims(in0, axis=0) # (5, 64, 224, 224) => (1, 5, 64, 224, 224)

        # Run inference on the input image.
        out = self.model([in0])["output"]
        best  = out.argmax(axis=1)[0]
        pred_class = self.classnames[best]
        scores = out[:, best]
        return {"Pred_class": pred_class, "Scores:": "{:.2f}".format(scores[0])}

    async def __call__(self, http_request: Request):
        data = await http_request.body()
        frames: np.ndarray = pickle.loads(data)
        result = self.predict(frames)
        return result

# def main():
def app_builder(args: dict[str, str]) -> Application:
     return RWF2000_Deployment.bind(args["model_ir_path"], args["device"], args["num_threads"])

# app = RWF2000_Deployment.bind(model_ir_path="model_dir/openvino/FGN.xml", device="CPU", num_threads=4)

# app = RWF2000_Deployment.bind()
# if __name__ == '__main__':
#     main()
