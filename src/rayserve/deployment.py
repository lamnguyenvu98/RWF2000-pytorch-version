import ray
from ray import serve
from ray.serve import Application
from src.utils import preprocessing
from src.data.augmentation import Normalize

from openvino.runtime import Core, properties

# from fastapi import FastAPI

import numpy as np
import pickle
from starlette.requests import Request

MODEL_CHECKPOINT = '/home/pep/Drive/PCLOUD/Projects/RWF2000-Flow-Gated-Net/model_dir/openvino/FGN.xml'

@serve.deployment(num_replicas=1, ray_actor_options={"num_cpus": 4, "num_gpus": 0}, route_prefix='/predict')
class RWF2000_Deployment:
    def __init__(self, model_ir_path: str, device: str, num_threads: int):        
        core = Core()
        model_ir = core.read_model(model=model_ir_path)
        self.compiled_model_ir = core.compile_model(model=model_ir, 
                                                    device_name=device,
                                                    config={"INFERENCE_NUM_THREADS": num_threads})
        
        
        self.normalizer = Normalize()
        self.classnames = ['Fight', 'NonFight']
    
    def predict(self, frames: np.ndarray) -> dict:
        res = preprocessing(frames=frames, dynamic_crop=False)
        res = self.normalizer(res)
        in0 = res.transpose((3, 0, 1, 2)).astype(np.float32) # (64, 224, 224, 5) => (5, 64, 224, 224)
        in0 = np.expand_dims(in0, axis=0) # (5, 64, 224, 224) => (1, 5, 64, 224, 224)
        # Get input and output layers.
        output_layer_ir = self.compiled_model_ir.output(0)

        # Run inference on the input image.
        out = self.compiled_model_ir([in0])[output_layer_ir]
        res  = out.flatten().argmax()
        pred_class = self.classnames[res]
        return {"Pred_class": pred_class}

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
