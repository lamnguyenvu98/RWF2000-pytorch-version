import ray
from ray import serve

from openvino.runtime import Core, Type, Layout
from openvino.preprocess import PrePostProcessor
from src.utils import custom_normalizer, custom_softmax, preprocessing

# from fastapi import FastAPI

import numpy as np
import pickle
from starlette.requests import Request

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
    
    def predict(self, in0: np.ndarray) -> list:        
        # Run inference on the input image.
        out = self.model([in0])["output"]
        best_ids  = out.argmax(axis=1)
        scores = out.max(axis=1)
        
        results = list()
        
        for i, (best, score) in enumerate(zip(best_ids, scores)):
            results.append({
                "batch_idx": i,
                "Class predict": self.classnames[best],
                "Scores": "{:.2f}".format(score)
            })
        
        return results

    async def __call__(self, http_request: Request):
        data = await http_request.body()
        frames = pickle.loads(data)
        in0 = [preprocessing(frames=frame, dynamic_crop=False) for frame in frames]
        in0 = np.stack(in0)
        result = self.predict(in0)
        return result

# def main():
def app_builder(args: dict[str, str]) -> serve.Application:
     return RWF2000_Deployment.bind(args["model_ir_path"], args["device"], args["num_threads"])

# app = RWF2000_Deployment.bind(model_ir_path="model_dir/openvino/FGN.xml", device="CPU", num_threads=4)

# app = RWF2000_Deployment.bind()
# if __name__ == '__main__':
#     main()
