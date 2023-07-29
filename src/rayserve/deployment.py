from ray import serve
from src.utils import preprocessing
from src.data.augmentation import Normalize

from openvino.runtime import Core, properties

import numpy as np
import pickle
from starlette.requests import Request

MODEL_CHECKPOINT = 'model_dir/openvino/FGN.xml'

@serve.deployment(num_replicas=1, ray_actor_options={"num_cpus": 4, "num_gpus": 0}, route_prefix='/predict')
class RWF2000_Deployment:
    def __init__(self):        
        core = Core()
        model_ir = core.read_model(model=MODEL_CHECKPOINT)
        self.compiled_model_ir = core.compile_model(model=model_ir, 
                                                    device_name="CPU",
                                                    config={"INFERENCE_NUM_THREADS": 8})
        
        
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

app = RWF2000_Deployment.bind()
# if __name__ == '__main__':
#     # handle = serve.run(RWF2000_Deployment.bind(),
#     #                    port=5050)
#     serve.start(detached=True, http_options={"port": 5050})
#     RWF2000_Deployment.deploy()
