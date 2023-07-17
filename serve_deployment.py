from ray import serve
from src.models.fgn_model import FlowGatedNetwork
from src.utils import preprocessing
from src.data.augmentation import Normalize, ToTensor

from torchvision import transforms

import os
import numpy as np
import torch
import pickle
from starlette.requests import Request

MODEL_CHECKPOINT = './model_dir/best.ckpt'

@serve.deployment(num_replicas=1, ray_actor_options={"num_cpus": 4, "num_gpus": 0}, route_prefix='/predict')
class RWF2000_Deployment:
    def __init__(self):
        self.model = FlowGatedNetwork()
        pl_checkpoint = torch.load(os.path.abspath(MODEL_CHECKPOINT), map_location="cpu")['state_dict']
        model_ckp = self.model.state_dict()
        for k, v in model_ckp.items():
            model_ckp[k] = pl_checkpoint['model.' + k]
        self.model.load_state_dict(model_ckp)
        self.model = self.model.cpu()
        self.data_transform = transforms.Compose([
            Normalize(),
            ToTensor()
            ])
        self.classnames = ['Fight', 'NonFight']

    def predict(self, frames: np.ndarray) -> dict:
        data = preprocessing(frames, dynamic_crop=False)
        data: torch.Tensor = self.data_transform(data)
        data = data.unsqueeze(0).permute(0, 4, 1, 2, 3).float()
        out: torch.Tensor = self.model(data.cpu())
        best_idx = out.softmax(-1).argmax(-1)
        score = out.softmax(-1)[0][best_idx].item()
        label = self.classnames[best_idx]
        
        return {"Class": label, "Score": "{:.1f}".format(score)}
    
    async def __call__(self, http_request: Request):
        data = await http_request.body()
        frames: np.ndarray = pickle.loads(data)
        result = self.predict(frames)
        return result

rwf2000_app = RWF2000_Deployment.bind()