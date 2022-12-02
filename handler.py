import sys
# sys.path.append('/home/pep/Drive/PCLOUD/Projects/RWF2000-Flow-Gated-Net')
sys.path.append('/app')
from serve.ts.torch_handler.base_handler import BaseHandler
from model import TrainingModel
from torchvision import transforms
from Dataset.augmentation import Normalize, ToTensor
from utils import preprocessing
import torch
import pickle
from base64 import b64decode
import numpy as np
import os
import logging
import json

class ModelHandler(BaseHandler):
    def __init__(self):
        super().__init__()
        self._context = None
        self.initialized = False
        self.explain = False
        self.target = 0
        self.tfms = transforms.Compose([
                    Normalize(),
                    ToTensor()])
    
    def initialize(self, context):
        self._context = context
        manifest = context.manifest
        properties = context.system_properties
        
        model_dir = properties.get("model_dir")
        
        # serialized_file = manifest['model']['serializedFile']
        model_file = manifest['model']['modelFile']
        model_ckp_path = os.path.join(model_dir, model_file)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if not os.path.isfile(model_ckp_path):
            raise RuntimeError("Missing model checkpoint...: {} not existed".format(model_ckp_path))

        mapping_file_path = os.path.join(model_dir, 'index_to_classes.json')
        
        if not os.path.isfile(mapping_file_path):
            raise RuntimeError("{} not existed".format(mapping_file_path))
        else:
            with open(mapping_file_path) as f:
                self.class_names = json.load(f)
            logging.info(f'Successful loaded mapping file')
        
        self.model = TrainingModel().load_from_checkpoint(model_ckp_path)
        self.initialized = True
    
    def preprocess(self, data):
        """
        Transform raw input into model input data.
        :param batch: list of raw requests, should match batch size
        :return: list of preprocessed model input data
        """
        batch = []
        for i in range(len(data)):
            
            preprocessed_data = data[i].get("data")
            if preprocessed_data is None:
                preprocessed_data = data[i].get("body")
            
            # preprocessed_data = np.frombuffer(base64.b64decode(preprocessed_data), dtype=np.float32).reshape(64, 224, 224, 3)
            preprocessed_data = pickle.loads(preprocessed_data)
            preprocessed_data = preprocessing(preprocessed_data.copy(), dynamic_crop=False).astype(np.float32)
            preprocessed_data = self.tfms(preprocessed_data.copy())
            batch.append(preprocessed_data)

        preprocessed_data = torch.stack(batch)
        preprocessed_data = preprocessed_data.permute(0, 4, 1, 2, 3).float()
        
        # preprocessed_data = data[0].get("data")
        # if preprocessed_data is None:
        #     preprocessed_data = data[0].get("body")
        
        # preprocessed_data = np.frombuffer(b64decode(preprocessed_data), dtype=np.float32).reshape(64, 224, 224, 3)
        # preprocessed_data = preprocessing(preprocessed_data.copy(), dynamic_crop=False).astype(np.float32)
        # preprocessed_data = self.tfms(preprocessed_data.copy())
        # preprocessed_data = preprocessed_data.unsqueeze(0).permute(0, 4, 1, 2, 3).float()
        return preprocessed_data
    
    def inference(self, data):
        model_output = self.model.forward(data)
        return model_output
    
    def postprocess(self, output):
        # best_idx_class = output.softmax(-1).argmax()
        preds = output.softmax(-1).detach().numpy()
        result = []
        for i in range(preds.shape[0]):
            # best_idx = preds[i].argmax()
            # score = preds[i][best_idx]
            # class_names = self.class_names[str(best_idx)]
            # result.append({"Predict": class_names, "Probability": "{:.2f}".format(score)})
            result.append({self.class_names["0"]: "{:.2f}".format(preds[i][0]), 
                           self.class_names["1"]: "{:.2f}".format(preds[i][1])})
        
        return result

    def handle(self, data, context):
        model_input = self.preprocess(data)
        output = self.inference(model_input)
        return self.postprocess(output)
        
        
        
        
