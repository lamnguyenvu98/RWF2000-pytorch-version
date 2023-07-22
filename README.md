# Unofficial implementation of Flow Gated Network using Pytorch-Lightning

Official repository in Keras: https://github.com/mchengny/RWF2000-Video-Database-for-Violence-Detection

## 1. Installation
- Require Python 3.10
```
git clone https://github.com/lamnguyenvu98/RWF2000-pytorch-version.git

cd RWF2000-pytorch-version

python -m pip install .
```

## 2. Training
### 2.1 Dataset structure
```
RWF2000 (root)
    ├── train
    |    ├── Fight
    |    |     ├── data1.npz
    |    |     ├── data2.npz
    |    |     └── ...
    |    └── NonFight
    |          ├── data1.npz
    |          ├── data2.npz
    |          └── ...
    |
    └── val
         ├── Fight
         |     ├── data1.npz
         |     ├── data2.npz
         |     └── ...
         └── NonFight
               ├── data1.npz
               ├── data2.npz
               └── ...
```

### 2.2 Preprocessing Dataset
#### 2.2.1 Install RWF2000 dataset
- Go into official repo: https://github.com/mchengny/RWF2000-Video-Database-for-Violence-Detection

- Download dataset in **Download** section.

#### 2.2.2 Preprocess dataset

```
build-dataset --source <path of downloaded dataset> --target <path of processed dataset directory>
```
- Download RWF2000 dataset and specify its path to `--source`
- Create a new directory and specify it in `--target`

### 2.3 Configure settings in `rwf2000.yaml`
**TOP_K**: [int] save k checkpoints with highest accuracy. (Default: 3)

**PRECISION**: [str] mixed precision training. (Default: 16-mixed)

**RESUME**: [boolean] resume training from last checkpoint. (Default: True)

**RESUME_CHECKPOINT**: [str] path of checkpoint to resume the training.

**CHECKPOINT_DIR**: [str] path of directory that store all training checkpoints.

**DATA_DIR**: [str] path of preprocessed dataset in step `2.2`

**LOG_DIR**: [str] path of log directory.

**NEPTUNE_LOGGER** section should leave empty if not use it for monitoring. **WITH_ID** can leave empty in case you use Neptune logger, only use it when you want to log into existing Neptune section.

### 2.4 Train model
```
python examples/train.py --config rwf2000.yaml
```

## 3 Inference
- Start ray serve:
```
serve run src.rayserve.deployment:app --port 5050
```

- Send request:

```
import requests
import imageio.v3 as iio
import pickle
import json

video_path = 'videos/_q5Nwh4Z6ao_3.avi'

frames = iio.imread(video_path, extension='.avi')[:64]

arr = pickle.dumps(frames)

url = "http://127.0.0.1:5050/predict"

headers={'Content-Type': 'application/json'}

response = requests.post(
    url=url,
    headers=headers,
    data=arr)

result = json.loads(response.content.decode('utf-8'))

print(result)
```

## 4. Export torchscript
```
export-script --checkpoint <path of checkpoint> --scriptpath <path of torchscript>
```

Examples:
```
export-script --checkpoint model_dir/best.ckpt --scriptpath model_dir/best.tc
```

## Cite:
```
@INPROCEEDINGS{9412502,
  author={Cheng, Ming and Cai, Kunjing and Li, Ming},
  booktitle={2020 25th International Conference on Pattern Recognition (ICPR)}, 
  title={RWF-2000: An Open Large Scale Video Database for Violence Detection}, 
  year={2021},
  volume={},
  number={},
  pages={4183-4190},
  doi={10.1109/ICPR48806.2021.9412502}}
```