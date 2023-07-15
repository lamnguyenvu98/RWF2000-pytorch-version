# RWF2000-pytorch-version

Unofficial pytorch implementation of paper "*RWF2000 - A Large Scale Video Database for Violence Detection*"

## 1. Installation
```
python -m venv rwf2000env
source activate rwf2000env/bin/activate
pip install -r requirements.txt
```

### 2. Prepare dataset
```
python dataset/build_dataset.py --source path/to/originald/dataset --target /path/to/build/folder
```

## 3. Training
### 3.1 Configuration yaml file

### 3.2 Train the model
```
python train.py --config rwf2000.yaml
```

## 4. Run inference
```
python inference.py --video test.mp4 --save-dir results/out.mp4 --checkpoint path/to/checkpoint
```

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
