# Underwater image enhancement via cross-wise transformer network focusing on pre-post differences

This repo contains the PyTorch implementation for our paper.

## Environment
```
1. Python 3.8.18
2. PyTorch 1.13.1
3. Torchvision 0.14.1
4. OpenCV-Python 4.8.1.78
5. NumPy 1.24.4
```

## Test
```
1. Clone repo
2. Download checkpoints folder and place it in repo
3. Put the images in your folder A with name B
4. Change 'input_dir' to A and 'dataset' to B in test.py
5. Run test.py
6. Find results in 'result_dir'
```

## Train
```
1. Change 'TRAIN_DIR', 'VAL_DIR' in training.yml
2. Run train.py
3. Find trained model in checkpoints
```

## Cite
If you use our code, please cite our paper. Thanks!
```
@article{HUANG2024112000,
title = {Underwater image enhancement via cross-wise transformer network focusing on pre-post differences},
journal = {Applied Soft Computing},
pages = {112000},
year = {2024},
issn = {1568-4946},
doi = {https://doi.org/10.1016/j.asoc.2024.112000}
}
```

## Contact
If you have any questions, please contact: Zhixiong Huang: hzxcyanwind@mail.dlut.edu.cn
