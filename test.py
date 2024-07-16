import os
import argparse
from tqdm import tqdm

import torch.nn as nn
import torch
from torch.utils.data import DataLoader
import utils
from thop import profile

from data_RGB import get_test_data
from Networks.model import Net
from skimage import img_as_ubyte

import numpy as np

parser = argparse.ArgumentParser()
parser.add_argument('--network', default='model_C', type=str, help='Network Type')
parser.add_argument('--input_dir', default='../../../share/UIE/datasets/', type=str, help='Directory of validation images')
parser.add_argument('--result_dir', default='./results/', type=str, help='Directory for results')
parser.add_argument('--weights', default='./checkpoints/model_C/model_latest.pth', type=str, help='Path to weights')
parser.add_argument('--dataset', default='Cycle_600', type=str, help='Test Dataset') # ['GoPro', 'HIDE', 'RealBlur_J', 'RealBlur_R']
parser.add_argument('--gpus', default='0', type=str, help='CUDA_VISIBLE_DEVICES')

args = parser.parse_args()

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = args.gpus

model_restoration = Net()
model_restoration.cuda()

utils.load_checkpoint(model_restoration, args.weights)
print("===>Testing using weights: ",args.weights)

# model_restoration = nn.DataParallel(model_restoration)
model_restoration.eval()

dataset = args.dataset
rgb_dir_test = os.path.join(args.input_dir, 'test', dataset)
print(rgb_dir_test)
test_dataset = get_test_data(rgb_dir_test, img_options={})
test_loader = DataLoader(dataset=test_dataset, batch_size=1, shuffle=False, num_workers=4, drop_last=False, pin_memory=True)

result_dir = os.path.join(args.result_dir, args.network, dataset)
utils.mkdir(result_dir)

with torch.no_grad():
    for ii, data_test in enumerate(tqdm(test_loader), 0):
        torch.cuda.ipc_collect()
        torch.cuda.empty_cache()

        input_ = data_test[0].cuda()
        filenames = data_test[1]

        restored = model_restoration(input_)
        restored0 = torch.clamp(restored[2], 0, 1)
        restored0 = restored0.permute(0, 2, 3, 1).cpu().detach().numpy()

        # restored2 = torch.clamp(restored[1], 0, 1)
        # restored2 = restored2.permute(0, 2, 3, 1).cpu().detach().numpy()
        #
        # restored4 = torch.clamp(restored[0], 0, 1)
        # restored4 = restored4.permute(0, 2, 3, 1).cpu().detach().numpy()

        restored_img0 = img_as_ubyte(restored0[0])
        utils.save_img((os.path.join(result_dir, filenames[0]+'.png')), restored_img0)
        # restored_img2 = img_as_ubyte(restored2[0])
        # utils.save_img((os.path.join(result_dir, filenames[0]+ '2.png')), restored_img2)
        # restored_img4 = img_as_ubyte(restored4[0])
        # utils.save_img((os.path.join(result_dir, filenames[0]+ '4.png')), restored_img4)

    flops, params = profile(model_restoration, inputs=(input_,))
    print('flops: ', flops, 'params: ', params)

