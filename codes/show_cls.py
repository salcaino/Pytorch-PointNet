from __future__ import print_function
import argparse
import torch
import torch.nn.parallel
import torch.utils.data
from torch.autograd import Variable
from dataset import ShapeNetDataset
from model import PointNetCls
import torch.nn.functional as F

import numpy as np

parser = argparse.ArgumentParser()
parser.add_argument('--model', type=str, default = '',  help='model path')
parser.add_argument('--num_points', type=int, default=2500, help='input batch size')

opt = parser.parse_args()
print(opt)

test_dataset = ShapeNetDataset(
    root='../shapenet_data/shapenetcore_partanno_segmentation_benchmark_v0',
    classification=True,
    split='test',
    npoints=2500)

test_dataloader = torch.utils.data.DataLoader(
    test_dataset,
    batch_size=32,
    shuffle=True,
    num_workers=4)

classifier = PointNetCls(num_classes=len(test_dataset.classes))
classifier.cuda()
classifier.load_state_dict(torch.load(opt.model)['model'])
classifier.eval()


total_preds = []
total_targets = []
with torch.no_grad():
    for i, data in enumerate(test_dataloader, 0):
        #TODO
        # calculate average classification accuracy