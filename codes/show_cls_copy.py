from __future__ import print_function
import argparse
import torch
import torch.nn.parallel
import torch.utils.data
from torch.autograd import Variable
from dataset import ShapeNetDataset
from model import PointNetCls
import torch.nn.functional as F
from tqdm import tqdm
import numpy as np
from show3d_balls import showpoints
import random
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt


device = "cuda" if torch.cuda.is_available() else "cpu"
parser = argparse.ArgumentParser()
parser.add_argument('--model', type=str, default='',  help='model path')
parser.add_argument('--num_points', type=int,
                    default=2500, help='input batch size')

opt = parser.parse_args()
print(opt)

test_dataset = ShapeNetDataset(
    root='../shapenet_data/shapenetcore_partanno_segmentation_benchmark_v0',
    classification=True,
    split='test',
    npoints=opt.num_points,
    data_augmentation=False)
num_classes = len(test_dataset.classes)

classifier = PointNetCls(num_classes=len(test_dataset.classes))
classifier.to(device)
path = "pretrained_networks/classification_feat_trans_True.pt"

checkpoint = torch.load(path, map_location=device)
classifier.load_state_dict(checkpoint['model_state_dict'])
best_acc = checkpoint['best_acc']
epochs = checkpoint['epoch']
print(f'Model loaded with best acc of {best_acc} and trained for {epochs} epochs')

classifier.eval()

max_idx = len(test_dataset)
idx = random.randint(0, max_idx)

print("model %d/%d" % (idx, len(test_dataset)))
points, cls = test_dataset[idx]
points = points.transpose(1, 0)
points = points.to(device)
print(points.size(), cls.size())
print(f"Class label {list(test_dataset.classes)[cls[0]]}")

cmap = plt.cm.get_cmap("hsv", num_classes)
cmap = np.array([cmap(i) for i in range(num_classes)])[:, :3]

points = Variable(points.view(1, points.size()[0], points.size()[1]))
pred, _, _ = classifier(points)
critical_indices = classifier.last_critical_points
pred_choice = pred.data.max(dim=1)[1]
pred_choice = pred_choice.numpy()
print(f"Class {pred_choice[0]} predicted:", list(test_dataset.classes)[pred_choice[0]])
pred_color = cmap[pred_choice[0], :]

critical_indices = critical_indices.squeeze()
critical_indices = critical_indices.unique(sorted=True)
critical_points = points.index_select(dim=2, index=critical_indices)
critical_points.squeeze_()
critical_points = critical_points.transpose(1,0)
critical_points_np = critical_points.numpy()

random_color = np.random.rand(critical_points_np.shape[0], 3)

showpoints(critical_points_np, None, random_color, background=(1,1,1))
