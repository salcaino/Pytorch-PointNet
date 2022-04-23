from __future__ import print_function
from show3d_balls import showpoints
import argparse
import numpy as np
import random
import torch
import torch.nn.parallel
import torch.utils.data
from torch.autograd import Variable
from dataset import ShapeNetDataset
from model import PointNetDenseCls
from tqdm import tqdm
import matplotlib.pyplot as plt
import random
device = "cuda" if torch.cuda.is_available() else "cpu"

path = "pretrained_networks/segmentation_feat_trans_True_Chair.pt"

parser = argparse.ArgumentParser()

parser.add_argument('--batchSize', type=int,
                    default=32, help='input batch size')
parser.add_argument('--workers', type=int,
                    help='number of data loading workers', default=0)
parser.add_argument('--model', type=str, default=path, action='store', dest="model", help='model path')
parser.add_argument('--idx', type=int, default=1, help='model index')
parser.add_argument('--feature_transform', default=True,
                    help="use feature transform", type=bool)
parser.add_argument('--dataset', type=str,
                    help='dataset path',
                    default="../shapenet_data/shapenetcore_partanno_segmentation_benchmark_v0")
parser.add_argument('--class_choice', type=str,
                    default='Chair', help='class choice')

opt = parser.parse_args()
print(opt)
path = opt.model
classchoice = opt.class_choice

train_dataset = ShapeNetDataset(
    root=opt.dataset,
    classification=False,
    data_augmentation=False,
    class_choice=[classchoice])
test_dataset = ShapeNetDataset(
    root=opt.dataset,
    classification=False,
    class_choice=[opt.class_choice],
    split='test',
    data_augmentation=False)

train_dataloader = torch.utils.data.DataLoader(
    train_dataset,
    batch_size=opt.batchSize,
    shuffle=False,
    num_workers=int(opt.workers))

test_dataloader = torch.utils.data.DataLoader(
    test_dataset,
    batch_size=opt.batchSize,
    shuffle=False,
    num_workers=int(opt.workers))

num_classes = test_dataset.num_seg_classes
classifier = PointNetDenseCls(
    num_classes=num_classes, feature_transform=opt.feature_transform)
classifier = classifier.to(device)
print(f"Loading model from: {path}")
checkpoint = torch.load(path, map_location=device)
classifier.load_state_dict(checkpoint['model_state_dict'])
best_acc = checkpoint['best_acc']
epochs = checkpoint['epoch']
print(
    f'Model loaded with best acc of {best_acc} and trained for {epochs} epochs')

classifier.eval()

# max_idx = len(test_dataset)
# idx = random.randint(0, max_idx)
idx = opt.idx

print("model %d/%d" % (idx, len(test_dataset)))
point, seg = test_dataset[idx]

point_np = point.numpy()

cmap = plt.cm.get_cmap("hsv", 10)
cmap = np.array([cmap(i) for i in range(10)])[:, :3]
gt = cmap[seg.numpy() - 1, :]
point = point.transpose(1, 0).contiguous()
point = Variable(point.view(1, point.size()[0], point.size()[1])).to(device)
pred, _, _ = classifier(point)
pred_choice = pred.data.max(dim=1)[1]

pred_color = cmap[pred_choice.cpu().numpy()[0], :]

showpoints(point_np, gt, pred_color)
datasets = ["train", "test"]
dataloaders = [train_dataloader, test_dataloader]
shape_ious = []

for i in range(len(datasets)):
    setname = datasets[i]
    loader = dataloaders[i]

    print(
        f"Starting to calculate accuracy for class {classchoice} on {setname} set with {len(loader)} batches")
    pbar = tqdm(enumerate(loader, 0))
    with torch.no_grad():
        for i, data in pbar:
            points, target = data
            points = points.transpose(2, 1)
            points, target = points.to(device), target.to(device)
            preds, _, _ = classifier(points)

            pred_choice = preds.data.max(1)[1]
            pred_np = pred_choice.cpu().data.numpy()
            target_np = target.cpu().data.numpy()

            for shape_idx in range(target_np.shape[0]):
                # np.unique(target_np[shape_idx])
                parts = range(num_classes)
                part_ious = []
                for part in parts:
                    I = np.sum(np.logical_and(
                        pred_np[shape_idx] == part, target_np[shape_idx] == part))
                    U = np.sum(np.logical_or(
                        pred_np[shape_idx] == part, target_np[shape_idx] == part))
                    if U == 0:
                        iou = 1  # If the union of groundtruth and prediction points is empty, then count part IoU as 1
                    else:
                        iou = I / float(U)
                    part_ious.append(iou)
                shape_ious.append(np.mean(part_ious))
    print("mIOU for class {}: {:.3f}%".format(
        classchoice, (np.mean(shape_ious) * 100)))
