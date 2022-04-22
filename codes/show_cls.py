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


import matplotlib.pyplot as plt
path = "pretrained_networks/classification_feat_trans_True.pt"

device = "cuda" if torch.cuda.is_available() else "cpu"
parser = argparse.ArgumentParser()
parser.add_argument('--batchSize', type=int, default=32, help='input batch size')
parser.add_argument('--workers', type=int, help='number of data loading workers', default=0)
parser.add_argument('--model', type=str, default=path,  help='model path')
parser.add_argument('--idx', type=int, default=1, help='model index')
parser.add_argument('--feature_transform', default=True, help="use feature transform", type=bool)
parser.add_argument('--dataset', type=str, help='dataset path', default="../shapenet_data/shapenetcore_partanno_segmentation_benchmark_v0")

parser.add_argument('--num_points', type=int,
                    default=2500, help='input batch size')

opt = parser.parse_args()
print(opt)


train_dataset = ShapeNetDataset(
    root=opt.dataset,
    classification=True,
    data_augmentation=False,
    npoints=opt.num_points)

train_dataloader = torch.utils.data.DataLoader(
    train_dataset,
    batch_size=opt.batchSize,
    shuffle=False,
    num_workers=int(opt.workers)
    )

test_dataset = ShapeNetDataset(
    root=opt.dataset,
    classification=True,
    split='test',
    npoints=opt.num_points,
    data_augmentation=False)
test_dataloader = torch.utils.data.DataLoader(
    test_dataset,
    batch_size=opt.batchSize,
    shuffle=False,
    num_workers=int(opt.workers))

num_classes = len(test_dataset.classes)

path = opt.model

num_classes = len(test_dataset.classes)
print('classes', num_classes)

classifier = PointNetCls(num_classes=num_classes, feature_transform=opt.feature_transform)
classifier = classifier.to(device)

checkpoint = torch.load(path, map_location=device)
classifier.load_state_dict(checkpoint['model_state_dict'])
best_acc = checkpoint['best_acc']
epochs = checkpoint['epoch']
print(f'Model loaded with best acc of {best_acc} and trained for {epochs} epochs')

classifier.eval()

idx = opt.idx
print("model %d/%d" % (idx, len(test_dataset)))
points, cls = test_dataset[idx]
points = points.transpose(1, 0)
points = points.to(device)

cmap = plt.cm.get_cmap("hsv", num_classes)
cmap = np.array([cmap(i) for i in range(num_classes)])[:, :3]

points = Variable(points.view(1, points.size()[0], points.size()[1]))
pred, _, _ = classifier(points)
critical_indices = classifier.last_critical_points
pred_choice = pred.data.max(dim=1)[1]
pred_choice = pred_choice.cpu().numpy()

pred_color = cmap[pred_choice[0], :]

critical_indices = critical_indices.squeeze()
critical_indices = critical_indices.unique(sorted=True)
critical_points = points.index_select(dim=2, index=critical_indices)
critical_points.squeeze_()
critical_points = critical_points.transpose(1,0)
critical_points_np = critical_points.cpu().numpy()

random_color = np.random.rand(critical_points_np.shape[0], 3)

showpoints(critical_points_np, None, random_color, background=(1,1,1))

datasets = ["train", "test"]
dataloaders = [train_dataloader, test_dataloader]

classifier.eval()
shape_ious = []
total_preds = []
total_targets = []

for i in range(len(datasets)):
    setname = datasets[i]
    loader = dataloaders[i]

    print(f"Starting to calculate accuracy on {setname} with {len(loader)} batches")
    pbar = tqdm(enumerate(loader, 0))
    with torch.no_grad():
        for i, batch in pbar:
            points, target = batch
            points = points.transpose(2, 1)
            points, target = points.to(device), target.to(device)
            preds, _, _ = classifier(points)
            
            target = target[:, 0]
            pred_labels = torch.max(preds, dim=1)[1]
            total_preds = np.concatenate(
                [total_preds, pred_labels.cpu().numpy()])
            total_targets = np.concatenate(
                [total_targets, target.cpu().numpy()])
            a = 0

    matches = (total_targets == total_preds)
    accuracy = 100 * matches.sum() / matches.size
    print('Mean Accuracy = {:.2f}%'.format(accuracy))
