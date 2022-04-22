# # PointNet
from __future__ import print_function
import argparse
import os
import random

import numpy as np
import torch
import torch.nn.parallel
import torch.nn as nn
import torch.optim as optim
import torch.utils.data
from dataset import ShapeNetDataset
from model import PointNetCls, feature_transform_regularizer
import torch.nn.functional as F
from tqdm import tqdm
import matplotlib.pyplot as plt

device = "cuda" if torch.cuda.is_available() else "cpu"


parser = argparse.ArgumentParser()
parser.add_argument('--batchSize', type=int, default=32, help='input batch size')
parser.add_argument('--num_points', type=int, default=2500, help='input batch size')
parser.add_argument('--workers', type=int, help='number of data loading workers', default=0)
parser.add_argument('--model', type=str, default='pretrained_networks/classification_feat_trans_True.pt', help='model path')
parser.add_argument('--nepoch', type=int, default=250, help='number of epochs to train for')
parser.add_argument('--outf', type=str, default='cls', help='output folder')
parser.add_argument('--dataset', type=str, required=False, help="dataset path", default="../shapenet_data/shapenetcore_partanno_segmentation_benchmark_v0")
parser.add_argument('--feature_transform', default='True', help="use feature transform")
parser.add_argument('--save_dir', default='pretrained_networks', help='directory to save model weights')

opt = parser.parse_args(args=[])
print(opt)

if not os.path.exists(opt.save_dir):
    os.makedirs(opt.save_dir, exist_ok=True)
try:
    os.makedirs(opt.outf, exist_ok=True)
except OSError:
    pass


opt.manualSeed = random.randint(1, 10000)  # fix seed
print("Random Seed: ", opt.manualSeed)
random.seed(opt.manualSeed)
torch.manual_seed(opt.manualSeed)


opt.manualSeed = random.randint(1, 10000)  # fix seed
print("Random Seed: ", opt.manualSeed)
random.seed(opt.manualSeed)
torch.manual_seed(opt.manualSeed)

 
# ## Dataloaders
dataset = ShapeNetDataset(
    root=opt.dataset,
    classification=True,
    npoints=opt.num_points)

dataloader = torch.utils.data.DataLoader(
    dataset,
    batch_size=opt.batchSize,
    shuffle=True,
    num_workers=int(opt.workers))

val_dataset = ShapeNetDataset(
    root=opt.dataset,
    classification=True,
    split='val',
    npoints=opt.num_points)

val_dataloader = torch.utils.data.DataLoader(
    val_dataset,
    batch_size=opt.batchSize,
    shuffle=False,
    num_workers=int(opt.workers))

print(len(dataset))

path = opt.model
# path = "pretrained_networks/classification_feat_trans_True.pt"
load = len(path) > 0

num_classes = len(dataset.classes)
print('classes', num_classes)
# classifier = PointNetCls(num_classes=num_classes, feature_transform=opt.feature_transform)
classifier = PointNetCls(num_classes=num_classes, feature_transform=False)


criterion = nn.NLLLoss()
optimizer = optim.Adam(classifier.parameters(), lr=0.001, betas=(0.9, 0.999))
lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.5, verbose=False)

classifier = classifier.to(device)
best_acc = -1
epochs = 0

if load:
  checkpoint = torch.load(path, map_location=device)
  classifier.load_state_dict(checkpoint['model_state_dict'])
  optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
  best_acc = checkpoint['best_acc']
  if 'epoch' in checkpoint:
    epochs = checkpoint['epoch']
  print(f'Model loaded with best acc of {best_acc} and trained for {epochs} epochs')

num_batch = len(dataloader)
epoch_losses = []
epoch_accuracy = []
learning_rates = []

train_epochs = opt.nepoch 
# ## Train
for epoch in range(epochs, epochs + opt.nepoch):
    classifier.train()
    epoch_loss = []
    pbar = tqdm(enumerate(dataloader), desc='Batches', leave=True)
    for i, data in pbar:
        points, target = data

        points = points.transpose(2, 1)
        points, target = points.to(device), target.to(device)
        scores, trans, trans_feat = classifier(points)
        if dataset.classification:
            target = target[:, 0]
        # perform forward and backward paths, optimize network
        loss = criterion(scores, target)
        if classifier.feature_transform:
          loss = loss + \
            feature_transform_regularizer(
                trans) + feature_transform_regularizer(trans_feat)
        loss.backward()
        optimizer.step()
        tloss = loss.item()
        epoch_loss.append(tloss)
        pbar.set_description(f"loss: {tloss}")

    epoch_losses.append(np.mean(epoch_loss))

    classifier.eval()
    shape_ious = []
    total_preds = []
    total_targets = []
    with torch.no_grad():
        pbar = tqdm(enumerate(val_dataloader, 0))
        for i, data in pbar:
            points, target = data
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
    print('{} Epoch - Accuracy = {:.2f}%'.format(epoch, accuracy))

    epoch_accuracy.append(accuracy)
    learning_rates.append(lr_scheduler.get_last_lr())
    lr_scheduler.step()
    accuracy = round(accuracy, 3)
    if accuracy > best_acc or best_acc < 0:
        best_acc = accuracy
        print(f"Saving new best model with best acc {best_acc}")
        savepath = os.path.join(opt.save_dir, f'classification_feat_trans_{opt.feature_transform}.pt')
        torch.save({
            'epoch': epoch+1,
            'model_state_dict': classifier.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'best_acc': best_acc,
        }, savepath)
