from __future__ import print_function
import argparse
import os
import random
import torch
import torch.nn.parallel
import torch.nn as nn
import torch.optim as optim
import torch.utils.data
from dataset import ShapeNetDataset
from model import PointNetDenseCls, feature_transform_regularizer
import torch.nn.functional as F
from tqdm import tqdm
import numpy as np

device = "cuda" if torch.cuda.is_available() else "cpu"

parser = argparse.ArgumentParser()
parser.add_argument('--batchSize', type=int, default=8, help='input batch size')
parser.add_argument('--workers', type=int, help='number of data loading workers', default=4)
parser.add_argument('--nepoch', type=int, default=25, help='number of epochs to train for')
parser.add_argument('--model', type=str, default='', help='model path')
parser.add_argument('--outf', type=str, default='seg', help='output folder')
parser.add_argument('--dataset', type=str, required=True, help="dataset path")
parser.add_argument('--class_choice', type=str, default='Chair', help="class_choice")
parser.add_argument('--feature_transform', default='True', help="use feature transform")
parser.add_argument('--save_dir', default='../pretrained_networks', help='directory to save model weights')

opt = parser.parse_args()
print(opt)

opt.manualSeed = random.randint(1, 10000)  # fix seed
print("Random Seed: ", opt.manualSeed)
random.seed(opt.manualSeed)
torch.manual_seed(opt.manualSeed)

dataset = ShapeNetDataset(
    root=opt.dataset,
    classification=False,
    class_choice=[opt.class_choice])
dataloader = torch.utils.data.DataLoader(
    dataset,
    batch_size=opt.batchSize,
    shuffle=True,
    num_workers=int(opt.workers))

val_dataset = ShapeNetDataset(
    root=opt.dataset,
    classification=False,
    class_choice=[opt.class_choice],
    split='test',
    data_augmentation=False)
val_dataloader = torch.utils.data.DataLoader(
    val_dataset,
    batch_size=opt.batchSize,
    shuffle=True,
    num_workers=int(opt.workers))

print(len(dataset), len(val_dataset))
num_classes = dataset.num_seg_classes
print('classes', num_classes)
try:
    os.makedirs(opt.outf)
except OSError:
    pass


# path = opt.model
path = "pretrained_networks/segmentation_feat_trans_True.pt"
load = False

classifier = PointNetDenseCls(num_classes=num_classes, feature_transform=opt.feature_transform)
optimizer = optim.Adam(classifier.parameters(), lr=0.001, betas=(0.9, 0.999))
classifier = classifier.to(device)
classifier_loss = nn.NLLLoss()
lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.5, verbose=False)

best_acc = -1
epochs = 0

if load:
    # classifier.load_state_dict(torch.load(opt.model))
  # model.load_state_dict(torch.load(path))
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
val_ious = []
train_ious = []
for epoch in range(epochs, epochs + opt.nepoch):
    classifier.train()
    epoch_loss = []
    shape_ious = []
    pbar = tqdm(enumerate(dataloader), desc='Batches', leave = True)
    for i, data in pbar:
        points, target = data
        points = points.transpose(2, 1)
        points, target = points.cuda(), target.cuda()
        #TODO
        # perform forward and backward paths, optimize network

        points, target = data
        target = target[:, 0]
        points = points.transpose(2, 1)
        points, target = points.to(device), target.to(device)
        
        # perform forward and backward paths, optimize network
        scores, trans, trans_feat = classifier(points)
        loss = classifier_loss(scores, target) 
        loss = loss + feature_transform_regularizer(trans) + feature_transform_regularizer(trans_feat)
        loss.backward()
        optimizer.step()
        tloss = loss.item()
        epoch_loss.append(tloss)
        pbar.set_description(f"loss: {tloss}")

        pred_choice = scores.data.max(1)[1]

        pred_np = pred_choice.cpu().data.numpy()
        target_np = target.cpu().data.numpy() - 1

        for shape_idx in range(target_np.shape[0]):
            parts = range(num_classes)#np.unique(target_np[shape_idx])
            part_ious = []
            for part in parts:
                I = np.sum(np.logical_and(pred_np[shape_idx] == part, target_np[shape_idx] == part))
                U = np.sum(np.logical_or(pred_np[shape_idx] == part, target_np[shape_idx] == part))
                if U == 0:
                    iou = 1 #If the union of groundtruth and prediction points is empty, then count part IoU as 1
                else:
                    iou = I / float(U)
                part_ious.append(iou)
            shape_ious.append(np.mean(part_ious))
        
    shape_ious = []
    total_preds = []
    total_targets = []
    
    with torch.no_grad():
        classifier.eval()
        pbar = tqdm(enumerate(val_dataloader, 0))
        for i, data in pbar:
            points, target = data
            points = points.transpose(2, 1)
            points, target = points.cuda(), target.cuda()
            pred, _, _ = classifier(points)
            pred_choice = pred.data.max(1)[1]

            pred_np = pred_choice.cpu().data.numpy()
            target_np = target.cpu().data.numpy() - 1

            for shape_idx in range(target_np.shape[0]):
                parts = range(num_classes)#np.unique(target_np[shape_idx])
                part_ious = []
                for part in parts:
                    I = np.sum(np.logical_and(pred_np[shape_idx] == part, target_np[shape_idx] == part))
                    U = np.sum(np.logical_or(pred_np[shape_idx] == part, target_np[shape_idx] == part))
                    if U == 0:
                        iou = 1 #If the union of groundtruth and prediction points is empty, then count part IoU as 1
                    else:
                        iou = I / float(U)
                    part_ious.append(iou)
                shape_ious.append(np.mean(part_ious))

        print("{} Epoch - mIOU for class {}: {:.4f}".format(epoch, opt.class_choice, np.mean(shape_ious)))
        accuracy = np.mean(shape_ious)
    epoch_accuracy.append(accuracy)
    learning_rates.append(lr_scheduler.get_last_lr())
    lr_scheduler.step()

    accuracy = round(accuracy, 3)
    if accuracy > best_acc or best_acc < 0:
        best_acc = accuracy
        print(f"Saving new best model with best acc {best_acc}")
        path = os.path.join(opt.save_dir, f'segmentation_feat_trans_{opt.feature_transform}.pt')
        torch.save({
                'epoch': epoch+1,
                'model_state_dict': classifier.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'best_acc' : best_acc,
                }, path)


## benchmark mIOU
    classifier.eval()
    shape_ious = []
    with torch.no_grad():
        for i,data in tqdm(enumerate(val_dataloader, 0)):
            points, target = data
            points = points.transpose(2, 1)
            points, target = points.cuda(), target.cuda()
            pred, _, _ = classifier(points)
            pred_choice = pred.data.max(1)[1]

            pred_np = pred_choice.cpu().data.numpy()
            target_np = target.cpu().data.numpy() - 1

            for shape_idx in range(target_np.shape[0]):
                parts = range(num_classes)#np.unique(target_np[shape_idx])
                part_ious = []
                for part in parts:
                    I = np.sum(np.logical_and(pred_np[shape_idx] == part, target_np[shape_idx] == part))
                    U = np.sum(np.logical_or(pred_np[shape_idx] == part, target_np[shape_idx] == part))
                    if U == 0:
                        iou = 1 #If the union of groundtruth and prediction points is empty, then count part IoU as 1
                    else:
                        iou = I / float(U)
                    part_ious.append(iou)
                shape_ious.append(np.mean(part_ious))

        print("mIOU for class {}: {:.4f}".format(opt.class_choice, np.mean(shape_ious)))
