from __future__ import print_function

import copy

import torch
import torch.nn as nn
import torch.nn.parallel
import torch.utils.data
from torch.autograd import Variable
import numpy as np
import torch.nn.functional as F


class TNet(nn.Module):
    '''
        The input of this network is a [b * k * n] tensor 
        - b is the batchsize
        - k is the dimension of each point
        - n is the number of points for each input in the batch

        The output should be a [b * k * k] tensor.
    '''

    def __init__(self, k=64):
        super(TNet, self).__init__()

        self.k = k

        # Each layer has batchnorm and relu on it
        # layer 1: k -> 64
        self.l1 = nn.Sequential(
          nn.Conv1d(in_channels=k, out_channels=64, kernel_size=1),
          nn.BatchNorm1d(num_features=64),
          nn.ReLU()
        )
        # layer 2:  64 -> 128
        self.l2 = nn.Sequential(
          nn.Conv1d(in_channels=64, out_channels=128, kernel_size=1),
          nn.BatchNorm1d(num_features=128),
          nn.ReLU()
        )
        # layer 3: 128 -> 1024
        self.l3 = nn.Sequential(
          nn.Conv1d(in_channels=128, out_channels=1024, kernel_size=1),
          nn.BatchNorm1d(num_features=1024),
          nn.ReLU()
        )
        self.firstblock = nn.Sequential(
          self.l1,
          self.l2,
          self.l3,
        )

        # fc 1024 -> 512
        self.fc = nn.Sequential(
          nn.Linear(in_features=1024, out_features=512),
          nn.BatchNorm1d(num_features=512),
          nn.ReLU()
        )

        # fc 512 -> 256
        self.fc2 = nn.Sequential(
          # nn.Dropout(p=0.3),
          nn.Linear(in_features=512, out_features=256),
          nn.BatchNorm1d(num_features=256),
          nn.ReLU()
        )
        # fc 256 -> k*k (no batchnorm, no relu)
        self.fc3 = nn.Linear(in_features=256, out_features=k*k)
        bias = torch.eye(k).view(1, -1).float()
        self.learnable_bias = nn.Parameter(bias)

        self.secondblock = nn.Sequential(
            self.fc,
            self.fc2,
            self.fc3,
        )

    def forward(self, x):
        batch_size, _, num_points = x.shape
        x1 = self.firstblock(x)
        x2 = torch.max(x1, dim=2, keepdim=True)[0]
        x2 = x2.reshape(-1, 1024)
        x3 = self.secondblock(x2)
        bias = self.learnable_bias.repeat(batch_size, 1)
        x4 = torch.add(x3, bias)
        x5 = x4.view(batch_size, self.k, self.k)
        return x5


class PointNetfeat(nn.Module):
    def __init__(self, global_feat=True, feature_transform=True):
        super(PointNetfeat, self).__init__()

        self.feature_transform = feature_transform
        self.global_feat = global_feat
        # Use TNet to apply transformation on input and multiply the input points with the transformation
        self.tnet1 = TNet(k=3)
        # layer 1:3 -> 64
        self.l1 = nn.Sequential(
            nn.Conv1d(in_channels=3, out_channels=64, kernel_size=1),
            nn.BatchNorm1d(num_features=64),
            nn.ReLU()
        )
        # Use TNet to apply transformation on features and multiply the input features with the transformation
        #                                                                        (if feature_transform is true)
        self.tnet2 = TNet(k=64)

        # layer2: 64 -> 128
        self.l2 = nn.Sequential(
            nn.Conv1d(in_channels=64, out_channels=128, kernel_size=1),
            nn.BatchNorm1d(num_features=128),
            nn.ReLU()
        )
        # layer 3: 128 -> 1024 (no relu)
        self.l3 = nn.Sequential(
            nn.Conv1d(in_channels=128, out_channels=1024, kernel_size=1),
            # nn.BatchNorm1d(num_features=128),
        )
        # ReLU activation
        self.activation = nn.ReLU()

    def forward(self, x):
        batch_size, _, num_points = x.shape
        # input transformation, you will need to return the transformation matrix as you will need it for the regularization loss
        t1_3x3 = self.tnet1(x)
        xout = torch.bmm(t1_3x3, x)

        # apply layer 1
        xout = self.l1(xout)
        # feature transformation, you will need to return the transformation matrix as you will need it for the regularization loss
        if self.feature_transform:
            t2_64x64 = self.tnet2(xout)
            xout = torch.bmm(t2_64x64, xout)
        else:
            t2_64x64 = None
        if not self.global_feat:
            device = xout.device
            xfeat = xout.clone().to(device)
        # apply layer 2
        xout = self.l2(xout)
        # apply layer 3
        xout = self.l3(xout)

        # apply maxpooling
        xout = torch.max(xout, dim=2, keepdim=True)[0]
        xout = xout.view(batch_size, -1)
        # return output, input transformation matrix, feature transformation matrix
        if self.global_feat:  # This shows if we're doing classification or segmentation
            # doing classification
            return xout, t1_3x3, t2_64x64
        else:
            # doing segmentation
            xout = xout.view(-1, 1024, 1).repeat(1, 1, num_points)
            return torch.cat([xfeat, xout], 1), t1_3x3, t2_64x64


class PointNetCls(nn.Module):
    def __init__(self, num_classes=2, feature_transform=False):
        super(PointNetCls, self).__init__()
        self.feature_transform = feature_transform
        self.feat = PointNetfeat(
            global_feat=True, feature_transform=feature_transform)
        self.fc1 = nn.Linear(1024, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, num_classes)
        self.dropout = nn.Dropout(p=0.3)
        self.bn1 = nn.BatchNorm1d(512)
        self.bn2 = nn.BatchNorm1d(256)
        self.relu = nn.ReLU()

    def forward(self, x):
        x, trans, trans_feat = self.feat(x)
        x = F.relu(self.bn1(self.fc1(x)))
        x = F.relu(self.bn2(self.dropout(self.fc2(x))))
        x = self.fc3(x)
        return F.log_softmax(x, dim=1), trans, trans_feat


class PointNetDenseCls(nn.Module):
    '''
      The input of this network is a [b * 3 * n] tensor 
        - b is the batchsize
        - 3 is the dimension of each point
        - n is the number of points for each input in the batch 
      The output should be a [b * n * k] tensor. 
        - k is the number of parts for this class of objects,
          so the output is k scores for each point in each point cloud in the batch

    '''

    def __init__(self, num_classes=2, feature_transform=False):
        super(PointNetDenseCls, self).__init__()
        # TODO
        # get global features + point features from PointNetfeat
        self.feature_transform = feature_transform
        self.feat = PointNetfeat(
            global_feat=False, feature_transform=feature_transform)
        self.block = nn.Sequential(
            nn.Conv1d(in_channels=1088, out_channels=512, kernel_size=1),
            nn.BatchNorm1d(num_features=512),
            nn.ReLU(),
            nn.Conv1d(in_channels=512, out_channels=256, kernel_size=1),
            nn.BatchNorm1d(num_features=256),
            nn.ReLU(),
            nn.Conv1d(in_channels=256, out_channels=128, kernel_size=1),
            nn.BatchNorm1d(num_features=128),
            nn.ReLU(),
            nn.Conv1d(in_channels=128, out_channels=num_classes),
            nn.LogSoftmax(dim=2)
        )

    def forward(self, x):
        print(x)
        x, t1_3x3, t2_64x64 = self.feat(x)
        x = self.block(x)


def feature_transform_regularizer(trans):
    '''
        The input of this function is a [b * k * k] tensor. 
            - b is the batchsize
            - k is the size of the matrix A
        The output should be a single value, the mean of the loss for all samples in the batch. 
        You can apply a weight 0.001 to this regularization loss as the original paper.
    '''
    batch_size, feature_size, _ = trans.shape
    device = trans.device
    weight = torch.FloatTensor([0.001]).to(device)
    # compute I - AA^t
    iden_ = torch.eye(feature_size).to(device)
    t = trans.clone().to(device).transpose(2, 1)
    a_ = torch.bmm(trans, t)
    input = iden_ - a_
    # compute norm
    loss = torch.norm(input, p='fro')
    return loss * weight


if __name__ == '__main__':
    sim_data = Variable(torch.rand(32, 3, 2500))
    trans = TNet(k=3)
    out = trans(sim_data)
    print('TNet', out.size())
    print('loss', feature_transform_regularizer(out))

    sim_data_64d = Variable(torch.rand(32, 64, 2500))
    trans = TNet(k=64)
    out = trans(sim_data_64d)
    print('TNet 64d', out.size())
    print('loss', feature_transform_regularizer(out))

    pointfeat = PointNetfeat(global_feat=True)
    out, _, _ = pointfeat(sim_data)
    print('global feat', out.size())

    pointfeat = PointNetfeat(global_feat=False)
    out, _, _ = pointfeat(sim_data)
    print('point feat', out.size())

    cls = PointNetCls(num_classes=5)
    out, _, _ = cls(sim_data)
    print('class', out.size())

    seg = PointNetDenseCls(num_classes=3)
    out, _, _ = seg(sim_data)
    print('seg', out.size())
