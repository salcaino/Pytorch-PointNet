# Pytorch implementation of PointNET

Codes directory contain all source code.


Ref: https://arxiv.org/abs/1612.00593

## Training

```
python codes/train_classification.py

python codes/train_segmentation.py

```
Jupyter notebook ```train.ipynb``` can be used to train/test in Google Colab.

## Testing scripts:

```
python codes/show_seg.py \
--feature_transform True \
--idx 10 \
--model seg/weights_with_transform/segmentation_feat_trans_True_Chair.pt \
--dataset ../shapenet_data/shapenetcore_partanno_segmentation_benchmark_v0 \
--class_choice Chair



python codes/show_seg.py \
--feature_transform False \
--idx 10 \
--model seg/weights_without_transform/segmentation_feat_trans_False_Chair.pt \
--dataset ../shapenet_data/shapenetcore_partanno_segmentation_benchmark_v0 \
--class_choice Chair



python codes/show_seg.py \
--feature_transform True \
--idx 10 \
--model seg/weights_with_transform/segmentation_feat_trans_True_Airplane.pt \
--dataset ../shapenet_data/shapenetcore_partanno_segmentation_benchmark_v0 \
--class_choice Airplane



python codes/show_seg.py \
--feature_transform False \
--idx 10 \
--model seg/weights_without_transform/segmentation_feat_trans_False_Airplane.pt \
--dataset ../shapenet_data/shapenetcore_partanno_segmentation_benchmark_v0 \
--class_choice Airplane



python codes/show_cls.py \
--feature_transform True \
--idx 10 \
--model cls/weights_with_transform/classification_feat_trans_True.pt \
--dataset ../shapenet_data/shapenetcore_partanno_segmentation_benchmark_v0



python codes/show_cls.py \
--feature_transform False \
--idx 10 \
--model cls/weights_without_transform/classification_feat_trans_False.pt \
--dataset ../shapenet_data/shapenetcore_partanno_segmentation_benchmark_v0
```
