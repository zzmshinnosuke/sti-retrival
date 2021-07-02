# sti-retrival
通过草图检索图像

## ImageModel
直接通过图像模型resnet，分别提取场景草图和图像的特征，通过triplet loss训练网络。实现场景草图对图像的检索。
```
#training
python scripts/train.py \
    --dataset StiSketchImageDataset \
    --dataset_root_path ~/datasets/STI \
    --model ImageModel \
    --note ImageModel \
    --batch_size 64 \
    --n_epoch 1000 
```

