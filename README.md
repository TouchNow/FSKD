# Distilling Structural Knowledge from CNNs to Vision Transformers for Data-Efficient Visual Recognition



### Training

```
python -m torch.distributed.launch --nproc_per_node=4 train.py --data_dir ./data/cifar/ --dataset cifar100 --config configs/cifar/vit_mlp.yaml  --model pit_ti --teacher convnext_tiny --num-classes 100 --distiller simikd --patch-align lg --channel-align cg --simi-global-weight 1.0 --simi-patch-weight 1.0 --simi-attn-weight 40000.0 --kd-loss-weight 1.0 --simi-stage 3 4
```
```
python -m torch.distributed.launch --nproc_per_node=8 train.py --dataset imagenet --data_dir /path/to/imagenet --config configs/imagenet/vit_mlp.yaml  --model deit_ti --teacher regnety_160 --num-classes 1000 --distiller simikd --patch-align lg --channel-align cg --simi-global-weight 100.0 --simi-patch-weight 1.0 --simi-attn-weight 1000000.0 --kd-loss-weight 1.0 --simi-stage 1 2
```

Other results can be reproduced following similar commands by modifying:

`--config `: configuration of training strategy. 

`--model`: student model architecture.

`--teacher`: teacher model architecture.

`--distiller`: which KD algorithm to use.

For information about other tunable parameters, please refer to `train.py`.



