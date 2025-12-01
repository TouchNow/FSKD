"""RegNet

Paper: `Designing Network Design Spaces` - https://arxiv.org/abs/2003.13678
Original Impl: https://github.com/facebookresearch/pycls/blob/master/pycls/models/regnet.py

Based on original PyTorch impl linked above, but re-wrote to use my own blocks (adapted from ResNet here)
and cleaned up with more descriptive variable names.

Weights from original impl have been modified
* first layer from BGR -> RGB as most PyTorch models are
* removed training specific dict entries from checkpoints and keep model state_dict only
* remap names to match the ones here

Hacked together by / Copyright 2020 Ross Wightman
"""
import math
from dataclasses import dataclass
from functools import partial
from typing import Optional, Union, Callable

import numpy as np
import torch
import torch.nn as nn

from timm.data import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
from timm.models.helpers import build_model_with_cfg, named_apply, checkpoint_seq
from timm.models.layers import ClassifierHead, AvgPool2dSame, ConvNormAct, SEModule, DropPath, GroupNormAct
from timm.models.layers import get_act_layer, get_norm_act_layer, create_conv2d
from timm.models.registry import register_model


def _create_regnet(variant, pretrained, **kwargs):
    return build_model_with_cfg(
        RegNet, variant, pretrained,
        model_cfg=model_cfgs[variant],
        pretrained_filter_fn=_filter_fn,
        **kwargs)


@register_model
def regnety_160(pretrained=False, **kwargs):
    """RegNetY-16GF"""
    return _create_regnet('regnety_160', pretrained, **kwargs)


@dataclass
class RegNetCfg:
    depth: int = 21
    w0: int = 80
    wa: float = 42.63
    wm: float = 2.66
    group_size: int = 24
    bottle_ratio: float = 1.
    se_ratio: float = 0.
    stem_width: int = 32
    downsample: Optional[str] = 'conv1x1'
    linear_out: bool = False
    preact: bool = False
    num_features: int = 0
    act_layer: Union[str, Callable] = 'relu'
    norm_layer: Union[str, Callable] = 'batchnorm'


# Model FLOPS = three trailing digits * 10^8
model_cfgs = dict(
    # RegNet-X
    regnetx_160=RegNetCfg(w0=216, wa=55.59, wm=2.1, group_size=128, depth=22),
    # RegNet-Y
    regnety_160=RegNetCfg(w0=200, wa=106.23, wm=2.48, group_size=112, depth=18, se_ratio=0.25),
)


def _cfg(url='', **kwargs):
    return {
        'url': url, 'num_classes': 1000, 'input_size': (3, 224, 224), 'pool_size': (7, 7),
        'crop_pct': 0.875, 'interpolation': 'bicubic',
        'mean': IMAGENET_DEFAULT_MEAN, 'std': IMAGENET_DEFAULT_STD,
        'first_conv': 'stem.conv', 'classifier': 'head.fc',
        **kwargs
    }


default_cfgs = dict(
    regnetx_160=_cfg(
        url='https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-regnet/regnetx_160-c98c4112.pth'),
    regnety_160=_cfg(
        url='https://dl.fbaipublicfiles.com/deit/regnety_160-a5fe301d.pth',  # from Facebook DeiT GitHub repository
        crop_pct=1.0, test_input_size=(3, 288, 288)),
)


def quantize_float(f, q):
    """Converts a float to closest non-zero int divisible by q."""
    return int(round(f / q) * q)


def adjust_widths_groups_comp(widths, bottle_ratios, groups):
    """Adjusts the compatibility of widths and groups."""
    bottleneck_widths = [int(w * b) for w, b in zip(widths, bottle_ratios)]
    groups = [min(g, w_bot) for g, w_bot in zip(groups, bottleneck_widths)]
    bottleneck_widths = [quantize_float(w_bot, g) for w_bot, g in zip(bottleneck_widths, groups)]
    widths = [int(w_bot / b) for w_bot, b in zip(bottleneck_widths, bottle_ratios)]
    return widths, groups


def generate_regnet(width_slope, width_initial, width_mult, depth, group_size, q=8):
    """Generates per block widths from RegNet parameters."""
    assert width_slope >= 0 and width_initial > 0 and width_mult > 1 and width_initial % q == 0
    # TODO dWr scaling?
    # depth = int(depth * (scale ** 0.1))
    # width_scale = scale ** 0.4  # dWr scale, exp 0.8 / 2, applied to both group and layer widths
    widths_cont = np.arange(depth) * width_slope + width_initial
    width_exps = np.round(np.log(widths_cont / width_initial) / np.log(width_mult))
    widths = width_initial * np.power(width_mult, width_exps)
    widths = np.round(np.divide(widths, q)) * q
    num_stages, max_stage = len(np.unique(widths)), width_exps.max() + 1
    groups = np.array([group_size for _ in range(num_stages)])
    return widths.astype(int).tolist(), num_stages, groups.astype(int).tolist()


def downsample_conv(in_chs, out_chs, kernel_size=1, stride=1, dilation=1, norm_layer=None, preact=False):
    norm_layer = norm_layer or nn.BatchNorm2d
    kernel_size = 1 if stride == 1 and dilation == 1 else kernel_size
    dilation = dilation if kernel_size > 1 else 1
    if preact:
        return create_conv2d(in_chs, out_chs, kernel_size, stride=stride, dilation=dilation)
    else:
        return ConvNormAct(
            in_chs, out_chs, kernel_size, stride=stride, dilation=dilation, norm_layer=norm_layer, apply_act=False)


def downsample_avg(in_chs, out_chs, kernel_size=1, stride=1, dilation=1, norm_layer=None, preact=False):
    """ AvgPool Downsampling as in 'D' ResNet variants. This is not in RegNet space but I might experiment."""
    norm_layer = norm_layer or nn.BatchNorm2d
    avg_stride = stride if dilation == 1 else 1
    pool = nn.Identity()
    if stride > 1 or dilation > 1:
        avg_pool_fn = AvgPool2dSame if avg_stride == 1 and dilation > 1 else nn.AvgPool2d
        pool = avg_pool_fn(2, avg_stride, ceil_mode=True, count_include_pad=False)
    if preact:
        conv = create_conv2d(in_chs, out_chs, 1, stride=1)
    else:
        conv = ConvNormAct(in_chs, out_chs, 1, stride=1, norm_layer=norm_layer, apply_act=False)
    return nn.Sequential(*[pool, conv])


def create_shortcut(
        downsample_type, in_chs, out_chs, kernel_size, stride, dilation=(1, 1), norm_layer=None, preact=False):
    assert downsample_type in ('avg', 'conv1x1', '', None)
    if in_chs != out_chs or stride != 1 or dilation[0] != dilation[1]:
        dargs = dict(stride=stride, dilation=dilation[0], norm_layer=norm_layer, preact=preact)
        if not downsample_type:
            return None  # no shortcut, no downsample
        elif downsample_type == 'avg':
            return downsample_avg(in_chs, out_chs, **dargs)
        else:
            return downsample_conv(in_chs, out_chs, kernel_size=kernel_size, **dargs)
    else:
        return nn.Identity()  # identity shortcut (no downsample)


class Bottleneck(nn.Module):
    """ RegNet Bottleneck

    This is almost exactly the same as a ResNet Bottlneck. The main difference is the SE block is moved from
    after conv3 to after conv2. Otherwise, it's just redefining the arguments for groups/bottleneck channels.
    """

    def __init__(
            self, in_chs, out_chs, stride=1, dilation=(1, 1), bottle_ratio=1, group_size=1, se_ratio=0.25,
            downsample='conv1x1', linear_out=False, act_layer=nn.ReLU, norm_layer=nn.BatchNorm2d,
            drop_block=None, drop_path_rate=0.):
        super(Bottleneck, self).__init__()
        act_layer = get_act_layer(act_layer)
        bottleneck_chs = int(round(out_chs * bottle_ratio))
        groups = bottleneck_chs // group_size

        cargs = dict(act_layer=act_layer, norm_layer=norm_layer)
        self.conv1 = ConvNormAct(in_chs, bottleneck_chs, kernel_size=1, **cargs)
        self.conv2 = ConvNormAct(
            bottleneck_chs, bottleneck_chs, kernel_size=3, stride=stride, dilation=dilation[0],
            groups=groups, drop_layer=drop_block, **cargs)
        if se_ratio:
            se_channels = int(round(in_chs * se_ratio))
            self.se = SEModule(bottleneck_chs, rd_channels=se_channels, act_layer=act_layer)
        else:
            self.se = nn.Identity()
        self.conv3 = ConvNormAct(bottleneck_chs, out_chs, kernel_size=1, apply_act=False, **cargs)
        self.act3 = nn.Identity() if linear_out else act_layer()
        self.downsample = create_shortcut(downsample, in_chs, out_chs, 1, stride, dilation, norm_layer=norm_layer)
        self.drop_path = DropPath(drop_path_rate) if drop_path_rate > 0 else nn.Identity()

    def zero_init_last(self):
        nn.init.zeros_(self.conv3.bn.weight)

    def forward(self, x):
        shortcut = x
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.se(x)
        x = self.conv3(x)
        if self.downsample is not None:
            # NOTE stuck with downsample as the attr name due to weight compatibility
            # now represents the shortcut, no shortcut if None, and non-downsample shortcut == nn.Identity()
            x = self.drop_path(x) + self.downsample(shortcut)
        x = self.act3(x)
        return x


class PreBottleneck(nn.Module):
    """ RegNet Bottleneck

    This is almost exactly the same as a ResNet Bottlneck. The main difference is the SE block is moved from
    after conv3 to after conv2. Otherwise, it's just redefining the arguments for groups/bottleneck channels.
    """

    def __init__(
            self, in_chs, out_chs, stride=1, dilation=(1, 1), bottle_ratio=1, group_size=1, se_ratio=0.25,
            downsample='conv1x1', linear_out=False, act_layer=nn.ReLU, norm_layer=nn.BatchNorm2d,
            drop_block=None, drop_path_rate=0.):
        super(PreBottleneck, self).__init__()
        norm_act_layer = get_norm_act_layer(norm_layer, act_layer)
        bottleneck_chs = int(round(out_chs * bottle_ratio))
        groups = bottleneck_chs // group_size

        self.norm1 = norm_act_layer(in_chs)
        self.conv1 = create_conv2d(in_chs, bottleneck_chs, kernel_size=1)
        self.norm2 = norm_act_layer(bottleneck_chs)
        self.conv2 = create_conv2d(
            bottleneck_chs, bottleneck_chs, kernel_size=3, stride=stride, dilation=dilation[0], groups=groups)
        if se_ratio:
            se_channels = int(round(in_chs * se_ratio))
            self.se = SEModule(bottleneck_chs, rd_channels=se_channels, act_layer=act_layer)
        else:
            self.se = nn.Identity()
        self.norm3 = norm_act_layer(bottleneck_chs)
        self.conv3 = create_conv2d(bottleneck_chs, out_chs, kernel_size=1)
        self.downsample = create_shortcut(downsample, in_chs, out_chs, 1, stride, dilation, preact=True)
        self.drop_path = DropPath(drop_path_rate) if drop_path_rate > 0 else nn.Identity()

    def zero_init_last(self):
        pass

    def forward(self, x):
        x = self.norm1(x)
        shortcut = x
        x = self.conv1(x)
        x = self.norm2(x)
        x = self.conv2(x)
        x = self.se(x)
        x = self.norm3(x)
        x = self.conv3(x)
        if self.downsample is not None:
            # NOTE stuck with downsample as the attr name due to weight compatibility
            # now represents the shortcut, no shortcut if None, and non-downsample shortcut == nn.Identity()
            x = self.drop_path(x) + self.downsample(shortcut)
        return x


class RegStage(nn.Module):
    """Stage (sequence of blocks w/ the same output shape)."""

    def __init__(
            self, depth, in_chs, out_chs, stride, dilation,
            drop_path_rates=None, block_fn=Bottleneck, **block_kwargs):
        super(RegStage, self).__init__()
        self.grad_checkpointing = False

        first_dilation = 1 if dilation in (1, 2) else 2
        for i in range(depth):
            block_stride = stride if i == 0 else 1
            block_in_chs = in_chs if i == 0 else out_chs
            block_dilation = (first_dilation, dilation)
            dpr = drop_path_rates[i] if drop_path_rates is not None else 0.
            name = "b{}".format(i + 1)
            self.add_module(
                name, block_fn(
                    block_in_chs, out_chs, stride=block_stride, dilation=block_dilation,
                    drop_path_rate=dpr, **block_kwargs)
            )
            first_dilation = dilation

    def forward(self, x):
        if self.grad_checkpointing and not torch.jit.is_scripting():
            x = checkpoint_seq(self.children(), x)
        else:
            for block in self.children():
                x = block(x)
        return x


class RegNet(nn.Module):
    """RegNet-X, Y, and Z Models

    Paper: https://arxiv.org/abs/2003.13678
    Original Impl: https://github.com/facebookresearch/pycls/blob/master/pycls/models/regnet.py
    """

    def __init__(
            self, cfg: RegNetCfg, in_chans=3, num_classes=1000, output_stride=32, global_pool='avg',
            drop_rate=0., drop_path_rate=0., zero_init_last=True):
        super().__init__()
        self.num_classes = num_classes
        self.drop_rate = drop_rate
        assert output_stride in (8, 16, 32)

        # Construct the stem
        stem_width = cfg.stem_width
        na_args = dict(act_layer=cfg.act_layer, norm_layer=cfg.norm_layer)
        if cfg.preact:
            self.stem = create_conv2d(in_chans, stem_width, 3, stride=2)
        else:
            self.stem = ConvNormAct(in_chans, stem_width, 3, stride=2, **na_args)
        self.feature_info = [dict(num_chs=stem_width, reduction=2, module='stem')]

        # Construct the stages
        prev_width = stem_width
        curr_stride = 2
        per_stage_args, common_args = self._get_stage_args(
            cfg, output_stride=output_stride, drop_path_rate=drop_path_rate)
        assert len(per_stage_args) == 4
        block_fn = PreBottleneck if cfg.preact else Bottleneck
        for i, stage_args in enumerate(per_stage_args):
            stage_name = "s{}".format(i + 1)
            self.add_module(stage_name, RegStage(in_chs=prev_width, block_fn=block_fn, **stage_args, **common_args))
            prev_width = stage_args['out_chs']
            curr_stride *= stage_args['stride']
            self.feature_info += [dict(num_chs=prev_width, reduction=curr_stride, module=stage_name)]

        # Construct the head
        if cfg.num_features:
            self.final_conv = ConvNormAct(prev_width, cfg.num_features, kernel_size=1, **na_args)
            self.num_features = cfg.num_features
        else:
            final_act = cfg.linear_out or cfg.preact
            self.final_conv = get_act_layer(cfg.act_layer)() if final_act else nn.Identity()
            self.num_features = prev_width
        self.head = ClassifierHead(
            in_chs=self.num_features, num_classes=num_classes, pool_type=global_pool, drop_rate=drop_rate)

        named_apply(partial(_init_weights, zero_init_last=zero_init_last), self)

    def _get_stage_args(self, cfg: RegNetCfg, default_stride=2, output_stride=32, drop_path_rate=0.):
        # Generate RegNet ws per block
        widths, num_stages, stage_gs = generate_regnet(cfg.wa, cfg.w0, cfg.wm, cfg.depth, cfg.group_size)

        # Convert to per stage format
        stage_widths, stage_depths = np.unique(widths, return_counts=True)
        stage_br = [cfg.bottle_ratio for _ in range(num_stages)]
        stage_strides = []
        stage_dilations = []
        net_stride = 2
        dilation = 1
        for _ in range(num_stages):
            if net_stride >= output_stride:
                dilation *= default_stride
                stride = 1
            else:
                stride = default_stride
                net_stride *= stride
            stage_strides.append(stride)
            stage_dilations.append(dilation)
        stage_dpr = np.split(np.linspace(0, drop_path_rate, sum(stage_depths)), np.cumsum(stage_depths[:-1]))

        # Adjust the compatibility of ws and gws
        stage_widths, stage_gs = adjust_widths_groups_comp(stage_widths, stage_br, stage_gs)
        arg_names = ['out_chs', 'stride', 'dilation', 'depth', 'bottle_ratio', 'group_size', 'drop_path_rates']
        per_stage_args = [
            dict(zip(arg_names, params)) for params in
            zip(stage_widths, stage_strides, stage_dilations, stage_depths, stage_br, stage_gs, stage_dpr)]
        common_args = dict(
            downsample=cfg.downsample, se_ratio=cfg.se_ratio, linear_out=cfg.linear_out,
            act_layer=cfg.act_layer, norm_layer=cfg.norm_layer)
        return per_stage_args, common_args

    @torch.jit.ignore
    def group_matcher(self, coarse=False):
        return dict(
            stem=r'^stem',
            blocks=r'^s(\d+)' if coarse else r'^s(\d+)\.b(\d+)',
        )

    @torch.jit.ignore
    def set_grad_checkpointing(self, enable=True):
        for s in list(self.children())[1:-1]:
            s.grad_checkpointing = enable

    @torch.jit.ignore
    def get_classifier(self):
        return self.head.fc

    def reset_classifier(self, num_classes, global_pool='avg'):
        self.head = ClassifierHead(self.num_features, num_classes, pool_type=global_pool, drop_rate=self.drop_rate)

    def forward_features(self, x):
        feat = []
        x = self.stem(x)
        x = self.s1(x)
        feat.append(x)
        x = self.s2(x)
        feat.append(x)
        x = self.s3(x)
        feat.append(x)
        x = self.s4(x)
        feat.append(x)
        x = self.final_conv(x)
        return x, feat

    def forward_head(self, x, pre_logits: bool = False):
        return self.head(x, pre_logits=pre_logits)

    def forward(self, x, requires_feat=False):
        x, feat = self.forward_features(x)
        x = self.forward_head(x)
        if requires_feat:
            return x, feat
        else:
            return x

    def stage_info(self, stage):
        if stage == 1:
            index = 0
            shape = (224, 56, 56)
        elif stage == 2:
            index = 1
            shape = (448, 28, 28)
        elif stage == 3:
            index = 2
            shape = (1232, 14, 14)
        elif stage == 4:
            index = 3
            shape = (3024, 7, 7)
        return index, shape


def _init_weights(module, name='', zero_init_last=False):
    if isinstance(module, nn.Conv2d):
        fan_out = module.kernel_size[0] * module.kernel_size[1] * module.out_channels
        fan_out //= module.groups
        module.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
        if module.bias is not None:
            module.bias.data.zero_()
    elif isinstance(module, nn.Linear):
        nn.init.normal_(module.weight, mean=0.0, std=0.01)
        if module.bias is not None:
            nn.init.zeros_(module.bias)
    elif zero_init_last and hasattr(module, 'zero_init_last'):
        module.zero_init_last()


def _filter_fn(state_dict):
    """ convert patch embedding weight from manual patchify + linear proj to conv"""
    if 'model' in state_dict:
        # For DeiT trained regnety_160 pretraiend model
        state_dict = state_dict['model']
    return state_dict


@register_model
def regnetx_160(pretrained=False, **kwargs):
    """RegNetX-16GF"""
    return _create_regnet('regnetx_160', pretrained, **kwargs)


if __name__ == '__main__':
    import torch
    import time

    model = regnety_160()
    model.eval()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    input_tensor = torch.randn(1, 3, 224, 224).to(device)
    for _ in range(20):
        _ = model(input_tensor)
    torch.cuda.synchronize() if device.type == "cuda" else None
    start_time = time.time()
    iters = 200  # 测200次求平均
    for _ in range(iters):
        _ = model(input_tensor)
    torch.cuda.synchronize() if device.type == "cuda" else None
    end_time = time.time()
    avg_inference_time = (end_time - start_time) / iters  # 秒
    fps = 1.0 / avg_inference_time
    print(f"Average inference time: {avg_inference_time * 1000:.3f} ms")
    print(f"FPS: {fps:.2f}")
