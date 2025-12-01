import math
import re
from copy import deepcopy
from functools import partial
from typing import Tuple

import torch
from torch import nn
from timm.models.registry import register_model
from timm.data import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
from timm.models.helpers import build_model_with_cfg
from timm.models.layers import trunc_normal_, to_2tuple
from timm.models.registry import register_model
from custom_model.vision_transformer import Block

__all__ = ['pit_ti']


# @register_model
# def pit_ti_distilled_224(pretrained, **kwargs):
@register_model
def pit_ti(pretrained, **kwargs):
    model_kwargs = dict(
        patch_size=16,
        stride=8,
        base_dims=[32, 32, 32],
        depth=[2, 6, 4],
        heads=[2, 4, 8],
        mlp_ratio=4,
        distilled=True,
        **kwargs
    )
    return _create_pit('pit_ti_distilled_224', pretrained, **model_kwargs)


class SequentialTuple(nn.Sequential):
    """ This module exists to work around torchscript typing issues list -> list"""

    def __init__(self, *args):
        super(SequentialTuple, self).__init__(*args)

    def forward(self, x: Tuple[torch.Tensor, torch.Tensor]) -> Tuple[torch.Tensor, torch.Tensor]:
        feat = []
        attn_list = []
        for module in self:
            x, feat, attn_list = module(x, feat, attn_list)
        return x, feat, attn_list


class Transformer(nn.Module):
    def __init__(
            self, base_dim, depth, heads, mlp_ratio, pool=None, drop_rate=.0, attn_drop_rate=.0, drop_path_prob=None):
        super(Transformer, self).__init__()
        self.layers = nn.ModuleList([])
        embed_dim = base_dim * heads
        self.blocks = nn.Sequential(*[
            Block(
                dim=embed_dim,
                num_heads=heads,
                mlp_ratio=mlp_ratio,
                qkv_bias=True,
                drop=drop_rate,
                attn_drop=attn_drop_rate,
                drop_path=drop_path_prob[i],
                norm_layer=partial(nn.LayerNorm, eps=1e-6)
            )
            for i in range(depth)])

        self.pool = pool

    def forward(self, x: Tuple[torch.Tensor, torch.Tensor], feat, attn_list) -> Tuple[torch.Tensor, torch.Tensor]:
        x, cls_tokens = x  # torch.Size([128, 2, 64])
        B, C, H, W = x.shape  # torch.Size([128, 64, 27, 27])
        token_length = cls_tokens.shape[1]  # 2
        x = x.flatten(2).transpose(1, 2)  # torch.Size([128, 729, 64])
        x = torch.cat((cls_tokens, x), dim=1)  # torch.Size([128, 731, 64])
        for i in range(len(self.blocks)):
            x, attn = self.blocks[i](x)
            feat.append(x)
            attn_list.append(attn)
        # x = self.blocks(x)
        cls_tokens = x[:, :token_length]  # torch.Size([128, 2, 64])
        x = x[:, token_length:]
        x = x.transpose(1, 2).contiguous().reshape(B, C, H, W)  # todo

        if self.pool is not None:
            x, cls_tokens = self.pool(x, cls_tokens)  # torch.Size([128, 128, 14, 14])  #torch.Size([128, 2, 128])
        return (x, cls_tokens), feat, attn_list


class ConvHeadPooling(nn.Module):
    def __init__(self, in_feature, out_feature, stride, padding_mode='zeros'):
        super(ConvHeadPooling, self).__init__()

        self.conv = nn.Conv2d(
            in_feature, out_feature, kernel_size=stride + 1, padding=stride // 2, stride=stride,
            padding_mode=padding_mode, groups=in_feature)
        self.fc = nn.Linear(in_feature, out_feature)

    def forward(self, x, cls_token) -> Tuple[torch.Tensor, torch.Tensor]:
        x = self.conv(x)
        cls_token = self.fc(cls_token)
        return x, cls_token


class ConvEmbedding(nn.Module):
    def __init__(self, in_channels, out_channels, patch_size, stride, padding):
        super(ConvEmbedding, self).__init__()
        self.conv = nn.Conv2d(
            in_channels, out_channels, kernel_size=patch_size, stride=stride, padding=padding, bias=True)

    def forward(self, x):
        x = self.conv(x)
        return x


class PoolingVisionTransformer(nn.Module):

    def __init__(
            self, img_size, patch_size, stride, base_dims, depth, heads,
            mlp_ratio, num_classes=1000, in_chans=3, global_pool='token',
            distilled=False, attn_drop_rate=.0, drop_rate=.0, drop_path_rate=.0):
        super(PoolingVisionTransformer, self).__init__()
        assert global_pool in ('token',)

        padding = 0
        img_size = to_2tuple(img_size)
        patch_size = to_2tuple(patch_size)
        height = math.floor((img_size[0] + 2 * padding - patch_size[0]) / stride + 1)
        width = math.floor((img_size[1] + 2 * padding - patch_size[1]) / stride + 1)

        self.base_dims = base_dims
        self.heads = heads
        self.num_classes = num_classes
        self.global_pool = global_pool
        self.num_tokens = 2 if distilled else 1

        self.patch_size = patch_size
        self.pos_embed = nn.Parameter(torch.randn(1, base_dims[0] * heads[0], height, width))
        self.patch_embed = ConvEmbedding(in_chans, base_dims[0] * heads[0], patch_size, stride, padding)

        self.cls_token = nn.Parameter(torch.randn(1, self.num_tokens, base_dims[0] * heads[0]))
        self.pos_drop = nn.Dropout(p=drop_rate)

        transformers = []
        # stochastic depth decay rule
        dpr = [x.tolist() for x in torch.linspace(0, drop_path_rate, sum(depth)).split(depth)]
        for stage in range(len(depth)):
            pool = None
            if stage < len(heads) - 1:
                pool = ConvHeadPooling(
                    base_dims[stage] * heads[stage], base_dims[stage + 1] * heads[stage + 1], stride=2)
            transformers += [Transformer(
                base_dims[stage], depth[stage], heads[stage], mlp_ratio, pool=pool,
                drop_rate=drop_rate, attn_drop_rate=attn_drop_rate, drop_path_prob=dpr[stage])
            ]
        self.transformers = SequentialTuple(*transformers)
        self.norm = nn.LayerNorm(base_dims[-1] * heads[-1], eps=1e-6)
        self.num_features = self.embed_dim = base_dims[-1] * heads[-1]

        # Classifier head
        self.head = nn.Linear(self.embed_dim, num_classes) if num_classes > 0 else nn.Identity()
        self.head_dist = None
        if distilled:
            self.head_dist = nn.Linear(self.embed_dim, self.num_classes) if num_classes > 0 else nn.Identity()
        self.distilled_training = True  # must set this True to train w/ distillation token

        trunc_normal_(self.pos_embed, std=.02)
        trunc_normal_(self.cls_token, std=.02)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    @torch.jit.ignore
    def no_weight_decay(self):
        return {'pos_embed', 'cls_token'}

    @torch.jit.ignore
    def set_distilled_training(self, enable=True):
        self.distilled_training = enable

    @torch.jit.ignore
    def set_grad_checkpointing(self, enable=True):
        assert not enable, 'gradient checkpointing not supported'

    def get_classifier(self):
        if self.head_dist is not None:
            return self.head, self.head_dist
        else:
            return self.head

    def reset_classifier(self, num_classes, global_pool=None):
        self.num_classes = num_classes
        self.head = nn.Linear(self.embed_dim, num_classes) if num_classes > 0 else nn.Identity()
        if self.head_dist is not None:
            self.head_dist = nn.Linear(self.embed_dim, self.num_classes) if num_classes > 0 else nn.Identity()

    def forward_features(self, x):  # torch.Size([128, 3, 224, 224])
        x = self.patch_embed(x)  # torch.Size([128, 64, 27, 27])
        x = self.pos_drop(x + self.pos_embed)
        cls_tokens = self.cls_token.expand(x.shape[0], -1, -1)
        x, feat, attn_list = self.transformers((x, cls_tokens))
        x, cls_tokens = x
        cls_tokens = self.norm(cls_tokens)
        return cls_tokens, feat, attn_list

    def forward_head(self, x, pre_logits: bool = False) -> torch.Tensor:
        if self.head_dist is not None:
            assert self.global_pool == 'token'
            x, x_dist = x[:, 0], x[:, 1]  # torch.Size([128, 256]) torch.Size([128, 256])
            if not pre_logits:
                x = self.head(x)
                x_dist = self.head_dist(x_dist)
            if self.distilled_training and self.training and not torch.jit.is_scripting():
                # only return separate classification predictions when training in distilled mode
                return x, x_dist
            else:
                # during standard train / finetune, inference average the classifier predictions
                return (x + x_dist) / 2
        else:
            if self.global_pool == 'token':
                x = x[:, 0]
            if not pre_logits:
                x = self.head(x)
            return x

    def forward(self, x, requires_feat=False, requires_attn=False):
        cls_tokens, feat, attn_list = self.forward_features(x)
        x = self.forward_head(cls_tokens)
        if requires_feat and requires_attn:
            return x, feat, attn_list
        elif requires_feat and not requires_attn:
            return x, feat
        else:
            return x

    def stage_info(self, stage):
        if stage == 1:
            shape = (731, 64)
            index = 0
        elif stage == 2:
            shape = (198, 128)
            index = 3
        elif stage == 3:
            index = 7
            shape = (198, 128)
        elif stage == 4:
            index = 11
            shape = (51, 256)
        elif stage == -1:
            index = -1
            shape = (51, 256)
        else:
            raise RuntimeError(f'Stage {stage} out of range (1-4)')
        return index, shape


def checkpoint_filter_fn(state_dict, model):
    """ preprocess checkpoints """
    out_dict = {}
    p_blocks = re.compile(r'pools\.(\d)\.')
    for k, v in state_dict.items():
        # FIXME need to update resize for PiT impl
        # if k == 'pos_embed' and v.shape != model.pos_embed.shape:
        #     # To resize pos embedding when using model at different size from pretrained weights
        #     v = resize_pos_embed(v, model.pos_embed)
        k = p_blocks.sub(lambda exp: f'transformers.{int(exp.group(1))}.pool.', k)
        out_dict[k] = v
    return out_dict


def _create_pit(variant, pretrained=False, **kwargs):
    if kwargs.get('features_only', None):
        raise RuntimeError('features_only not implemented for Vision Transformer models.')

    model = build_model_with_cfg(
        PoolingVisionTransformer, variant, pretrained,
        pretrained_filter_fn=checkpoint_filter_fn,
        **kwargs)
    return model


if __name__ == '__main__':
    import torch
    import time

    model = pit_ti(pretrained=False)
    from test2 import measure_fps

    avg_ms, fps = measure_fps(model)
    print(f"{1}: {avg_ms:.3f} ms per image, {fps:.2f} images/sec")
