from functools import partial

import torch
from torch import nn as nn

from timm.data import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
from custom_model.vision_transformer import VisionTransformer, trunc_normal_, checkpoint_filter_fn

from timm.models.helpers import build_model_with_cfg, checkpoint_seq
from timm.models.registry import register_model


@register_model
def deit_ti(pretrained=False, **kwargs):
    model_kwargs = dict(patch_size=16, embed_dim=192, depth=12, num_heads=3, **kwargs)
    model = _create_deit(
        'deit_tiny_distilled_patch16_224', pretrained=pretrained, **model_kwargs)
    return model


@register_model
def deit_small(pretrained=False, **kwargs):
    model_kwargs = dict(patch_size=16, embed_dim=384, depth=12, num_heads=6, **kwargs)
    model = _create_deit(
        'deit_small_distilled_patch16_224', pretrained=pretrained, **model_kwargs)
    return model


def _create_deit(variant, pretrained=False, **kwargs):
    if kwargs.get('features_only', None):
        raise RuntimeError('features_only not implemented for Vision Transformer models.')
    model_cls = VisionTransformerDistilled
    model = build_model_with_cfg(
        model_cls, variant, pretrained,
        pretrained_filter_fn=partial(checkpoint_filter_fn, adapt_layer_scale=True),
        **kwargs)
    return model


class VisionTransformerDistilled(VisionTransformer):

    def __init__(self, *args, **kwargs):
        weight_init = kwargs.pop('weight_init', '')
        super().__init__(*args, **kwargs, weight_init='skip')
        assert self.global_pool in ('token',)

        self.num_prefix_tokens = 2
        self.dist_token = nn.Parameter(torch.zeros(1, 1, self.embed_dim))
        self.pos_embed = nn.Parameter(
            torch.zeros(1, self.patch_embed.num_patches + self.num_prefix_tokens, self.embed_dim))
        self.head_dist = nn.Linear(self.embed_dim, self.num_classes) if self.num_classes > 0 else nn.Identity()
        self.distilled_training = True  # must set this True to train w/ distillation token

        self.init_weights(weight_init)

    def init_weights(self, mode=''):
        trunc_normal_(self.dist_token, std=.02)
        super().init_weights(mode=mode)

    @torch.jit.ignore
    def group_matcher(self, coarse=False):
        return dict(
            stem=r'^cls_token|pos_embed|patch_embed|dist_token',
            blocks=[
                (r'^blocks\.(\d+)', None),
                (r'^norm', (99999,))]  # final norm w/ last block
        )

    @torch.jit.ignore
    def get_classifier(self):
        return self.head, self.head_dist

    def reset_classifier(self, num_classes, global_pool=None):
        self.num_classes = num_classes
        self.head = nn.Linear(self.embed_dim, num_classes) if num_classes > 0 else nn.Identity()
        self.head_dist = nn.Linear(self.embed_dim, self.num_classes) if num_classes > 0 else nn.Identity()

    @torch.jit.ignore
    def set_distilled_training(self, enable=True):
        self.distilled_training = enable

    def forward_features(self, x) -> torch.Tensor:
        x = self.patch_embed(x)
        x = torch.cat((
            self.cls_token.expand(x.shape[0], -1, -1),
            self.dist_token.expand(x.shape[0], -1, -1), x), dim=1)
        x = self.pos_drop(x + self.pos_embed)
        feat = []
        attn_list = []
        if self.grad_checkpointing and not torch.jit.is_scripting():
            x = checkpoint_seq(self.blocks, x)
        else:
            for i in range(len(self.blocks)):
                x, attn = self.blocks[i](x)
                feat.append(x)
                attn_list.append(attn)
            # x, attn = self.blocks(x)
        x = self.norm(x)
        return x, feat, attn_list

    def forward_head(self, x, pre_logits: bool = False) -> torch.Tensor:
        if pre_logits:
            return (x[:, 0] + x[:, 1]) / 2
        x, x_dist = self.head(x[:, 0]), self.head_dist(x[:, 1])
        if self.distilled_training and self.training and not torch.jit.is_scripting():
            # only return separate classification predictions when training in distilled mode
            return x, x_dist
        else:
            # during standard train / finetune, inference average the classifier predictions
            return (x + x_dist) / 2

    def forward(self, x, requires_feat=False, requires_attn=False):
        x, feat, attn_list = self.forward_features(x)
        x = self.forward_head(x)
        if requires_feat and requires_attn:
            return x, feat, attn_list
        elif requires_feat and not requires_attn:
            return x, feat
        else:
            return x

    def stage_info(self, stage):
        if self.embed_dim == 192:
            shape = (198, 192)
        elif self.embed_dim == 384:
            shape = (198, 384)
        if stage == 1:
            index = 0
        elif stage == 2:
            index = 3
        elif stage == 3:
            index = 9
        elif stage == 4:
            index = 11
        else:
            raise RuntimeError(f'Stage {stage} out of range (1-4)')
        return index, shape


if __name__ == '__main__':
    import torch
    import time

    model = deit_small()
    print(model.stage_info(1))
    # model.eval()
    # from test2 import measure_fps
    #
    # avg_ms, fps = measure_fps(model)
    # print(f"{1}: {avg_ms:.3f} ms per image, {fps:.2f} images/sec")
