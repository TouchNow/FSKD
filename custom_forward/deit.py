import torch

from timm.models.vision_transformer import VisionTransformer
from .registry import register_method
from timm.models.deit import VisionTransformerDistilled
from timm.models.helpers import build_model_with_cfg, checkpoint_seq

_target_class = VisionTransformerDistilled


@register_method
def forward(self, x, requires_feat=False):
    if requires_feat:
        x, feat = self.forward_features(x, requires_feat=True)
        x = self.forward_head(x)
        feat.append(x)
        return x, feat
    else:
        x = self.forward_features(x, requires_feat=False)
        x = self.forward_head(x)
    return x


@register_method
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


@register_method
def forward_features(self, x, requires_feat):
    feat = []
    x = self.patch_embed(x)
    x = torch.cat((self.cls_token.expand(x.shape[0], -1, -1),
                   self.dist_token.expand(x.shape[0], -1, -1), x), dim=1)
    x = self.pos_drop(x + self.pos_embed)
    if requires_feat:
        for blk in self.blocks:
            x = blk(x)
            feat.append(x)
    else:
        x = self.blocks(x)
    x = self.norm(x)
    return (x, feat) if requires_feat else x


@register_method
def stage_info(self, stage):
    if self.pretrained_cfg['architecture'] == 'vit_small_patch16_224':
        shape = (197, 384)
        if stage == 1:
            index = 1
        elif stage == 2:
            index = 3
        elif stage == 3:
            index = 9
        elif stage == 4:
            index = 11
        elif stage == -1:
            index = -1
            shape = 384
        else:
            raise RuntimeError(f'Stage {stage} out of range (1-4)')
    elif self.pretrained_cfg['architecture'] == 'vit_base_patch16_224':
        shape = (197, 768)
        if stage == 1:
            index = 1
        elif stage == 2:
            index = 3
        elif stage == 3:
            index = 9
        elif stage == 4:
            index = 11
        elif stage == -1:
            index = -1
            shape = 768
        else:
            raise RuntimeError(f'Stage {stage} out of range (1-4)')
    elif self.pretrained_cfg['architecture'] == 'deit_tiny_patch16_224':
        shape = (198, 192)
        if stage == 1:
            index = 1
        elif stage == 2:
            index = 3
        elif stage == 3:
            index = 9
        elif stage == 4:
            index = 11
        elif stage == -1:
            index = -1
            shape = 192
        else:
            raise RuntimeError(f'Stage {stage} out of range (1-4)')
    elif self.pretrained_cfg['architecture'] == 'deit_small_distilled_patch16_224':
        shape = (198, 384)
        if stage == 1:
            index = 1
        elif stage == 2:
            index = 3
        elif stage == 3:
            index = 9
        elif stage == 4:
            index = 11
        elif stage == -1:
            index = -1
            shape = 384
        else:
            raise RuntimeError(f'Stage {stage} out of range (1-4)')
    else:
        raise NotImplementedError(f'undefined stage_info() for model {self.pretrained_cfg["architecture"]}')
    return index, shape
