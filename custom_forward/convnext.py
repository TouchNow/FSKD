from timm.models.convnext import ConvNeXt
from custom_forward.registry import register_method
import timm
import sys
from collections import defaultdict
from types import MethodType

_target_class = ConvNeXt

_implemented_methods = defaultdict(dict)


@register_method
def forward(self, x, requires_feat=False):
    if requires_feat:
        x, feat = self.forward_features(x, requires_feat=True)
        x = self.forward_head(x, pre_logits=True)
        feat.append(x)
        x = self.head.fc(x)
        return x, feat
    else:
        x = self.forward_features(x, requires_feat=False)
        x = self.forward_head(x)
        return x


@register_method
def forward_features(self, x, requires_feat):
    feat = []
    x = self.stem(x)
    if requires_feat:
        for stage in self.stages:
            x = stage(x)
            feat.append(x)
    else:
        x = self.stages(x)
    x = self.norm_pre(x)
    return (x, feat) if requires_feat else x


@register_method
def stage_info(self, stage):
    if self.pretrained_cfg['architecture'] == 'convnext_tiny' or self.pretrained_cfg[
        'architecture'] == 'convnext_small':
        if stage == 1:
            index = 0
            shape = (96, 56, 56)
        elif stage == 2:
            index = 1
            shape = (192, 28, 28)
        elif stage == 3:
            index = 2
            shape = (384, 14, 14)
        elif stage == 4:
            index = 3
            shape = (768, 7, 7)
        elif stage == -1:
            index = -1
            shape = 768
        else:
            raise RuntimeError(f'Stage {stage} out of range (1-4)')
    else:
        raise NotImplementedError(f'undefined stage_info() for model {self.pretrained_cfg["architecture"]}')
    return index, shape


if __name__ == '__main__':
    import torch
    import torchvision
    import timm.models as models
    from timm.models import create_model

    model = create_model('convnext_tiny', pretrained=False)
    data = torch.randn(1, 3, 224, 224)
    out, feat = model(data, requires_feat=True)
    print(out.shape)
    print(len(feat))
    for f in feat:
        print(f.shape)
