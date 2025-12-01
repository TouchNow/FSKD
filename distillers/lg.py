import torch
import torch.nn.functional as F
from torch import nn

from ._base import BaseDistiller
from .registry import register_distiller
from .utils import get_module_dict, init_weights, is_cnn_model, set_module_dict
import math
from einops import rearrange
from .utils import kd_loss


@register_distiller
class LG(BaseDistiller):
    requires_feat = True

    def __init__(self, student, teacher, criterion, args, **kwargs):
        super(LG, self).__init__(student, teacher, criterion, args)

        self.projector = nn.ModuleDict()

        for stage in self.args.lg_stage:
            _, size_s = self.student.stage_info(stage)
            _, size_t = self.teacher.stage_info(stage)
            projector = LGCriterion(size_s, size_t)
            set_module_dict(self.projector, stage, projector)

        self.projector.apply(init_weights)

    def forward(self, image, label, *args, **kwargs):
        with torch.no_grad():
            self.teacher.eval()
            logits_teacher, feat_teacher = self.teacher(image, requires_feat=True)

        logits_student, feat_student = self.student(image, requires_feat=True)
        if len(logits_student) == 2:
            logits_student, logits_student_dist = logits_student
            loss_kd = self.args.kd_loss_weight * kd_loss(logits_student_dist, logits_teacher, self.args.kd_temperature)
        else:
            loss_kd = self.args.kd_loss_weight * kd_loss(logits_student, logits_teacher, self.args.kd_temperature)

        lg_losses = []
        for stage in self.args.lg_stage:
            idx_s, _ = self.student.stage_info(stage)
            idx_t, _ = self.teacher.stage_info(stage)
            lg_loss = get_module_dict(self.projector, stage)(feat_student[idx_s], feat_teacher[idx_t])
            lg_losses.append(lg_loss)

            loss_lg = self.args.lg_loss_weight * sum(lg_losses)
            loss_gt = self.args.gt_loss_weight * self.criterion(logits_student, label)
            losses_dict = {
                "loss_gt": loss_gt,
                "loss_lg": loss_lg,
                "loss_kd": loss_kd
            }
        return logits_student, losses_dict


class LGCriterion(nn.Module):
    def __init__(self, size_s, size_t):
        super().__init__()
        C, H, W = size_t
        if len(size_s) == 2:
            self.vit_feat = True
            L, Embed = size_s
            self.extra_token = get_extra_token(L)
            print("this is vit feat, the extra token is: ", self.extra_token, "!")
            self.h_s = int(math.sqrt(L))
            self.w_s = int(math.sqrt(L))
            self.channel_align = nn.Sequential(
                nn.Conv2d(Embed, C, kernel_size=1),
            )

        elif len(size_s) == 3:
            self.vit_feat = False
            print("this is cnn feat")
            self.c_s, self.h_s, self.w_s = size_s
            self.channel_align = nn.Sequential(
                nn.Conv2d(self.c_s, C, kernel_size=1),
            )

    def forward(self, x, feat_t=None):
        B = feat_t.size(0)
        if self.vit_feat:
            x = x[:, self.extra_token:]
            x = x.view(B, self.h_s, self.w_s, -1)
            x = x.permute(0, 3, 1, 2).contiguous()
        dsize = (max(feat_t.size(-2), x.size(-2)), max(feat_t.size(-1), x.size(-1)))
        feat_t = F.interpolate(feat_t, dsize, mode='bilinear', align_corners=False)
        x = F.interpolate(x, dsize, mode='bilinear', align_corners=False)
        x = self.channel_align(x)
        loss = (feat_t - x).pow(2).mean() / B
        return loss


def get_extra_token(L):
    extra_token = L - int(math.sqrt(L)) ** 2
    return extra_token
