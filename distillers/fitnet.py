import torch
import torch.nn.functional as F
from torch import nn

from ._base import BaseDistiller
from .registry import register_distiller
from .utils import get_module_dict, init_weights, is_cnn_model, set_module_dict
import math
from einops import rearrange
from .utils import kd_loss


# self.relu(self.bn(x)), F.adaptive_avg_pool2d(t, (self.s_H, self.s_H))
@register_distiller
class FitNet(BaseDistiller):
    requires_feat = True

    def __init__(self, student, teacher, criterion, args, **kwargs):
        super(FitNet, self).__init__(student, teacher, criterion, args)

        self.projector = nn.ModuleDict()

        for stage in self.args.fitnet_stage:
            _, size_s = self.student.stage_info(stage)
            _, size_t = self.teacher.stage_info(stage)
            in_chans_s, _, _ = size_s
            in_chans_t, _, _ = size_t
            projector = nn.Conv2d(in_chans_s, in_chans_t, 1, 1, 0, bias=False)
            # projector = FitNetCriterion(size_s, size_t)
            set_module_dict(self.projector, stage, projector)

        self.projector.apply(init_weights)

    def forward(self, image, label, *args, **kwargs):
        with torch.no_grad():
            self.teacher.eval()
            logits_teacher, feat_teacher = self.teacher(image, requires_feat=True)

        logits_student, feat_student = self.student(image, requires_feat=True)
        if len(logits_student) == 2:
            logits_student, logits_student_dist = logits_student
            loss_kd = 0 * kd_loss(logits_student_dist, logits_teacher, self.args.kd_temperature)
        else:
            loss_kd = 0 * kd_loss(logits_student, logits_teacher, self.args.kd_temperature)

        fitnet_losses = []
        for stage in self.args.fitnet_stage:
            idx_s, _ = self.student.stage_info(stage)
            idx_t, _ = self.teacher.stage_info(stage)
            # fitnet_loss = get_module_dict(self.projector, stage)(feat_student[idx_s], feat_teacher[idx_t])
            # fitnet_loss = get_module_dict(self.projector, stage)(feat_student[idx_s])
            # fitnet_losses.append(fitnet_loss)
            feat_s = get_module_dict(self.projector, stage)(feat_student[idx_s])
            feat_t = feat_teacher[idx_t]
            fitnet_losses.append(F.mse_loss(feat_s, feat_t))

            loss_fitnet = self.args.fitnet_loss_weight * sum(fitnet_losses)
            loss_gt = self.args.gt_loss_weight * self.criterion(logits_student, label)
            losses_dict = {
                "loss_gt": loss_gt,
                "loss_fitnet": loss_fitnet,
                "loss_kd": loss_kd
            }
        return logits_student, losses_dict


class FitNetCriterion(nn.Module):
    def __init__(self, size_s, size_t, vit_feat=None):
        super().__init__()
        C, H, W = size_t
        self.vit_feat = vit_feat
        if len(size_s) == 2:
            self.vit_feat = True
            L, Embed = size_s
            self.extra_token = get_extra_token(L)
            print("this is vit feat, the extra token is: ", self.extra_token, "!")
            self.h_s = int(math.sqrt(L))
            self.w_s = int(math.sqrt(L))
            self.channel_align = nn.Sequential(
                nn.Conv2d(Embed, C, kernel_size=3, stride=1, padding=1),
                nn.BatchNorm2d(C),
                nn.ReLU(inplace=True),
            )

        elif len(size_s) == 3:
            self.vit_feat = False
            print("this is cnn feat")
            self.c_s, self.h_s, self.w_s = size_s
            self.channel_align = nn.Sequential(
                nn.Conv2d(self.c_s, C, kernel_size=3, stride=1, padding=1),
                nn.BatchNorm2d(C),
                nn.ReLU(inplace=True),
            )

    def forward(self, x, feat_t=None):
        B = feat_t.size(0)
        if self.vit_feat:
            x = x[:, self.extra_token:]
            x = x.permute(0, 2, 1).contiguous()
            x = x.view(B, -1, self.h_s, self.w_s).contiguous()
        x = self.channel_align(x)
        feat_t = F.adaptive_avg_pool2d(feat_t, (self.h_s, self.w_s))
        loss = nn.MSELoss()(x, feat_t) / B
        return loss


def get_extra_token(L):
    extra_token = L - int(math.sqrt(L)) ** 2
    return extra_token
