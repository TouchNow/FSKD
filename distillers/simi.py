import torch
import torch.nn.functional as F
from torch import nn

from ._base import BaseDistiller
from .registry import register_distiller
from .utils import get_module_dict, init_weights, is_cnn_model, set_module_dict
import math
from einops import rearrange
from .utils import kd_loss
import torchsort


def get_extra_token(L):
    extra_token = L - int(math.sqrt(L)) ** 2
    return extra_token


@register_distiller
class SimiKD(BaseDistiller):
    requires_feat = True

    def __init__(self, student, teacher, criterion, args, **kwargs):
        super(SimiKD, self).__init__(student, teacher, criterion, args)
        self.projector = nn.ModuleDict()
        for stage in self.args.simi_stage:
            _, size_s = self.student.stage_info(stage)
            _, size_t = self.teacher.stage_info(stage)
            projector = FeatureAlignment(args, size_s, size_t)
            set_module_dict(self.projector, stage, projector)
        self.projector.apply(init_weights)

    def forward(self, image, label, *args, **kwargs):
        with torch.no_grad():
            self.teacher.eval()
            logits_teacher, feat_teacher = self.teacher(image, requires_feat=True)
        logits_student, feat_student, attn_list = self.student(image, requires_feat=True, requires_attn=True)
        if len(logits_student) == 2:
            logits_student, logits_student_dist = logits_student
            loss_kd = self.args.kd_loss_weight * kd_loss(logits_student_dist, logits_teacher, self.args.kd_temperature)
        else:
            loss_kd = self.args.kd_loss_weight * kd_loss(logits_student, logits_teacher, self.args.kd_temperature)
        global_losses = []
        patch_losses = []
        for stage in self.args.simi_stage:
            idx_s, _ = self.student.stage_info(stage)
            idx_t, _ = self.teacher.stage_info(stage)
            # patch_losses.append(self.get_patch_loss(feat_student[idx_s], feat_teacher[idx_t]))
            feat_s = get_module_dict(self.projector, stage)(feat_student[idx_s], feat_teacher[idx_t])
            global_losses.append(F.mse_loss(feat_s, feat_teacher[idx_t]) / feat_teacher[idx_t].shape[0])
            if len(feat_student[idx_s].shape) == 3:
                h = int(math.sqrt(feat_student[idx_s].size(1)))
                resize_feat_t = F.interpolate(feat_teacher[idx_t], (h, h), mode='bilinear', align_corners=False)
            else:
                h, w = feat_student[idx_s].size(2), feat_student[idx_s].size(3)
                resize_feat_t = F.interpolate(feat_teacher[idx_t], (h, w), mode='bilinear', align_corners=False)
            patch_losses.append(self.get_norepatch_loss(feat_student[idx_s], resize_feat_t))

        loss_attn = self.args.simi_attn_weight * self.get_patch_attn_loss(feat_teacher, attn_list[-1])
        loss_global = self.args.simi_global_weight * sum(global_losses)
        loss_patch = self.args.simi_patch_weight * sum(patch_losses)
        # loss_global = torch.tensor(0.0).cuda()
        # loss_patch = torch.tensor(0.0).cuda()
        loss_gt = self.args.gt_loss_weight * self.criterion(logits_student, label)

        losses_dict = {
            "loss_gt": loss_gt,
            "loss_global": loss_global,
            "loss_patch": loss_patch,
            "loss_attn": loss_attn,
            "loss_kd": loss_kd,
        }
        return logits_student, losses_dict

    def get_norepatch_loss(self, feat_s, feat_t):
        if len(feat_s.shape) == 3:
            B, L, Embed = feat_s.shape
            extra_token = get_extra_token(L)
            feat_s = feat_s[:, extra_token:]
        else:
            feat_s = feat_s.flatten(2).transpose(1, 2)
        feat_t = feat_t.flatten(2).transpose(1, 2)
        feat_t = feat_t / feat_t.norm(dim=-1, keepdim=True)
        feat_s = feat_s / feat_s.norm(dim=-1, keepdim=True)
        score_s = feat_s @ feat_s.transpose(-1, -2)
        score_t = feat_t @ feat_t.transpose(-1, -2)
        loss = F.mse_loss(score_s, score_t)
        return loss

    def get_patch_attn_loss(self, teacher_feat, student_attn):
        L = student_attn.shape[-1]
        extra_token = get_extra_token(L)
        L = int(math.sqrt(L))
        teacher_feat_last = F.interpolate(
            teacher_feat[3], size=(L, L), mode='bilinear', align_corners=False
        )  # [B, C, grid_size[0], grid_size[1]]
        teacher_feat_last = teacher_feat_last.pow(2).mean(dim=1)  # [B, grid_size[0], grid_size[1]]
        teacher_feat_last = teacher_feat_last.flatten(1)  # [B, L]
        t_importance = F.softmax(teacher_feat_last, dim=-1)  # [B, L]
        t_ranks = torchsort.soft_rank(t_importance, regularization_strength=self.args.regu)
        #cls_attn = student_attn[:, :, 0, extra_token:]  # [B, num_heads, L]
        #cls_attn = cls_attn.mean(dim=1)  # [B, L]
        if extra_token > 0:
            # 有 CLS token：取 CLS 对空间 token 的注意力
            s_importance = student_attn[:, :, 0, extra_token:].mean(dim=1)
        else:
            # 无 CLS token（如 PVTv2）：计算各空间位置被关注的平均程度
            # [B, num_heads, N_query, N_key] -> [B, N_key]
            s_importance = student_attn.mean(dim=(1, 2))
        s_ranks = torchsort.soft_rank(s_importance, regularization_strength=self.args.regu)
        rank_diff = (t_ranks - s_ranks) ** 2
        # Spearman损失
        n = t_importance.size(1)  # L
        spearman_corr = 1 - (6 * rank_diff.sum(dim=1)) / (n * (n ** 2 - 1))
        # 转换为最小化问题
        loss_spearman = 1 - spearman_corr.mean()
        # loss_spearman = loss_spearman * (n * (n ** 2 - 1))
        # loss_spearman = (1 - spearman_corr).sum()
        return loss_spearman

    def get_patch_loss(self, feat_s, feat_t):
        B, L, C = feat_s.shape
        H = int(math.sqrt(L))
        extra_token = get_extra_token(L)
        feat_s = feat_s[:, extra_token:, :]

        feat_s = feat_s.permute(0, 2, 1).contiguous()
        feat_s = feat_s.view(B, C, H, H).contiguous()
        _, _, h, w = feat_t.shape
        feat_s = F.interpolate(feat_s, size=(h, w), mode='bilinear', align_corners=False)
        feat_s = feat_s.flatten(2).transpose(1, 2)

        # feat_t = F.interpolate(feat_t, size=(H, H), mode='bilinear', align_corners=False)
        feat_t = feat_t.flatten(2).transpose(1, 2)
        feat_t = feat_t / feat_t.norm(dim=-1, keepdim=True)
        feat_s = feat_s / feat_s.norm(dim=-1, keepdim=True)
        score_s = feat_s @ feat_s.transpose(-1, -2)
        score_t = feat_t @ feat_t.transpose(-1, -2)
        loss = F.mse_loss(score_s, score_t)
        return loss


# class FeatureAlignment(nn.Module):
#     def __init__(self, size_s, size_t, vit_feat=None):
#         super().__init__()
#         C, H, W = size_t
#         self.vit_feat = vit_feat
#         if len(size_s) == 2:
#             self.vit_feat = True
#             L, Embeding = size_s
#             self.extra_token = get_extra_token(L)
#             print("this is vit feat, the extra token is: ", self.extra_token, "!")
#             # L = int(math.sqrt(L)) ** 2
#             self.h = int(math.sqrt(L))
#             self.w = int(math.sqrt(L))
#         elif len(size_s) == 3:
#             self.vit_feat = False
#             print("this is cnn feat")
#             Embeding, self.h, self.w = size_s
#         self.channel_align = nn.Sequential(
#             nn.Conv2d(Embeding, C // 2, kernel_size=3, stride=1, padding=1),
#             nn.GELU(),
#             nn.Conv2d(C // 2, C, kernel_size=3, stride=1, padding=1),
#         )
#
#     def forward(self, x, feat_t=None):
#         B = feat_t.size(0)
#         feat_t = F.interpolate(feat_t, (self.h, self.w), mode='bilinear', align_corners=False)
#         if self.vit_feat:
#             x = x[:, self.extra_token:]
#             x = x.permute(0, 2, 1).contiguous()
#             x = x.view(B, -1, self.h, self.w).contiguous()
#         x = self.channel_align(x)  # b c h w
#         return x, feat_t


class FeatureAlignment(nn.Module):
    def __init__(self, args, size_s, size_t, vit_feat=None):
        super().__init__()
        C, H, W = size_t
        self.vit_feat = vit_feat
        if len(size_s) == 2:
            self.vit_feat = True
            L, Embed = size_s
            self.extra_token = get_extra_token(L)
            print("this is vit feat, the extra token is: ", self.extra_token, "!")
            L = int(math.sqrt(L)) ** 2
            if args.channel_align == "c":
                self.channel_align = nn.Sequential(
                    nn.Conv2d(Embed, C, kernel_size=3, stride=1, padding=1),
                )
            elif args.channel_align == "cr":
                self.channel_align = nn.Sequential(
                    nn.Conv2d(Embed, C, kernel_size=3, stride=1, padding=1),
                    nn.ReLU(),
                )
            elif args.channel_align == "crc":
                self.channel_align = nn.Sequential(
                    nn.Conv2d(Embed, C // 2, kernel_size=3, stride=1, padding=1),
                    nn.ReLU(),
                    nn.Conv2d(C // 2, C, kernel_size=3, stride=1, padding=1),
                )
            elif args.channel_align == "cg":
                self.channel_align = nn.Sequential(
                    nn.Conv2d(Embed, C, kernel_size=3, stride=1, padding=1),
                    nn.GELU(),
                )
            elif args.channel_align == "cgc":
                self.channel_align = nn.Sequential(
                    nn.Conv2d(Embed, C // 2, kernel_size=3, stride=1, padding=1),
                    nn.GELU(),
                    nn.Conv2d(C // 2, C, kernel_size=3, stride=1, padding=1),
                )
            if args.patch_align == "l":
                self.spatial_align = nn.Sequential(
                    nn.Linear(L, H * W),
                )
            elif args.patch_align == "lr":
                self.spatial_align = nn.Sequential(
                    nn.Linear(L, H * W),
                    nn.ReLU(),
                )
            elif args.patch_align == "lrl":
                self.spatial_align = nn.Sequential(
                    nn.Linear(L, L // 2),
                    nn.ReLU(),
                    nn.Linear(L // 2, H * W),
                )
            elif args.patch_align == "lgl":
                self.spatial_align = nn.Sequential(
                    nn.Linear(L, L // 2),
                    nn.GELU(),
                    nn.Linear(L // 2, H * W),
                )
            elif args.patch_align == "lg":
                self.spatial_align = nn.Sequential(
                    nn.Linear(L, H * W),
                    nn.GELU(),
                )
        elif len(size_s) == 3:
            self.vit_feat = False
            print("this is cnn feat")
            c_s, h_s, w_s = size_s
            if args.channel_align == "c":
                self.channel_align = nn.Sequential(
                    nn.Conv2d(c_s, C, kernel_size=3, stride=1, padding=1),
                )
            elif args.channel_align == "cr":
                self.channel_align = nn.Sequential(
                    nn.Conv2d(c_s, C, kernel_size=3, stride=1, padding=1),
                    nn.ReLU(),
                )
            elif args.channel_align == "crc":
                self.channel_align = nn.Sequential(
                    nn.Conv2d(c_s, C // 2, kernel_size=3, stride=1, padding=1),
                    nn.ReLU(),
                    nn.Conv2d(C // 2, C, kernel_size=3, stride=1, padding=1),
                )
            elif args.channel_align == "cg":
                self.channel_align = nn.Sequential(
                    nn.Conv2d(c_s, C, kernel_size=3, stride=1, padding=1),
                    nn.GELU(),
                )
            elif args.channel_align == "cgc":
                self.channel_align = nn.Sequential(
                    nn.Conv2d(c_s, C // 2, kernel_size=3, stride=1, padding=1),
                    nn.GELU(),
                    nn.Conv2d(C // 2, C, kernel_size=3, stride=1, padding=1),
                )
            if args.patch_align == "l":
                self.spatial_align = nn.Sequential(
                    nn.Linear(h_s * w_s, H * W),
                )
            elif args.patch_align == "lr":
                self.spatial_align = nn.Sequential(
                    nn.Linear(h_s * w_s, H * W),
                    nn.ReLU(),
                )
            elif args.patch_align == "lrl":
                self.spatial_align = nn.Sequential(
                    nn.Linear(h_s * w_s, h_s * w_s // 2),
                    nn.ReLU(),
                    nn.Linear(h_s * w_s // 2, H * W),
                )
            elif args.patch_align == "lgl":
                self.spatial_align = nn.Sequential(
                    nn.Linear(h_s * w_s, h_s * w_s // 2),
                    nn.GELU(),
                    nn.Linear(h_s * w_s // 2, H * W),
                )
            elif args.patch_align == "lg":
                self.spatial_align = nn.Sequential(
                    nn.Linear(h_s * w_s, H * W),
                    nn.GELU(),
                )

    def forward(self, x, feat_t=None):
        B, C, H, W = feat_t.shape
        if self.vit_feat:
            x = x[:, self.extra_token:]
            x = x.permute(0, 2, 1).contiguous()
        elif not self.vit_feat:
            x = x.flatten(2)
        x = self.spatial_align(x)
        x = x.view(B, -1, H, W).contiguous()
        x = self.channel_align(x)  # b c h w
        return x


class AttnAlignment(nn.Module):
    def __init__(self, size_s, size_t):
        super().__init__()
        L, Embed = size_s
        self.extra_token = get_extra_token(L)
        L = int(math.sqrt(L)) ** 2
        C, H, W = size_t
        self.attn_align = nn.Sequential(
            nn.Linear(L, L // 2),
            nn.GELU(),
            nn.Linear(L // 2, H * W),
        )

    def forward(self, x, feat_t=None):
        x = x[:, :, 0, self.extra_token:]  # [B, num_heads, L]
        x = x.mean(dim=1)  # [B, L]
        x = self.attn_align(x)  # [B, H * W]
        feat_t = feat_t.pow(2).mean(dim=1)
        feat_t = feat_t.view(feat_t.shape[0], -1).contiguous()
        # feat_t = (feat_t - feat_t.mean(dim=1, keepdim=True)) / feat_t.std(dim=1, keepdim=True)
        # x = (x - x.mean(dim=1, keepdim=True)) / x.std(dim=1, keepdim=True)
        return x, feat_t
