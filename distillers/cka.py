import torch
import torch.nn.functional as F

from ._base import BaseDistiller
from .registry import register_distiller
from .utils import kd_loss
import math


def get_extra_token(L):
    extra_token = L - int(math.sqrt(L)) ** 2
    return extra_token


def cka_feature_loss(f_t, f_s, eps=1e-8, print_out=True):
    """
    Compute feature-based Centered Kernel Alignment (CKA) loss
    as described in 'Rethinking Centered Kernel Alignment in Knowledge Distillation' (Zhou et al., 2024).

    Returns:
        loss: scalar tensor (1 - CKA similarity)
    """
    # Flatten spatial dimensions
    size_s = f_s.size()
    B, C, H, W = f_t.shape
    if len(size_s) == 2:
        L, Embed = size_s
        extra_token = get_extra_token(L)
        if print_out:
            print("this is vit feat, the extra token is: ", extra_token, "!")
            print_out = False
        L = int(math.sqrt(L)) ** 2
        f_s = f_s[:, extra_token:]
        f_s = f_s.permute(0, 2, 1).contiguous()
        f_s = f_s.reshape(B, -1, L, L)
    else:
        print("this is cnn feat")

    f_s = f_s.reshape(B, -1)  # [B, C_s*H_s*W_s]
    f_t = f_t.reshape(B, -1)  # [B, C_t*H_t*W_t]

    # Center the features
    f_s = f_s - f_s.mean(dim=0, keepdim=True)
    f_t = f_t - f_t.mean(dim=0, keepdim=True)
    # Compute Gram matrices
    K_s = f_s @ f_s.T
    K_t = f_t @ f_t.T
    # Center the Gram matrices
    H_s = K_s - K_s.mean(dim=0, keepdim=True) - K_s.mean(dim=1, keepdim=True) + K_s.mean()
    H_t = K_t - K_t.mean(dim=0, keepdim=True) - K_t.mean(dim=1, keepdim=True) + K_t.mean()
    # Compute CKA similarity
    numerator = (H_s * H_t).sum()
    denominator = torch.sqrt((H_s ** 2).sum() * (H_t ** 2).sum() + eps)
    cka_value = numerator / denominator
    # Feature CKA loss
    loss = 1 - cka_value
    return loss


def CenterKernelAlignment(X, Y, with_l2_norm):
    """Compute the CKA similarity betweem samples"""
    # Compute Gram matrix
    gram_X = torch.matmul(X, X.t())
    gram_Y = torch.matmul(Y, Y.t())

    # l2 norm or not
    if with_l2_norm:
        gram_X = gram_X / torch.sqrt(torch.diag(gram_X)[:, None])
        gram_Y = gram_Y / torch.sqrt(torch.diag(gram_Y)[:, None])

    # compute cka
    cka = torch.trace(torch.matmul(gram_X, gram_Y.t())) / torch.sqrt(
        torch.trace(torch.matmul(gram_X, gram_X.t())) * torch.trace(torch.matmul(gram_Y, gram_Y.t()))
    )

    return cka


def cka_logit_loss(teacher_logits, student_logits, with_l2_norm=False):
    """Compute the CKA similarity between samples
    CKA computes similarity between batches
    input: (N, P) ----> output: (N, N) similarity matrix
    """
    N_t = teacher_logits.shape[0]
    N_s = student_logits.shape[0]
    assert N_s == N_t  # when use cka, you need to make sure N the same

    # get a similarity score between teacher and student
    similarity_martix = CenterKernelAlignment(teacher_logits, student_logits, with_l2_norm)

    # maximize the likelihood of it
    return -similarity_martix


@register_distiller
class CKA(BaseDistiller):
    requires_feat = False

    def __init__(self, student, teacher, criterion, args, **kwargs):
        super(CKA, self).__init__(student, teacher, criterion, args)

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
        # loss_cka_logit = self.args.cka_logitloss_weight * cka_logit_loss(logits_teacher, logits_student)
        loss_cka_feat = self.args.cka_featloss_weight * cka_feature_loss(feat_teacher[-2], feat_student[-2])
        loss_gt = self.args.gt_loss_weight * self.criterion(logits_student, label)
        losses_dict = {
            "loss_gt": loss_gt,
            # "loss_cka_logit": loss_cka_logit,
            "loss_cka_feat": loss_cka_feat,
            "loss_kd": loss_kd,
        }
        return logits_student, losses_dict
