import torch
import torch.nn.functional as F

from ._base import BaseDistiller
from .registry import register_distiller
from .utils import kd_loss

import torch
import torch.nn.functional as F


def rekd_loss(
        student_logits: torch.Tensor,
        teacher_logits: torch.Tensor,
        k: int = 7,
        T: float = 4.0,
        alpha: float = 1.0,
        beta: float = 1.0,
        reduction: str = "batchmean",  # 与 F.kl_div 一致: 'batchmean' | 'mean' | 'sum' | 'none'
):
    """
    ReKD (TISD + NTID) 蒸馏损失。
    只需提供教师/学生的 logits；若提供 ce_target，会同时计算 CE 损失。

    Args:
        student_logits: (B, C)
        teacher_logits: (B, C)
        k: Top-K（头部类别数）
        T: 温度
        alpha: TISD 权重
        beta: NTID 权重
        ce_target: (B,) 可选标签；提供则会计算交叉熵
        ce_weight: CE 权重系数
        reduction: KL 的 reduction 方式（默认 batchmean）

    Returns:
        dict 包含:
            'L_TISD', 'L_NTID', 'L_KD', 以及（若提供标签）'L_CE' 与 'L_total'
    """
    assert student_logits.shape == teacher_logits.shape, "student/teacher logits 形状必须一致"
    B, C = student_logits.shape
    if C < 2:
        raise ValueError("类别数 C 必须 >= 2")
    # 规范化 k
    k = max(1, min(int(k), C - 1))

    # --- 构造掩码 ---
    # non_topk_mask: 非Top-K处为1，Top-K处为0
    topk_idx = torch.topk(student_logits, k, dim=1).indices
    non_topk_mask = torch.ones_like(student_logits, dtype=student_logits.dtype)
    non_topk_mask.scatter_(1, topk_idx, 0.0)
    # topk_mask: Top-K处为1，其他为0
    topk_mask = 1.0 - non_topk_mask

    # --- TISD（仅头部类别）---
    # 通过给非Top-K位置减去一个大数来“屏蔽”它们
    s_topk = student_logits / T - 1000.0 * non_topk_mask
    t_topk = teacher_logits / T - 1000.0 * non_topk_mask
    logp_s_topk = F.log_softmax(s_topk, dim=1)
    p_t_topk = F.softmax(t_topk, dim=1)
    L_TISD = F.kl_div(logp_s_topk, p_t_topk, reduction=reduction) * (T * T)

    # --- NTID（仅尾部类别）---
    s_tail = student_logits / T - 1000.0 * topk_mask
    t_tail = teacher_logits / T - 1000.0 * topk_mask
    logp_s_tail = F.log_softmax(s_tail, dim=1)
    p_t_tail = F.softmax(t_tail, dim=1)
    L_NTID = F.kl_div(logp_s_tail, p_t_tail, reduction=reduction) * (T * T)

    # KD 总和
    loss = alpha * L_TISD + beta * L_NTID

    return loss


@register_distiller
class REKD(BaseDistiller):
    requires_feat = False

    def __init__(self, student, teacher, criterion, args, **kwargs):
        super(REKD, self).__init__(student, teacher, criterion, args)

    def forward(self, image, label, *args, **kwargs):
        with torch.no_grad():
            self.teacher.eval()
            logits_teacher = self.teacher(image)
        logits_student = self.student(image)
        if len(logits_student) == 2:
            logits_student, logits_student_dist = logits_student
            loss_kd = self.args.rekd_loss_weight * rekd_loss(logits_student_dist, logits_teacher)
        else:
            loss_kd = self.args.rekd_loss_weight * rekd_loss(logits_student, logits_teacher)
        loss_gt = self.args.gt_loss_weight * self.criterion(logits_student, label)
        losses_dict = {
            "loss_gt": loss_gt,
            "loss_kd": loss_kd,
        }
        return logits_student, losses_dict
