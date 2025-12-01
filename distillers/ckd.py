import torch
import torch.nn.functional as F

from ._base import BaseDistiller
from .registry import register_distiller
from .utils import kd_loss


def ckd_info_loss(logits_teacher, logits_student, temperature=1.0):
    B = logits_teacher.shape[0]
    device = logits_teacher.device
    logits_teacher = F.normalize(logits_teacher, dim=1)
    logits_student = F.normalize(logits_student, dim=1)
    similarity = logits_teacher @ logits_student.t()
    target = torch.arange(B).to(device)
    contrastive_loss = F.cross_entropy(similarity / temperature, target)
    return contrastive_loss


@register_distiller
class CKD(BaseDistiller):
    requires_feat = False

    def __init__(self, student, teacher, criterion, args, **kwargs):
        super(CKD, self).__init__(student, teacher, criterion, args)

    def forward(self, image, label, *args, **kwargs):
        with torch.no_grad():
            self.teacher.eval()
            logits_teacher = self.teacher(image)
        logits_student = self.student(image)
        if len(logits_student) == 2:
            logits_student, logits_student_dist = logits_student
            loss_kd = self.args.kd_loss_weight * kd_loss(logits_student_dist, logits_teacher, self.args.kd_temperature)
            loss_kd = loss_kd + self.args.ckd_loss_weight * ckd_info_loss(logits_student_dist, logits_teacher)
        else:
            loss_kd = self.args.kd_loss_weight * kd_loss(logits_student, logits_teacher, self.args.kd_temperature)
            loss_kd = loss_kd + self.args.ckd_loss_weight * ckd_info_loss(logits_student, logits_teacher)
        loss_gt = self.args.gt_loss_weight * self.criterion(logits_student, label)
        losses_dict = {
            "loss_gt": loss_gt,
            "loss_kd": loss_kd,
        }
        return logits_student, losses_dict
