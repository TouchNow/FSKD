import torch
import torch.nn.functional as F

from ._base import BaseDistiller
from .registry import register_distiller


@register_distiller
class NKD(BaseDistiller):
    requires_feat = False

    def __init__(self, student, teacher, criterion, args, **kwargs):
        super(NKD, self).__init__(student, teacher, criterion, args)

    def forward(self, image, label, *args, **kwargs):
        with torch.no_grad():
            self.teacher.eval()
            logits_teacher = self.teacher(image)

        logits_student = self.student(image)

        if len(label.shape) == 2:  # mixup / smoothing
            target = label.max(1)[1]
        else:
            target = label
        if len(logits_student) == 2:
            logits_student, logits_student_dist = logits_student
            if self.args.lcskd:
                loss_kd = self.args.kd_loss_weight * ltfnkd_loss(logits_student_dist, logits_teacher, target,
                                                                 self.args.nkd_temperature, self.args.step,
                                                                 self.args.window, self.args.nkd_gamma)
            else:
                loss_kd = self.args.kd_loss_weight * nkd_loss(logits_student_dist, logits_teacher, target,
                                                              self.args.nkd_temperature, self.args.nkd_gamma)
        else:
            if self.args.lcskd:
                loss_kd = self.args.kd_loss_weight * ltfnkd_loss(logits_student, logits_teacher, target,
                                                                 self.args.nkd_temperature, self.args.step,
                                                                 self.args.window, self.args.nkd_gamma)
            else:
                loss_kd = self.args.kd_loss_weight * nkd_loss(logits_student, logits_teacher, target,
                                                              self.args.nkd_temperature, self.args.nkd_gamma)
        loss_gt = self.args.gt_loss_weight * self.criterion(logits_student, label)
        losses_dict = {
            "loss_gt": loss_gt,
            "loss_kd": loss_kd,
        }
        return logits_student, losses_dict


def nkd_loss(logit_s, logit_t, gt_label, temp, gamma=3.5):
    if len(gt_label.size()) > 1:
        label = torch.max(gt_label, dim=1, keepdim=True)[1]
    else:
        label = gt_label.view(len(gt_label), 1)

        # N*class
    N, c = logit_s.shape
    s_i = F.log_softmax(logit_s, dim=1)
    t_i = F.softmax(logit_t, dim=1)
    # N*1
    s_t = torch.gather(s_i, 1, label)
    t_t = torch.gather(t_i, 1, label).detach()

    loss_t = - (t_t * s_t).mean()

    mask = torch.ones_like(logit_s).scatter_(1, label, 0).bool()
    logit_s = logit_s[mask].reshape(N, -1)
    logit_t = logit_t[mask].reshape(N, -1)

    # N*class
    S_i = F.log_softmax(logit_s / temp, dim=1)
    T_i = F.softmax(logit_t / temp, dim=1)

    loss_non = (T_i * S_i).sum(dim=1).mean()
    loss_non = - gamma * (temp ** 2) * loss_non

    return loss_t + loss_non


def ltf_nonloss(logits_student_in, logits_teacher_all, logits_teacher_in, temperature, step, window):
    loss_kd = 0
    batch_size, num_classes = logits_student_in.size()
    # # sort
    # logits_student_in, sorted_indices = torch.sort(logits_student_in, dim=1, descending=True)
    # batch_indices = torch.arange(batch_size).unsqueeze(-1)
    # logits_teacher_in = logits_teacher_in[batch_indices, sorted_indices]

    logit_t_std = torch.std(logits_teacher_all.clone().detach(), dim=1, keepdim=True)
    for i in range(0, num_classes, window):
        window_student_logits = logits_student_in[:, i:i + window]
        window_teacher_logits = logits_teacher_in[:, i:i + window]
        window_teacher_logits_std = torch.std(window_teacher_logits.clone().detach(), dim=1, keepdim=True)
        window_T = torch.tanh(-1 / window_teacher_logits_std + 1 / logit_t_std) * step + temperature
        log_pred_student = F.log_softmax(window_student_logits / window_T, dim=1)
        pred_teacher = F.softmax(window_teacher_logits / window_T, dim=1)
        loss_kd += -((pred_teacher * log_pred_student).sum(dim=1).mean()) * (temperature ** 2)
        # loss_kd += -((pred_teacher * log_pred_student).sum(dim=1).mean()) * (window_T ** 2)

    return loss_kd


def ltfnkd_loss(logit_s, logit_t, gt_label, temp, step, window, gamma):
    if len(gt_label.size()) > 1:
        label = torch.max(gt_label, dim=1, keepdim=True)[1]
    else:
        label = gt_label.view(len(gt_label), 1)

        # N*class
    N, c = logit_s.shape
    s_i = F.log_softmax(logit_s, dim=1)
    t_i = F.softmax(logit_t, dim=1)
    # N*1
    s_t = torch.gather(s_i, 1, label)
    t_t = torch.gather(t_i, 1, label).detach()

    loss_t = - (t_t * s_t).mean()

    mask = torch.ones_like(logit_s).scatter_(1, label, 0).bool()
    logit_s_non = logit_s[mask].reshape(N, -1)
    logit_t_non = logit_t[mask].reshape(N, -1)

    # N*class
    loss_non = gamma * ltf_nonloss(logit_s_non, logit_t, logit_t_non, temp, step, window)

    return loss_t + loss_non
