import torch
import torch.nn as nn
import numpy as np


class CELoss(nn.Module):

    def __init__(self):
        super(CELoss, self).__init__()
        self.loss = nn.CrossEntropyLoss()

    def forward(self, input, target, tb_tools):
        total_loss = self.loss(input, target)

        if tb_tools['local_rank'] == 0:
            acc1, acc5 = accuracy(input, target, topk=(1, 5))
            tb_tools['tb_writer'].add_scalar('{}/acc1_cls'.format(tb_tools['prefix']), acc1[0], global_step=tb_tools['global_step'])
            tb_tools['tb_writer'].add_scalar('{}/acc5_cls'.format(tb_tools['prefix']), acc5[0], global_step=tb_tools['global_step'])

        return total_loss


def accuracy(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res
