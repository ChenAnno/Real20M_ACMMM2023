import torch
import torch.nn as nn
import torch.nn.functional as F


class RankLoss(nn.Module):
    
    def __init__(self, weight=None, margin_pos=1, margin_neg=0):
        super(RankLoss, self).__init__()
        self.weight = weight
        self.reduction = 'mean'
        self.margin_pos = margin_pos
        self.margin_neg = margin_neg
        
    def forward(self, inputs, targets):
        log_prob = F.log_softmax(inputs, dim=-1)

        with torch.no_grad():
            prob = torch.exp(log_prob)[:, 1]
            #ignore = targets*prob.gt(self.margin_pos).long() + (1-targets)*prob.lt(self.margin_neg).long()
            ignore = prob.gt(self.margin_pos).long() + prob.lt(self.margin_neg).long()
            ignore = ignore.view(-1, 1).float()

        return F.nll_loss(
            (1 - ignore) * log_prob, 
            targets, 
            weight=self.weight,
            reduction=self.reduction
        )
    