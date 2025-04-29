import torch
import torch.nn as nn
import torch.nn.functional as F


class FocalLoss(nn.Module):
    def __init__(self, gamma=2, alpha=None, reduction='mean'):
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.alpha = alpha
        self.reduction = reduction

    def forward(self, input, target):
        ce_loss = F.cross_entropy(input, target, reduction='none')
        pt = torch.exp(-ce_loss)
        focal_loss = (1 - pt) ** self.gamma * ce_loss
        if self.alpha is not None:
            focal_loss = self.alpha[target] * focal_loss
        if self.reduction == 'mean':
            return torch.mean(focal_loss)
        elif self.reduction == 'sum':
            return torch.sum(focal_loss)
        else:
            return focal_loss



class CustomPunitiveLoss(nn.Module):
    def __init__(self):
        super(CustomPunitiveLoss, self).__init__()

    def forward(self, input, target):
        log_probs = F.log_softmax(input, dim=1)
        nll_loss = F.nll_loss(log_probs, target, reduction='none')
        probs = torch.exp(log_probs)
        punishment = torch.zeros_like(nll_loss)
        for i in range(input.size(0)):
            for j in range(input.size(1)):
                if j != target[i]:
                    punishment[i] += (1 - probs[i, j]) ** 2 # Penalize low probability for incorrect classes

        loss = nll_loss + 0.1 * punishment # Combine standard NLL with the punishment term
        return torch.mean(loss)


